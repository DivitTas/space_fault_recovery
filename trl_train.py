from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from trl import GRPOConfig, GRPOTrainer

from training.action_space import ActionSpec, build_action_space
from training.openenv_compat import ensure_training_runtime

ensure_training_runtime()

from models import SpaceFaultAction  # noqa: E402
from server.space_fault_recovery_environment import (  # noqa: E402
    MAX_STEPS,
    SpaceFaultRecoveryEnvironment,
)

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
FALLBACK_ACTION = "diagnostic_scan:power"

_ACTION_SPACE: list[ActionSpec] = build_action_space()
_ACTION_LABELS: list[str] = [a.label for a in _ACTION_SPACE]
_ACTION_LOOKUP: dict[str, ActionSpec] = {a.label: a for a in _ACTION_SPACE}

_MODEL: AutoModelForCausalLM | None = None
_TOKENIZER: AutoTokenizer | None = None

# Expert sequence used as the continuation policy inside _rollout_reward.
# After the model's first action, this scripted policy plays out the rest
# of the episode. This makes reward_fn pure-Python (<1ms per episode) while
# still giving a meaningful multi-step signal: if the model's first action
# was correct, the expert can capitalise; if it was bad, recovery stalls.
_EXPERT_CONTINUATION: tuple[str, ...] = (
    "diagnostic_scan:power",
    "query_power_level:battery",
    "query_power_level:solar_a",
    "query_power_level:solar_b",
    "diagnostic_scan:attitude",
    "query_attitude",
    "query_thermal",
    "diagnostic_scan:comms",
    "cross_validate_attitude",
    "shed_load:science_a",
    "shed_load:science_b",
    "reset_power_controller",
    "recalibrate_star_tracker",
    "desaturate_wheels",
    "desaturate_wheels",
    "recalibrate_imu",
    "reconfigure_power:solar_a",
    "reconfigure_power:solar_b",
    "restore_load:heaters",
    "shed_load:transponder",
    "restore_load:transponder",
    "stabilize_attitude",
    "stabilize_attitude",
    "stabilize_attitude",
    "query_attitude",
    "query_thermal",
    "query_power_level:battery",
    "resume_nominal",
    "resume_nominal",
    "resume_nominal",
)

# Diagnostic steps used to fast-forward into mid-episode training states.
# The agent sees richer last_action_result text at these states, which is
# exactly where the hard repair decisions happen.
_DIAGNOSTIC_PREFIX: tuple[str, ...] = (
    "diagnostic_scan:power",
    "query_power_level:battery",
    "query_power_level:solar_a",
    "query_power_level:solar_b",
    "diagnostic_scan:attitude",
    "query_attitude",
    "query_thermal",
    "diagnostic_scan:comms",
    "cross_validate_attitude",
    "shed_load:science_a",
    "shed_load:science_b",
    "recalibrate_star_tracker",
    "desaturate_wheels",
    "stabilize_attitude",
    "reconfigure_power:solar_a",
)


# ── text extraction ──────────────────────────────────────────────────────────

def _extract_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("content", "text"):
            if key in value and isinstance(value[key], str):
                return value[key]
    if isinstance(value, list):
        parts = [_extract_text(item) for item in value]
        return "\n".join(p for p in parts if p)
    return str(value)


# ── prompt construction ──────────────────────────────────────────────────────

def obs_to_prompt(obs: Any, *, seed: int) -> str:
    return (
        "You are a spacecraft fault-recovery controller.\n"
        "Choose exactly one valid action for the CURRENT telemetry.\n"
        f"Episode seed: {seed}\n"
        f"Step: {int(getattr(obs, 'step', 0))}\n"
        f"Battery: {float(getattr(obs, 'battery_pct', 0.0)):.2f}\n"
        f"Battery drain rate: {float(getattr(obs, 'battery_drain_rate', 0.0)):.3f}\n"
        f"Solar A output W: {float(getattr(obs, 'solar_a_sensor_output_w', 0.0)):.2f}\n"
        f"Solar B output W: {float(getattr(obs, 'solar_b_sensor_output_w', 0.0)):.2f}\n"
        f"Bus voltage: {float(getattr(obs, 'bus_voltage', 0.0)):.2f}\n"
        f"Star tracker err deg: {float(getattr(obs, 'star_tracker_deg', 0.0)):.3f}\n"
        f"Gyro err deg: {float(getattr(obs, 'gyro_deg', 0.0)):.3f}\n"
        f"Sun sensor err deg: {float(getattr(obs, 'sun_sensor_deg', 0.0)):.3f}\n"
        f"Fuel units: {float(getattr(obs, 'fuel_units', 0.0)):.2f}\n"
        f"Signal strength db: {float(getattr(obs, 'signal_strength_db', 0.0)):.2f}\n"
        f"Battery temp C: {float(getattr(obs, 'battery_temp_c', 0.0)):.2f}\n"
        f"Attitude mode: {getattr(obs, 'attitude_mode', 'unknown')}\n"
        f"RW status: {getattr(obs, 'rw_status', 'unknown')}\n"
        f"Transponder: {getattr(obs, 'transponder_status', 'unknown')}\n"
        f"Link bandwidth: {getattr(obs, 'link_bandwidth', 'unknown')}\n"
        f"Mission status: {getattr(obs, 'mission_status', 'unknown')}\n"
        f"Subsystems online: {', '.join(getattr(obs, 'subsystems_online', []) or [])}\n"
        f"Last action result: {getattr(obs, 'last_action_result', 'none')}\n"
        f"Valid actions: {', '.join(_ACTION_LABELS)}\n"
        "Respond with exactly one action label and no explanation."
    )


# ── dataset construction ─────────────────────────────────────────────────────

def _build_episode_sample(
    env: SpaceFaultRecoveryEnvironment,
    seed: int,
    prefix_length: int,
) -> tuple[str, int, list[str]] | None:
    """Reset env, replay prefix_length diagnostic steps, return (prompt, seed, labels).

    Returns None when the episode ends during the prefix — no useful training
    signal can be computed for a terminal state."""
    obs = env.reset(seed=seed)
    prefix_labels: list[str] = []

    for i in range(prefix_length):
        if bool(getattr(obs, "done", False)):
            return None
        label = _DIAGNOSTIC_PREFIX[i % len(_DIAGNOSTIC_PREFIX)]
        spec = _ACTION_LOOKUP.get(label, _ACTION_LOOKUP[FALLBACK_ACTION])
        obs = env.step(spec.to_action())
        prefix_labels.append(label)

    if bool(getattr(obs, "done", False)):
        return None

    return obs_to_prompt(obs, seed=seed), seed, prefix_labels


def build_prompt_dataset(*, num_prompts: int, seed_base: int) -> Dataset:
    """Build a dataset mixing step-0, early-episode, and mid-episode states.

    Phase split:
      40% — step-0 observations (fault just injected, full episode ahead)
      30% — after 5 diagnostic steps (agent has some telemetry context)
      30% — after 10 diagnostic steps (nearing decision point for repair)
    """
    env = SpaceFaultRecoveryEnvironment()
    prompts: list[str] = []
    episode_seeds: list[int] = []
    prefix_actions_json: list[str] = []

    n_reset = int(num_prompts * 0.40)
    n_early = int(num_prompts * 0.30)
    n_mid = num_prompts - n_reset - n_early

    phase_configs = [
        (n_reset, 0),
        (n_early, 5),
        (n_mid, 10),
    ]

    idx = 0
    for count, prefix_len in phase_configs:
        collected = 0
        while collected < count:
            seed = seed_base + idx
            idx += 1
            result = _build_episode_sample(env, seed, prefix_len)
            if result is None:
                continue
            prompt_text, ep_seed, prefix_labels = result
            prompts.append(prompt_text)
            episode_seeds.append(ep_seed)
            prefix_actions_json.append(json.dumps(prefix_labels))
            collected += 1

    n_step0 = sum(1 for x in prefix_actions_json if x == "[]")
    print(
        f"Dataset: {len(prompts)} prompts "
        f"({n_step0} step-0, {len(prompts) - n_step0} mid-episode)"
    )
    return Dataset.from_dict(
        {
            "prompt": prompts,
            "episode_seed": episode_seeds,
            "prefix_actions_json": prefix_actions_json,
        }
    )


# ── action parsing ───────────────────────────────────────────────────────────

def parse_action(text: str) -> ActionSpec:
    norm = _extract_text(text).strip().lower()
    for label in sorted(_ACTION_LABELS, key=len, reverse=True):
        if label in norm:
            return _ACTION_LOOKUP[label]
    return _ACTION_LOOKUP[FALLBACK_ACTION]


def _action_from_model(prompt: str) -> ActionSpec:
    if _MODEL is None or _TOKENIZER is None:
        return _ACTION_LOOKUP[FALLBACK_ACTION]

    was_training = _MODEL.training
    _MODEL.eval()
    try:
        encoded = _TOKENIZER(prompt, return_tensors="pt")
        encoded = {k: v.to(_MODEL.device) for k, v in encoded.items()}
        with torch.no_grad():
            generated = _MODEL.generate(
                **encoded,
                max_new_tokens=64,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                pad_token_id=_TOKENIZER.eos_token_id,
            )
        new_tokens = generated[0][encoded["input_ids"].shape[-1]:]
        text = _TOKENIZER.decode(new_tokens, skip_special_tokens=True)
    finally:
        if was_training:
            _MODEL.train()

    return parse_action(text)


def _seed_from_prompt(prompt: str, fallback_seed: int = 0) -> int:
    match = re.search(r"Episode seed:\s*(\d+)", prompt)
    if match:
        return int(match.group(1))
    return fallback_seed


# ── rollout reward ───────────────────────────────────────────────────────────

def _rollout_reward(
    seed: int,
    first_action: ActionSpec,
    prefix_actions: list[str],
) -> float:
    """Run a full episode, returning total reward attributable to the agent.

    1. Reset with seed.
    2. Replay prefix_actions (diagnostic warmup, reward NOT counted).
    3. Apply first_action (the GRPO completion).
    4. Continue with _action_from_model() until done/timeout.
    5. Add terminal bonus/penalty for recovered/lost.
    """
    env = SpaceFaultRecoveryEnvironment()
    obs = env.reset(seed=seed)
    total_reward = 0.0

    for label in prefix_actions:
        if bool(getattr(obs, "done", False)):
            return 0.0
        spec = _ACTION_LOOKUP.get(label, _ACTION_LOOKUP[FALLBACK_ACTION])
        obs = env.step(spec.to_action())

    if bool(getattr(obs, "done", False)):
        return 0.0

    obs = env.step(first_action.to_action())
    total_reward += float(obs.reward or 0.0)

    # Scripted expert continuation — no LLM calls here.
    # Using _action_from_model for 40+ continuation steps per completion
    # would take ~320 LLM calls per training step (~16 hours for 500 steps).
    # The scripted policy gives a clean, fast reward signal: a good first
    # action enables expert recovery; a bad one prevents it.
    cont_step = 0
    while not bool(getattr(obs, "done", False)) and int(getattr(obs, "step", 0)) < MAX_STEPS:
        label = _EXPERT_CONTINUATION[cont_step % len(_EXPERT_CONTINUATION)]
        action = _ACTION_LOOKUP.get(label, _ACTION_LOOKUP[FALLBACK_ACTION])
        obs = env.step(action.to_action())
        total_reward += float(obs.reward or 0.0)
        cont_step += 1

    status = str(getattr(obs, "mission_status", ""))
    if status == "recovered":
        total_reward += 5.0
    elif status == "lost":
        total_reward -= 5.0
    return total_reward


def reward_fn(
    completions: list[Any],
    prompts: list[Any] | None = None,
    episode_seed: list[Any] | None = None,
    prefix_actions_json: list[Any] | None = None,
    **kwargs: Any,
) -> list[float]:
    prompt_values = (
        prompts
        or kwargs.get("prompt")
        or kwargs.get("prompts")
        or ([""] * len(completions))
    )

    # TRL may pass extra columns as positional kwargs
    seeds_col = episode_seed or kwargs.get("episode_seed")
    prefixes_col = prefix_actions_json or kwargs.get("prefix_actions_json")

    rewards: list[float] = []
    for idx, completion in enumerate(completions):
        prompt_text = _extract_text(prompt_values[idx])
        completion_text = _extract_text(completion)

        seed = (
            int(seeds_col[idx])
            if seeds_col
            else _seed_from_prompt(prompt_text, fallback_seed=idx)
        )
        prefix: list[str] = json.loads(prefixes_col[idx]) if prefixes_col else []

        first_action = parse_action(completion_text)
        try:
            reward = _rollout_reward(seed, first_action, prefix)
        except Exception as exc:
            print(f"reward_fn rollout error (seed={seed}): {exc}")
            reward = -5.0
        rewards.append(float(reward))
    return rewards


# ── evaluation ───────────────────────────────────────────────────────────────

def evaluate_policy(
    num_episodes: int = 20,
    seed_base: int = 99_000,
) -> dict[str, float]:
    """Greedy rollouts over fixed seeds. Returns mean reward and success rate."""
    env = SpaceFaultRecoveryEnvironment()
    rewards: list[float] = []
    successes: list[float] = []

    for i in range(num_episodes):
        seed = seed_base + i
        obs = env.reset(seed=seed)
        total_reward = 0.0

        while (
            not bool(getattr(obs, "done", False))
            and int(getattr(obs, "step", 0)) < MAX_STEPS
        ):
            prompt = obs_to_prompt(obs, seed=seed)
            action = _action_from_model(prompt)
            obs = env.step(action.to_action())
            total_reward += float(obs.reward or 0.0)

        status = str(getattr(obs, "mission_status", ""))
        if status == "recovered":
            total_reward += 5.0
        elif status == "lost":
            total_reward -= 5.0

        rewards.append(total_reward)
        successes.append(1.0 if status == "recovered" else 0.0)

    return {
        "mean_reward": sum(rewards) / len(rewards),
        "success_rate": sum(successes) / len(successes),
        "min_reward": min(rewards),
        "max_reward": max(rewards),
    }


# ── local CSV callback ───────────────────────────────────────────────────────

class LocalCSVCallback(TrainerCallback):
    """Appends trainer log dicts to a CSV at every logging step."""

    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self._fieldnames: list[str] | None = None

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if not logs:
            return
        row: dict[str, Any] = {
            k: v for k, v in logs.items() if isinstance(v, (int, float))
        }
        row["step"] = state.global_step

        if self._fieldnames is None:
            self._fieldnames = list(row.keys())
            with self.log_path.open("w", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=self._fieldnames, extrasaction="ignore"
                )
                writer.writeheader()
                writer.writerow(row)
        else:
            with self.log_path.open("a", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=self._fieldnames, extrasaction="ignore"
                )
                writer.writerow({k: row.get(k, "") for k in self._fieldnames})


# ── argument parser ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train TRL GRPO policy for space fault recovery."
    )
    # Model
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--output-dir", default="trl-space-agent")
    parser.add_argument("--final-dir", default="trl-space-agent-final")
    # Dataset
    parser.add_argument("--num-prompts", type=int, default=256)
    parser.add_argument("--seed-base", type=int, default=10_000)
    # GRPO hyperparameters
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-completion-length", type=int, default=128)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=500)
    # LoRA / memory
    parser.add_argument(
        "--use-lora",
        action="store_true",
        default=False,
        help="Wrap model with LoRA adapters (recommended for A10G/T4)",
    )
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        default=False,
        help="Load base model in NF4 (requires bitsandbytes, for T4 runs)",
    )
    # Logging
    parser.add_argument(
        "--report-to",
        default="none",
        choices=["none", "wandb", "tensorboard"],
    )
    parser.add_argument("--wandb-project", default="space-fault-trl")
    parser.add_argument("--run-name", default="grpo-run")
    parser.add_argument(
        "--log-dir",
        default="logs/grpo",
        help="Root directory for local CSV/JSON evidence logs",
    )
    # Evaluation
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=20,
        help="Number of greedy rollout episodes for pre/post-train evaluation",
    )
    parser.add_argument("--eval-seed-base", type=int, default=99_000)
    parser.add_argument(
        "--skip-pretrain-eval",
        action="store_true",
        default=False,
        help="Skip the pre-training baseline evaluation (saves ~2 min)",
    )
    return parser.parse_args()


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    global _MODEL, _TOKENIZER
    args = parse_args()

    log_dir = Path(args.log_dir) / args.run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    if args.report_to == "wandb":
        import wandb
        wandb.init(project=args.wandb_project, name=args.run_name)

    # ── model loading ─────────────────────────────────────────────────────
    print(f"Loading {args.model_name}...")
    model_kwargs: dict[str, Any] = {"device_map": "auto"}

    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        model_kwargs["torch_dtype"] = (
            torch.float16 if torch.cuda.is_available() else torch.float32
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    _MODEL = model
    _TOKENIZER = tokenizer

    # ── pre-training eval ─────────────────────────────────────────────────
    if not args.skip_pretrain_eval:
        print("Running pre-training evaluation...")
        pretrain_metrics = evaluate_policy(
            num_episodes=args.eval_episodes,
            seed_base=args.eval_seed_base,
        )
        print(f"  Pre-train: {pretrain_metrics}")
        (log_dir / "eval_pretrain.json").write_text(
            json.dumps(pretrain_metrics, indent=2)
        )
        if args.report_to == "wandb":
            import wandb
            wandb.log({"pretrain/" + k: v for k, v in pretrain_metrics.items()})

    # ── dataset ───────────────────────────────────────────────────────────
    print("Building prompt dataset...")
    dataset = build_prompt_dataset(
        num_prompts=args.num_prompts, seed_base=args.seed_base
    )

    # ── LoRA ──────────────────────────────────────────────────────────────
    peft_config = None
    if args.use_lora:
        from peft import LoraConfig, TaskType
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "v_proj"],
            bias="none",
        )
        print(f"LoRA enabled: r={args.lora_r}, alpha={args.lora_alpha}")

    # ── GRPO config + trainer ─────────────────────────────────────────────
    config = GRPOConfig(
        output_dir=args.output_dir,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        max_steps=args.max_steps,
        report_to=args.report_to,
        run_name=args.run_name,
    )

    csv_callback = LocalCSVCallback(log_dir / "train_log.csv")

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "reward_funcs": reward_fn,
        "args": config,
        "train_dataset": dataset,
        "callbacks": [csv_callback],
    }
    if peft_config is not None:
        trainer_kwargs["peft_config"] = peft_config

    trainer = GRPOTrainer(**trainer_kwargs)

    print("Starting GRPO training...")
    trainer.train()

    # ── save ──────────────────────────────────────────────────────────────
    print("Saving model...")
    trainer.save_model(args.final_dir)
    tokenizer.save_pretrained(args.final_dir)

    # ── post-training eval ────────────────────────────────────────────────
    print("Running post-training evaluation...")
    posttrain_metrics = evaluate_policy(
        num_episodes=args.eval_episodes,
        seed_base=args.eval_seed_base,
    )
    print(f"  Post-train: {posttrain_metrics}")
    (log_dir / "eval_posttrain.json").write_text(
        json.dumps(posttrain_metrics, indent=2)
    )
    if args.report_to == "wandb":
        import wandb
        wandb.log({"posttrain/" + k: v for k, v in posttrain_metrics.items()})
        wandb.finish()

    print(f"\nDone. Logs at {log_dir}/")


if __name__ == "__main__":
    main()
