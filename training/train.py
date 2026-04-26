"""Train a lightweight Q-learning policy and write logs/plots."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from .action_space import ActionSpec, build_action_space
    from .agent import LinearQAgent
    from .features import FEATURE_NAMES, encode_observation
    from .openenv_compat import ensure_training_runtime
    from .plotting import generate_training_plots
except ImportError:
    from training.action_space import ActionSpec, build_action_space
    from training.agent import LinearQAgent
    from training.features import FEATURE_NAMES, encode_observation
    from training.openenv_compat import ensure_training_runtime
    from training.plotting import generate_training_plots

ensure_training_runtime()

try:
    from ..server.space_fault_recovery_environment import (  # noqa: E402
        MAX_STEPS,
        SpaceFaultRecoveryEnvironment,
    )
except ImportError:
    from server.space_fault_recovery_environment import MAX_STEPS, SpaceFaultRecoveryEnvironment  # noqa: E402


EXPERT_MACRO: tuple[tuple[str, str | None], ...] = (
    ("query_power_level", "solar_a"),
    ("query_power_level", "solar_b"),
    ("query_power_level", "battery"),
    ("diagnostic_scan", "power"),
    ("diagnostic_scan", "attitude"),
    ("diagnostic_scan", "comms"),
    ("query_thermal", None),
    ("cross_validate_attitude", None),
    ("query_attitude", None),
    ("shed_load", "science_a"),
    ("shed_load", "science_b"),
    ("reset_power_controller", None),
    ("reconfigure_power", "solar_a"),
    ("reconfigure_power", "solar_b"),
    ("reconfigure_power", "solar_a"),
    ("reconfigure_power", "solar_b"),
    ("recalibrate_star_tracker", None),
    ("desaturate_wheels", None),
    ("desaturate_wheels", None),
    ("recalibrate_imu", None),
    ("restore_load", "heaters"),
    ("shed_load", "transponder"),
    ("restore_load", "transponder"),
    ("stabilize_attitude", None),
    ("stabilize_attitude", None),
    ("stabilize_attitude", None),
    ("query_attitude", None),
    ("query_thermal", None),
    ("query_power_level", "battery"),
    ("resume_nominal", None),
    ("resume_nominal", None),
    ("resume_nominal", None),
    ("resume_nominal", None),
)


EPISODE_METRIC_FIELDS = [
    "episode",
    "seed",
    "epsilon",
    "expert_mix_rate",
    "total_reward",
    "mean_loss",
    "max_loss",
    "steps",
    "done",
    "recovered",
    "lost",
    "final_status",
    "final_battery_pct",
    "final_attitude_error",
    "final_fuel_units",
    "expert_action_count",
    "invalid_action_count",
    "diagnostic_action_count",
    "rolling_reward_20",
    "rolling_loss_20",
    "rolling_success_rate_20",
    "eval_mean_reward",
    "eval_success_rate",
]

STEP_METRIC_FIELDS = [
    "episode",
    "step",
    "action_index",
    "action_label",
    "reward",
    "loss",
    "done",
    "mission_status",
    "battery_pct",
    "attitude_error",
    "fuel_units",
    "last_action_result",
]


METRIC_MANIFEST = {
    "episode_metrics_csv": "metrics.csv",
    "step_metrics_csv": "steps.csv",
    "model_json": "model.json",
    "config_json": "config.json",
    "metrics": {
        "total_reward": "Sum of environment rewards collected in the episode.",
        "expert_mix_rate": "Probability of following the diagnostic recovery demonstration in this episode.",
        "mean_loss": "Mean temporal-difference loss from Q-learning updates.",
        "max_loss": "Largest temporal-difference loss seen in the episode.",
        "steps": "Number of environment steps executed before termination or timeout.",
        "recovered": "1 when the final mission_status is recovered, else 0.",
        "lost": "1 when the final mission_status is lost, else 0.",
        "final_status": "Final mission_status reported by the environment.",
        "final_battery_pct": "Battery state of charge at the end of the episode.",
        "final_attitude_error": "Sun-sensor pointing error at the end of the episode.",
        "final_fuel_units": "Remaining thruster fuel at the end of the episode.",
        "expert_action_count": "Actions selected from the demonstration recovery macro.",
        "invalid_action_count": "Count of actions producing error/refused results.",
        "diagnostic_action_count": "Count of diagnostic-style actions used in the episode.",
        "rolling_reward_20": "Trailing 20-episode mean of total_reward.",
        "rolling_loss_20": "Trailing 20-episode mean of mean_loss.",
        "rolling_success_rate_20": "Trailing 20-episode recovery rate.",
        "eval_mean_reward": "Greedy-policy evaluation reward when --eval-every fires.",
        "eval_success_rate": "Greedy-policy recovery rate when --eval-every fires.",
    },
    "graphs": {
        "plots/reward_curve.svg": "Episode total reward with trailing 20-episode average.",
        "plots/loss_curve.svg": "Mean TD loss with trailing 20-episode average.",
        "plots/success_rate_curve.svg": "Trailing 20-episode recovery success rate.",
        "plots/episode_length_curve.svg": "Episode length in steps.",
        "plots/step_reward_curve.svg": "Per-step reward across the entire run (all episodes).",
    },
}


def epsilon_for_episode(args: argparse.Namespace, episode: int) -> float:
    if args.epsilon_decay_episodes <= 0:
        return args.epsilon_end
    fraction = min(1.0, episode / float(args.epsilon_decay_episodes))
    return args.epsilon_start + fraction * (args.epsilon_end - args.epsilon_start)


def expert_mix_for_episode(args: argparse.Namespace, episode: int) -> float:
    if args.expert_decay_episodes <= 0:
        return args.expert_mix_end
    fraction = min(1.0, episode / float(args.expert_decay_episodes))
    return args.expert_mix_start + fraction * (args.expert_mix_end - args.expert_mix_start)


def rolling(values: list[float], window: int = 20) -> float:
    subset = values[-window:]
    return sum(subset) / max(1, len(subset))


def is_diagnostic(action: ActionSpec) -> bool:
    return action.command in {
        "query_power_level",
        "query_attitude",
        "query_thermal",
        "diagnostic_scan",
        "cross_validate_attitude",
    }


def build_action_index(actions: list[ActionSpec]) -> dict[tuple[str, str | None], int]:
    return {(action.command, action.target): action.index for action in actions}


def build_expert_indices(actions: list[ActionSpec]) -> list[int]:
    action_index = build_action_index(actions)
    missing = [spec for spec in EXPERT_MACRO if spec not in action_index]
    if missing:
        raise ValueError(f"Expert macro references actions missing from action space: {missing}")
    return [action_index[spec] for spec in EXPERT_MACRO]


def run_episode(
    *,
    env: SpaceFaultRecoveryEnvironment,
    agent: LinearQAgent,
    actions: list[ActionSpec],
    seed: int,
    episode: int,
    epsilon: float,
    max_steps: int,
    train: bool,
    expert_mix_rate: float = 0.0,
    expert_indices: list[int] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    obs = env.reset(seed=seed)
    features = encode_observation(obs, max_steps=max_steps)

    total_reward = 0.0
    losses: list[float] = []
    step_rows: list[dict[str, Any]] = []
    expert_action_count = 0
    invalid_action_count = 0
    diagnostic_action_count = 0
    use_expert_episode = (
        train
        and expert_indices is not None
        and agent.rng.random() < expert_mix_rate
    )

    for step_idx in range(max_steps):
        use_expert = use_expert_episode and step_idx < len(expert_indices)
        if use_expert:
            action_idx = expert_indices[step_idx]
            expert_action_count += 1
        elif train:
            action_idx = agent.select_action(features, epsilon)
        else:
            action_idx = agent.greedy_action(features)

        action_spec = actions[action_idx]
        next_obs = env.step(action_spec.to_action())
        next_features = encode_observation(next_obs, max_steps=max_steps)
        reward = float(next_obs.reward or 0.0)
        done = bool(next_obs.done)
        loss = (
            agent.update(features, action_idx, reward, next_features, done)
            if train
            else 0.0
        )

        result_text = str(next_obs.last_action_result)
        if result_text.startswith("error:") or result_text.startswith("refused:"):
            invalid_action_count += 1
        if is_diagnostic(action_spec):
            diagnostic_action_count += 1

        total_reward += reward
        losses.append(loss)
        step_rows.append(
            {
                "episode": episode,
                "step": next_obs.step,
                "action_index": action_idx,
                "action_label": action_spec.label,
                "reward": f"{reward:.6f}",
                "loss": f"{loss:.6f}",
                "done": int(done),
                "mission_status": next_obs.mission_status,
                "battery_pct": f"{float(next_obs.battery_pct):.3f}",
                "attitude_error": f"{float(next_obs.sun_sensor_deg):.3f}",
                "fuel_units": f"{float(next_obs.fuel_units):.3f}",
                "last_action_result": result_text,
            }
        )

        features = next_features
        obs = next_obs
        if done:
            break

    final_status = str(obs.mission_status)
    episode_row = {
        "episode": episode,
        "seed": seed,
        "epsilon": f"{epsilon:.6f}",
        "expert_mix_rate": f"{expert_mix_rate:.6f}",
        "total_reward": f"{total_reward:.6f}",
        "mean_loss": f"{(statistics.fmean(losses) if losses else 0.0):.6f}",
        "max_loss": f"{(max(losses) if losses else 0.0):.6f}",
        "steps": int(getattr(obs, "step", 0) or 0),
        "done": int(bool(getattr(obs, "done", False))),
        "recovered": int(final_status == "recovered"),
        "lost": int(final_status == "lost"),
        "final_status": final_status,
        "final_battery_pct": f"{float(getattr(obs, 'battery_pct', 0.0)):.3f}",
        "final_attitude_error": f"{float(getattr(obs, 'sun_sensor_deg', 0.0)):.3f}",
        "final_fuel_units": f"{float(getattr(obs, 'fuel_units', 0.0)):.3f}",
        "expert_action_count": expert_action_count,
        "invalid_action_count": invalid_action_count,
        "diagnostic_action_count": diagnostic_action_count,
    }
    return episode_row, step_rows


def evaluate_policy(
    *,
    agent: LinearQAgent,
    actions: list[ActionSpec],
    base_seed: int,
    eval_episodes: int,
    max_steps: int,
) -> tuple[float, float]:
    rewards: list[float] = []
    successes = 0
    env = SpaceFaultRecoveryEnvironment()
    for offset in range(eval_episodes):
        row, _ = run_episode(
            env=env,
            agent=agent,
            actions=actions,
            seed=base_seed + offset,
            episode=offset,
            epsilon=0.0,
            expert_mix_rate=0.0,
            max_steps=max_steps,
            train=False,
        )
        rewards.append(float(row["total_reward"]))
        successes += int(row["recovered"])
    return statistics.fmean(rewards) if rewards else 0.0, successes / max(1, eval_episodes)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=250)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-dir", type=Path, default=Path("logs") / "training")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-episodes", type=int, default=180)
    parser.add_argument("--expert-mix-start", type=float, default=0.75)
    parser.add_argument("--expert-mix-end", type=float, default=0.0)
    parser.add_argument("--expert-decay-episodes", type=int, default=140)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_name = args.run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.log_dir / run_name
    plots_dir = run_dir / "plots"
    run_dir.mkdir(parents=True, exist_ok=True)

    actions = build_action_space()
    expert_indices = build_expert_indices(actions)
    agent = LinearQAgent(
        n_features=len(FEATURE_NAMES),
        n_actions=len(actions),
        learning_rate=args.lr,
        gamma=args.gamma,
        gradient_clip=args.gradient_clip,
        seed=args.seed,
    )

    config = {
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "learning_rate": args.lr,
        "gamma": args.gamma,
        "gradient_clip": args.gradient_clip,
        "epsilon_start": args.epsilon_start,
        "epsilon_end": args.epsilon_end,
        "epsilon_decay_episodes": args.epsilon_decay_episodes,
        "expert_mix_start": args.expert_mix_start,
        "expert_mix_end": args.expert_mix_end,
        "expert_decay_episodes": args.expert_decay_episodes,
        "eval_every": args.eval_every,
        "eval_episodes": args.eval_episodes,
        "n_features": len(FEATURE_NAMES),
        "n_actions": len(actions),
        "feature_names": list(FEATURE_NAMES),
        "action_space": [action.to_dict() for action in actions],
    }
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "metrics_manifest.json", METRIC_MANIFEST)

    metrics_rows: list[dict[str, Any]] = []
    reward_history: list[float] = []
    loss_history: list[float] = []
    success_history: list[float] = []

    env = SpaceFaultRecoveryEnvironment()
    metrics_path = run_dir / "metrics.csv"
    steps_path = run_dir / "steps.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as metrics_handle, steps_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as steps_handle:
        metrics_writer = csv.DictWriter(metrics_handle, fieldnames=EPISODE_METRIC_FIELDS)
        steps_writer = csv.DictWriter(steps_handle, fieldnames=STEP_METRIC_FIELDS)
        metrics_writer.writeheader()
        steps_writer.writeheader()

        for episode in range(1, args.episodes + 1):
            epsilon = epsilon_for_episode(args, episode - 1)
            expert_mix_rate = expert_mix_for_episode(args, episode - 1)
            row, step_rows = run_episode(
                env=env,
                agent=agent,
                actions=actions,
                seed=args.seed + episode - 1,
                episode=episode,
                epsilon=epsilon,
                expert_mix_rate=expert_mix_rate,
                expert_indices=expert_indices,
                max_steps=args.max_steps,
                train=True,
            )

            reward_history.append(float(row["total_reward"]))
            loss_history.append(float(row["mean_loss"]))
            success_history.append(float(row["recovered"]))

            row["rolling_reward_20"] = f"{rolling(reward_history):.6f}"
            row["rolling_loss_20"] = f"{rolling(loss_history):.6f}"
            row["rolling_success_rate_20"] = f"{rolling(success_history):.6f}"
            row["eval_mean_reward"] = ""
            row["eval_success_rate"] = ""

            if args.eval_every > 0 and episode % args.eval_every == 0:
                eval_reward, eval_success = evaluate_policy(
                    agent=agent,
                    actions=actions,
                    base_seed=args.seed + 100_000 + episode * 100,
                    eval_episodes=args.eval_episodes,
                    max_steps=args.max_steps,
                )
                row["eval_mean_reward"] = f"{eval_reward:.6f}"
                row["eval_success_rate"] = f"{eval_success:.6f}"

            metrics_writer.writerow(row)
            steps_writer.writerows(step_rows)
            metrics_handle.flush()
            steps_handle.flush()
            metrics_rows.append(row.copy())

            if episode == 1 or episode % max(1, args.episodes // 10) == 0:
                print(
                    "episode "
                    f"{episode:04d}/{args.episodes} "
                    f"reward={float(row['total_reward']):+.2f} "
                    f"loss={float(row['mean_loss']):.4f} "
                    f"expert={float(row['expert_mix_rate']):.2f} "
                    f"success20={float(row['rolling_success_rate_20']):.2f} "
                    f"status={row['final_status']}"
                )

    agent.save(
        run_dir / "model.json",
        feature_names=list(FEATURE_NAMES),
        action_specs=[action.to_dict() for action in actions],
        extra_config=config,
    )

    plot_paths: list[Path] = []
    if not args.no_plots:
        plot_paths = generate_training_plots(metrics_rows, plots_dir, steps_path)

    print(f"training run: {run_dir}")
    print(f"episode metrics: {metrics_path}")
    print(f"step metrics: {steps_path}")
    print(f"model: {run_dir / 'model.json'}")
    if plot_paths:
        print("plots:")
        for path in plot_paths:
            print(f"  {path}")


if __name__ == "__main__":
    main()
