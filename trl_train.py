import os
import random
import torch
import wandb

from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOTrainer, GRPOConfig

# 🔥 IMPORT YOUR ENV
from server.space_fault_recovery_environment import SpaceFaultRecoveryEnvironment
from models import SpaceFaultAction

# ================================
# ✅ CONFIG
# ================================

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

# 🔥 SMALL ACTION SPACE (VERY IMPORTANT)
ACTIONS = [
    "query_power_level:battery",
    "diagnostic_scan:power",
    "reset_power_controller",
    "reconfigure_power:solar_a",
    "reconfigure_power:solar_b",
    "stabilize_attitude",
    "resume_nominal",
]

# ================================
# ✅ PROMPT FUNCTION
# ================================

def obs_to_prompt(obs):
    return f"""
You are an expert spacecraft fault recovery agent.

Current spacecraft telemetry:
- Battery: {obs.battery_pct}%
- Attitude error: {obs.sun_sensor_deg} degrees
- Fuel: {obs.fuel_units}
- Status: {obs.mission_status}

Available actions:
{", ".join(ACTIONS)}

Choose the SINGLE best next action.
Respond ONLY with the action string.
"""

# ================================
# ✅ PARSE LLM OUTPUT
# ================================

def parse_action(text):
    text = text.lower()
    for action in ACTIONS:
        if action in text:
            return action
    return "diagnostic_scan:power"  # fallback safe action

# ================================
# ✅ REWARD FUNCTION (CORE RL)
# ================================

def reward_fn(completions, **kwargs):
    rewards = []

    for completion in completions:
        try:
            env = SpaceFaultRecoveryEnvironment()
            obs = env.reset()

            action_str = parse_action(completion)

            if ":" in action_str:
                cmd, target = action_str.split(":")
            else:
                cmd, target = action_str, None

            obs = env.step(SpaceFaultAction(command=cmd, target=target))

            reward = float(obs.reward)

            # 🔥 EXTRA SHAPING (IMPORTANT)
            if obs.mission_status == "recovered":
                reward += 5.0
            elif obs.mission_status == "lost":
                reward -= 5.0

        except Exception as e:
            print("Error in reward_fn:", e)
            reward = -1.0

        rewards.append(reward)

    # ✅ LOG TO WANDB
    if len(rewards) > 0:
        wandb.log({
            "reward_mean": sum(rewards) / len(rewards),
            "reward_max": max(rewards),
            "reward_min": min(rewards),
        })

    return rewards

# ================================
# ✅ MAIN TRAINING
# ================================

def main():

    # 🔥 WANDB INIT (MANDATORY FOR JUDGING)
    wandb.init(project="space-fault-trl", name="trl-run")

    # =========================
    # LOAD MODEL
    # =========================
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    # =========================
    # DATASET (SIMPLE PROMPTS)
    # =========================
    dataset = ["Recover the spacecraft"] * 200

    # =========================
    # GRPO CONFIG
    # =========================
    config = GRPOConfig(
        output_dir="trl-space-agent",
        num_generations=4,
        max_completion_length=40,
        per_device_train_batch_size=2,
        learning_rate=5e-6,
        logging_steps=1,
        report_to="wandb"
    )

    # =========================
    # TRAINER
    # =========================
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=config,
        train_dataset=dataset,
    )

    # =========================
    # TRAIN 🚀
    # =========================
    print("Starting training...")
    trainer.train()

    # =========================
    # SAVE MODEL
    # =========================
    print("Saving model...")
    trainer.save_model("trl-space-agent-final")
    tokenizer.save_pretrained("trl-space-agent-final")

    print("✅ Training complete!")


if __name__ == "__main__":
    main()
