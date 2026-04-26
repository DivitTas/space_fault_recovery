# Training Pipeline

This project is a discrete-action reinforcement-learning task around spacecraft
fault recovery. The environment injects hidden faults, exposes only telemetry
and diagnostic text, and rewards policies that diagnose before repair, preserve
power/attitude margins, and explicitly resume nominal operations.

---

## GRPO Training with HF TRL (required for hackathon submission)

`trl_train.py` trains `Qwen/Qwen2.5-1.5B-Instruct` with GRPO against the live
environment. The reward function runs full episode rollouts — the model sees
real telemetry and is rewarded for recovering the spacecraft.

### Install training dependencies

```bash
pip install -r training/requirements-train.txt
```

### Quick smoke run (CPU, ~1 min, no GPU needed)

> **Note:** `training/requirements-train.txt` includes `bitsandbytes`, which may
> fail to install on CPU-only or non-Linux machines. If you hit install errors,
> skip it with `pip install $(grep -v bitsandbytes training/requirements-train.txt | grep -v '^#')`.

```bash
python trl_train.py \
  --max-steps 2 \
  --num-prompts 8 \
  --eval-episodes 2 \
  --skip-pretrain-eval \
  --report-to none
```

### Full run on HF Jobs / Colab A10G (recommended)

```bash
python trl_train.py \
  --use-lora \
  --max-steps 500 \
  --num-prompts 256 \
  --eval-episodes 20 \
  --report-to wandb \
  --wandb-project space-fault-trl \
  --run-name grpo-lora-500
```

Expected wall time: ~35–45 min on A10G (~$1.50 in HF credits).  
Use `--load-in-4bit` for T4 runs to fit within 16 GB VRAM.

### Colab notebook

`training/train_grpo.ipynb` is a self-contained Colab-runnable notebook that
installs deps, runs pre/post-training evaluation, and saves reward curve PNGs.

### Key flags

| Flag | Default | Description |
|---|---|---|
| `--use-lora` | off | LoRA adapters (r=16, alpha=32) — required for A10G |
| `--load-in-4bit` | off | NF4 quantization — for T4 (16 GB) runs |
| `--max-steps` | 500 | GRPO gradient steps |
| `--num-prompts` | 256 | Dataset size (40% step-0, 30% after 5 steps, 30% after 10) |
| `--report-to wandb` | none | Enable WandB logging |
| `--run-name` | grpo-run | Name for output dir and WandB run |
| `--skip-pretrain-eval` | off | Skip base-model eval (saves ~2 min) |

### Outputs

Each run writes to `logs/grpo/<run-name>/`:

```
config.json           — hyperparameter record
train_log.csv         — per-step reward and loss
eval_pretrain.json    — base model mean reward + success rate
eval_posttrain.json   — trained model mean reward + success rate
plots/
  grpo_reward_curve.png
  grpo_loss_curve.png
```

---

## Linear Q-learning baseline (for comparison only)

Run a local training job:

```bash
python training/train.py --episodes 250
```

Short smoke run:

```bash
python training/train.py --episodes 10 --eval-every 0 --run-name smoke
```

Each run writes to:

```text
logs/training/<run-name>/
  config.json
  metrics_manifest.json
  metrics.csv
  steps.csv
  model.json
  plots/
    reward_curve.svg
    loss_curve.svg
    success_rate_curve.svg
    episode_length_curve.svg
    step_reward_curve.svg
```

## Metrics Logged

- `total_reward`: episode reward sum.
- `mean_loss`: mean TD loss for the linear Q-learning updates.
- `max_loss`: largest TD loss in the episode.
- `steps`: number of environment steps used.
- `recovered`: whether the mission reached `recovered`.
- `lost`: whether the mission reached `lost`.
- `final_status`: final `mission_status`.
- `final_battery_pct`: ending battery state of charge.
- `final_attitude_error`: ending sun-sensor pointing error.
- `final_fuel_units`: ending thruster fuel.
- `expert_mix_rate`: probability of following the diagnostic recovery demonstration.
- `expert_action_count`: actions selected from the demonstration recovery macro.
- `invalid_action_count`: commands that produced `error:` or `refused:`.
- `diagnostic_action_count`: diagnostic commands used in the episode.
- `rolling_reward_20`: trailing 20-episode reward average.
- `rolling_loss_20`: trailing 20-episode loss average.
- `rolling_success_rate_20`: trailing 20-episode recovery rate.
- `eval_mean_reward`: greedy evaluation reward when enabled.
- `eval_success_rate`: greedy evaluation recovery rate when enabled.

## Graphs

- `reward_curve.svg`: episode reward and 20-episode average.
- `loss_curve.svg`: mean training loss and 20-episode average.
- `success_rate_curve.svg`: rolling recovery rate.
- `episode_length_curve.svg`: steps per episode.
- `step_reward_curve.svg`: per-step rewards across the full run.

The trainer uses a lightweight linear Q-learning model plus a decaying
demonstration warmup so it can see successful diagnostic/recovery trajectories
before exploration fully takes over. If OpenEnv or Pydantic are not installed,
the trainer uses small compatibility shims only for direct local training; the
server contract is unchanged.
