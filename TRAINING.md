# Training Pipeline

This project is a discrete-action reinforcement-learning task around spacecraft
fault recovery. The environment injects hidden faults, exposes only telemetry
and diagnostic text, and rewards policies that diagnose before repair, preserve
power/attitude margins, and explicitly resume nominal operations.

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
