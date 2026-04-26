---
title: Space Fault Recovery Environment Server
emoji: 🛰️
colorFrom: pink
colorTo: purple
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - reinforcement-learning
  - llm-agents
  - partial-observability
  - spacecraft
  - fault-recovery
  - cascade-failure
---

> **Live demo:** [huggingface.co/spaces/DivitTas/Space-Fault-Recovery](https://huggingface.co/spaces/DivitTas/Space-Fault-Recovery)

# Space Fault Recovery Environment

A spacecraft fault-cascade simulation built on the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) framework. An RL agent (LLM) takes the role of an onboard flight controller and must recover a degraded spacecraft over up to 50 decision steps before the mission is lost.

Each episode resets with 1–3 injected faults sampled from a fault library (solar-panel degradation, battery drain, reaction-wheel faults, attitude drift, thermal faults, comms degradation). The agent only sees sensor telemetry — true subsystem health is hidden and must be inferred from readings or revealed via diagnostic commands.

## Quick Start

```python
from space_fault_recovery import SpaceFaultAction, SpaceFaultRecoveryEnv

try:
    env = SpaceFaultRecoveryEnv.from_docker_image("space_fault_recovery-env:latest")

    obs = env.reset().observation
    print(f"Mission: {obs.mission_status}, battery: {obs.battery_pct:.1f}%")

    # Run a diagnostic, then act
    result = env.step(SpaceFaultAction(command="diagnostic_scan", target="power"))
    print(f"Diagnostic: {result.observation.last_action_result}")

    result = env.step(SpaceFaultAction(command="shed_load", target="science_a"))
    print(f"Reward: {result.reward}, status: {result.observation.mission_status}")

finally:
    env.close()
```

`SpaceFaultRecoveryEnv.from_docker_image()` starts the container, waits for the server to be ready, connects, and tears the container down on `close()`.

## Building the Docker Image

```bash
docker build -t space_fault_recovery-env:latest -f server/Dockerfile .
```

## Deploying to Hugging Face Spaces

```bash
# From the environment directory (where openenv.yaml lives)
openenv push

# Or with options
openenv push --namespace my-org --private
```

`openenv push` validates the directory, prepares a Hugging Face Docker space build (with the web UI enabled), and uploads it.

### Options

- `--directory`, `-d`: Directory containing the OpenEnv environment (default: cwd)
- `--repo-id`, `-r`: Repo in `username/repo-name` form (default from `openenv.yaml`)
- `--base-image`, `-b`: Base Docker image (overrides Dockerfile `FROM`)
- `--private`: Deploy as a private space

The deployed space exposes `/web` (interactive UI), `/docs` (OpenAPI), `/health`, and `/ws` (WebSocket session endpoint).

## Environment Details

### Action — `SpaceFaultAction`

- `command` (str) — verb from `VALID_COMMANDS` in `models.py`
- `target` (str, optional) — required only for targeted commands (e.g. `shed_load → "science_a"`)

Command groups:

| Group | Commands |
| --- | --- |
| Power | `shed_load`, `restore_load`, `switch_to_backup_battery`, `reset_power_controller`, `reconfigure_power` |
| Attitude | `stabilize_attitude`, `switch_to_thruster_control`, `desaturate_wheels`, `recalibrate_star_tracker` |
| Sensor validation | `cross_validate_attitude`, `switch_attitude_reference`, `recalibrate_imu` |
| Diagnostics | `query_power_level`, `query_attitude`, `query_thermal`, `diagnostic_scan` |
| General | `safe_mode`, `resume_nominal` |

See `models.py` for the full `TARGETED_COMMANDS` mapping.

### Observation — `SpaceFaultObservation`

Sensor-level telemetry only. Key fields:

- **Power:** `battery_pct`, `battery_drain_rate`, `solar_a_sensor_output_w`, `solar_b_sensor_output_w`, `bus_voltage`
- **Attitude:** `star_tracker_deg`, `gyro_deg`, `sun_sensor_deg`, `attitude_mode`, `rw_status`, `fuel_units`
- **Comms:** `signal_strength_db`, `transponder_status`, `link_bandwidth`
- **Thermal:** `battery_temp_c`, `heater_status`
- **Mission meta:** `subsystems_online`, `step`, `mission_status`, `last_action_result`

True subsystem health (e.g. `solar_a_health`) is *not* exposed — the agent must run diagnostics or infer it from sensor readings. Diagnostic command results land in `last_action_result`.

### Mission Outcomes

`mission_status` transitions through `nominal → degraded → critical`, ending in either `recovered` or `lost`. Episodes are truncated at 50 steps; timeout always sets `lost` with a −10 penalty.

## Advanced Usage

### Connecting to an existing server

```python
from space_fault_recovery import SpaceFaultAction, SpaceFaultRecoveryEnv

env = SpaceFaultRecoveryEnv(base_url="http://localhost:8000")
result = env.reset()
result = env.step(SpaceFaultAction(command="safe_mode"))
```

When connecting to an existing server, `env.close()` will not stop the server.

### Context manager (WebSocket session)

```python
with SpaceFaultRecoveryEnv(base_url="http://localhost:8000") as env:
    env.reset()
    for cmd in ["diagnostic_scan", "shed_load", "stabilize_attitude"]:
        target = "power" if cmd == "diagnostic_scan" else ("science_a" if cmd == "shed_load" else None)
        result = env.step(SpaceFaultAction(command=cmd, target=target))
        print(result.observation.mission_status, result.reward)
```

The client uses WebSockets for lower latency and persistent session state across an episode.

### Concurrent sessions

To allow multiple concurrent WebSocket clients, edit `server/app.py` to pass the environment **class** (not instance) and set `max_concurrent_envs`:

```python
app = create_app(
    SpaceFaultRecoveryEnvironment,
    SpaceFaultAction,
    SpaceFaultObservation,
    max_concurrent_envs=4,
)
```

## Development & Testing

```bash
# Install deps (uv.lock is committed — use uv sync, not pip)
uv sync

# Test environment logic directly, no server
python3 server/space_fault_recovery_environment.py

# Run server locally with reload
uvicorn server.app:app --reload

# Smoke test
curl -X POST localhost:8000/reset
curl -X POST localhost:8000/step -H "Content-Type: application/json" \
  -d '{"command": "shed_load", "target": "science_a"}'
```

## Training an LLM with GRPO

The environment is designed to train LLMs via reinforcement learning.
We use HF TRL's GRPO algorithm on `Qwen/Qwen2.5-1.5B-Instruct`.

**Colab notebook:** [`training/train_grpo.ipynb`](training/train_grpo.ipynb)  
**Training guide:** [`TRAINING.md`](TRAINING.md)

Quick start (A10G, ~35 min, ~$1.50):

```bash
pip install -r training/requirements-train.txt
python trl_train.py --use-lora --max-steps 500 --report-to wandb --run-name grpo-lora-500
```

Training evidence (reward curves, before/after evaluation) is written to
`logs/grpo/<run-name>/` and synced to WandB.

<!-- TODO: add WandB run link and reward curve PNG after first training run -->

## Project Structure

```
space_fault_recovery/
├── __init__.py                              # Module exports
├── README.md                                # This file
├── CLAUDE.md                                # Repo guidance for Claude Code
├── TRAINING.md                              # Training pipeline guide
├── openenv.yaml                             # OpenEnv manifest
├── pyproject.toml                           # Project metadata and dependencies
├── uv.lock                                  # Locked dependencies
├── trl_train.py                             # GRPO training script (HF TRL)
├── client.py                                # SpaceFaultRecoveryEnv WebSocket client
├── models.py                                # SpaceFaultAction / SpaceFaultObservation
├── training/
│   ├── train_grpo.ipynb                     # Colab-runnable GRPO notebook
│   ├── train.py                             # Linear Q-learning baseline
│   ├── requirements-train.txt               # Training-only pip deps
│   ├── action_space.py                      # Full 31-action discrete space
│   ├── features.py                          # Observation encoder (Q-learning)
│   ├── agent.py                             # LinearQAgent
│   ├── plotting.py                          # SVG plot generation
│   └── openenv_compat.py                    # Shims for training without OpenEnv installed
└── server/
    ├── __init__.py
    ├── space_fault_recovery_environment.py  # Cascade simulation logic
    ├── app.py                               # FastAPI app (HTTP + WebSocket)
    └── Dockerfile
```
