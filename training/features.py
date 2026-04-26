"""Observation featurization for lightweight Q-learning."""

from __future__ import annotations

from typing import Any


ATTITUDE_MODES = ("nominal", "thruster", "tumbling")
RW_STATUSES = ("nominal", "degraded", "failed")
TRANSPONDER_STATUSES = ("nominal", "degraded", "offline")
LINK_BANDWIDTHS = ("high", "low", "none")
MISSION_STATUSES = ("nominal", "degraded", "critical", "recovered", "lost")
SUBSYSTEMS = ("science_a", "science_b", "heaters", "transponder")


def _clip(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _one_hot(value: str, choices: tuple[str, ...]) -> list[float]:
    return [1.0 if value == choice else 0.0 for choice in choices]


FEATURE_NAMES: tuple[str, ...] = (
    "bias",
    "battery_pct",
    "battery_drain_rate",
    "solar_a_output",
    "solar_b_output",
    "bus_voltage",
    "star_tracker_error",
    "gyro_error",
    "sun_sensor_error",
    "fuel_units",
    "signal_strength",
    "battery_temp",
    "episode_progress",
    "science_a_online",
    "science_b_online",
    "heaters_online",
    "transponder_online",
    "last_action_error",
    "last_action_refused",
    "last_action_diag",
    *(f"attitude_mode_{name}" for name in ATTITUDE_MODES),
    *(f"rw_status_{name}" for name in RW_STATUSES),
    *(f"transponder_status_{name}" for name in TRANSPONDER_STATUSES),
    *(f"link_bandwidth_{name}" for name in LINK_BANDWIDTHS),
    *(f"mission_status_{name}" for name in MISSION_STATUSES),
)


def encode_observation(obs: Any, max_steps: int = 50) -> list[float]:
    subsystems = set(getattr(obs, "subsystems_online", []) or [])
    last_action = str(getattr(obs, "last_action_result", "") or "").lower()
    step = float(getattr(obs, "step", 0) or 0)

    values = [
        1.0,
        _clip(float(getattr(obs, "battery_pct", 100.0)) / 100.0, 0.0, 1.5),
        _clip(float(getattr(obs, "battery_drain_rate", 0.0)) / 20.0),
        _clip(float(getattr(obs, "solar_a_sensor_output_w", 0.0)) / 90.0, 0.0, 1.5),
        _clip(float(getattr(obs, "solar_b_sensor_output_w", 0.0)) / 90.0, 0.0, 1.5),
        _clip(float(getattr(obs, "bus_voltage", 28.0)) / 28.0, 0.0, 1.5),
        _clip(float(getattr(obs, "star_tracker_deg", 0.0)) / 30.0),
        _clip(float(getattr(obs, "gyro_deg", 0.0)) / 30.0),
        _clip(float(getattr(obs, "sun_sensor_deg", 0.0)) / 30.0),
        _clip(float(getattr(obs, "fuel_units", 50.0)) / 50.0, 0.0, 1.5),
        _clip(float(getattr(obs, "signal_strength_db", 0.0)) / 25.0, 0.0, 1.5),
        _clip((float(getattr(obs, "battery_temp_c", 20.0)) + 20.0) / 60.0, 0.0, 1.5),
        _clip(step / float(max_steps), 0.0, 1.0),
        *[1.0 if name in subsystems else 0.0 for name in SUBSYSTEMS],
        1.0 if last_action.startswith("error:") else 0.0,
        1.0 if last_action.startswith("refused:") else 0.0,
        1.0 if "diag" in last_action else 0.0,
        *_one_hot(str(getattr(obs, "attitude_mode", "nominal")), ATTITUDE_MODES),
        *_one_hot(str(getattr(obs, "rw_status", "nominal")), RW_STATUSES),
        *_one_hot(str(getattr(obs, "transponder_status", "nominal")), TRANSPONDER_STATUSES),
        *_one_hot(str(getattr(obs, "link_bandwidth", "high")), LINK_BANDWIDTHS),
        *_one_hot(str(getattr(obs, "mission_status", "nominal")), MISSION_STATUSES),
    ]

    if len(values) != len(FEATURE_NAMES):
        raise RuntimeError(
            f"Feature vector length mismatch: {len(values)} values for {len(FEATURE_NAMES)} names"
        )
    return values

