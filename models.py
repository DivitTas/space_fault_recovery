# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Space Fault Recovery Environment.

Defines the action and observation schemas for a partially-observable
spacecraft fault-recovery task.  The agent sends a SpaceFaultAction each
step; the environment returns a SpaceFaultObservation built from sensor
readings (never raw hidden state).
"""

from typing import Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


# ── Valid commands and targets ──────────────────────────────────────────

VALID_COMMANDS: set[str] = {
    # Power
    "shed_load",
    "restore_load",
    "switch_to_backup_battery",
    "reset_power_controller",
    "reconfigure_power",
    # Attitude (4)
    "stabilize_attitude",
    "switch_to_thruster_control",
    "desaturate_wheels",
    "recalibrate_star_tracker",
    # Sensor validation (3)
    "cross_validate_attitude",
    "switch_attitude_reference",
    "recalibrate_imu",
    # Diagnostics (4)
    "query_power_level",
    "query_attitude",
    "query_thermal",
    "diagnostic_scan",
    # General (2)
    "safe_mode",
    "resume_nominal",
}

TARGETED_COMMANDS: dict[str, set[str]] = {
    "shed_load": {"science_a", "science_b", "heaters", "transponder"},
    "restore_load": {"science_a", "science_b", "heaters", "transponder"},
    "reconfigure_power": {"solar_a", "solar_b"},
    "switch_attitude_reference": {"star_tracker", "gyro", "sun_sensor"},
    "query_power_level": {"battery", "solar_a", "solar_b"},
    "diagnostic_scan": {"power", "attitude", "comms"},
}

DIAGNOSTIC_COMMANDS: set[str] = {
    "query_power_level",
    "query_attitude",
    "query_thermal",
    "diagnostic_scan",
    "cross_validate_attitude",
}


# ── Action ──────────────────────────────────────────────────────────────

class SpaceFaultAction(Action):
    """
    One command issued to the spacecraft each step.

    Attributes:
        command:  The command verb (must be in VALID_COMMANDS).
        target:   Required for targeted commands (e.g. shed_load → "science_a"),
                  ignored for un-targeted commands.
    """

    command: str = Field(
        ...,
        description="Command verb, e.g. 'shed_load', 'stabilize_attitude'.",
    )
    target: Optional[str] = Field(
        default=None,
        description="Target subsystem for commands that require one.",
    )


# ── Observation ─────────────────────────────────────────────────────────

class SpaceFaultObservation(Observation):
    """
    Sensor-level spacecraft status returned to the agent each step.

    Every field here is something a real flight controller could read from
    telemetry.  Hidden values (e.g. true solar-panel health) are *not*
    exposed — the agent must run diagnostic commands to reveal them.
    """

    # ── Power ───────────────────────────────────────────────────────────
    battery_pct: float = Field(
        default=100.0,
        description="Battery state-of-charge, 0-100 %.",
    )
    battery_drain_rate: float = Field(
        default=0.0,
        description="Change in battery_pct since last step (%/step).",
    )
    solar_a_sensor_output_w: float = Field(
        default=0.0,
        description="Solar panel A power sensor reading (watts).",
    )
    solar_b_sensor_output_w: float = Field(
        default=0.0,
        description="Solar panel B power sensor reading (watts).",
    )
    bus_voltage: float = Field(
        default=28.0,
        description="Main power-bus voltage (V). Drops under heavy load.",
    )

    # ── Attitude ────────────────────────────────────────────────────────
    star_tracker_deg: float = Field(
        default=0.0,
        description="Star-tracker pointing-error estimate (degrees).",
    )
    gyro_deg: float = Field(
        default=0.0,
        description="Gyroscope pointing-error estimate (degrees).",
    )
    sun_sensor_deg: float = Field(
        default=0.0,
        description="Sun-sensor pointing-error estimate (degrees).",
    )
    attitude_mode: str = Field(
        default="nominal",
        description="ACS mode: 'nominal' | 'thruster' | 'tumbling'.",
    )
    rw_status: str = Field(
        default="nominal",
        description="Reaction-wheel status: 'nominal' | 'degraded' | 'failed'.",
    )
    fuel_units: float = Field(
        default=50.0,
        description="Remaining RCS thruster fuel (arbitrary units).",
    )

    # ── Communications ──────────────────────────────────────────────────
    signal_strength_db: float = Field(
        default=0.0,
        description="Downlink signal strength (dB above noise floor).",
    )
    transponder_status: str = Field(
        default="nominal",
        description="Transponder: 'nominal' | 'degraded' | 'offline'.",
    )
    link_bandwidth: str = Field(
        default="high",
        description="Current link quality: 'high' | 'low' | 'none'.",
    )

    # ── Thermal ─────────────────────────────────────────────────────────
    battery_temp_c: float = Field(
        default=20.0,
        description="Battery temperature (°C). Below 0 → capacity loss.",
    )
    heater_status: str = Field(
        default="online",
        description="Heater subsystem: 'online' | 'offline'.",
    )

    # ── Mission meta ────────────────────────────────────────────────────
    subsystems_online: list[str] = Field(
        default_factory=list,
        description="Names of subsystems currently powered on.",
    )
    step: int = Field(
        default=0,
        description="Current step in the episode.",
    )
    mission_status: str = Field(
        default="nominal",
        description="Overall status: 'nominal' | 'degraded' | 'critical' | 'recovered' | 'lost'.",
    )
    last_action_result: str = Field(
        default="ok",
        description="Result of the previous action: 'ok', diagnostic output, or error string.",
    )
