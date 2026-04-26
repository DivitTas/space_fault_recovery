"""Discrete action space for the spacecraft fault-recovery task."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .openenv_compat import ensure_training_runtime

ensure_training_runtime()

try:
    from ..models import SpaceFaultAction, TARGETED_COMMANDS, VALID_COMMANDS  # noqa: E402
except ImportError:
    from models import SpaceFaultAction, TARGETED_COMMANDS, VALID_COMMANDS  # noqa: E402


COMMAND_ORDER: tuple[str, ...] = (
    "shed_load",
    "restore_load",
    "switch_to_backup_battery",
    "reset_power_controller",
    "reconfigure_power",
    "stabilize_attitude",
    "switch_to_thruster_control",
    "desaturate_wheels",
    "recalibrate_star_tracker",
    "cross_validate_attitude",
    "switch_attitude_reference",
    "recalibrate_imu",
    "query_power_level",
    "query_attitude",
    "query_thermal",
    "diagnostic_scan",
    "safe_mode",
    "resume_nominal",
)

TARGET_ORDER: dict[str, tuple[str, ...]] = {
    "shed_load": ("science_a", "science_b", "heaters", "transponder"),
    "restore_load": ("science_a", "science_b", "heaters", "transponder"),
    "reconfigure_power": ("solar_a", "solar_b"),
    "switch_attitude_reference": ("star_tracker", "gyro", "sun_sensor"),
    "query_power_level": ("battery", "solar_a", "solar_b"),
    "diagnostic_scan": ("power", "attitude", "comms"),
}


@dataclass(frozen=True)
class ActionSpec:
    index: int
    command: str
    target: Optional[str] = None

    @property
    def label(self) -> str:
        if self.target is None:
            return self.command
        return f"{self.command}:{self.target}"

    def to_action(self) -> SpaceFaultAction:
        return SpaceFaultAction(command=self.command, target=self.target)

    def to_dict(self) -> dict[str, str | int | None]:
        return {
            "index": self.index,
            "command": self.command,
            "target": self.target,
            "label": self.label,
        }


def build_action_space() -> list[ActionSpec]:
    actions: list[ActionSpec] = []
    for command in COMMAND_ORDER:
        if command not in VALID_COMMANDS:
            raise ValueError(f"Unknown command in training action order: {command}")

        targets = TARGET_ORDER.get(command)
        if targets is None:
            actions.append(ActionSpec(index=len(actions), command=command))
            continue

        expected_targets = TARGETED_COMMANDS[command]
        for target in targets:
            if target not in expected_targets:
                raise ValueError(f"Invalid training target {target!r} for {command!r}")
            actions.append(ActionSpec(index=len(actions), command=command, target=target))

    return actions
