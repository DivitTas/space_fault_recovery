# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Space Fault Recovery Environment: a partially-observable spacecraft cascade."""

import random
from dataclasses import dataclass, field
from typing import Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import SpaceFaultAction, SpaceFaultObservation
    from ..models import VALID_COMMANDS, TARGETED_COMMANDS
except ImportError:
    import sys as _sys, os as _os
    _pkg_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    if _pkg_root not in _sys.path:
        _sys.path.insert(0, _pkg_root)
    from models import SpaceFaultAction, SpaceFaultObservation
    from models import VALID_COMMANDS, TARGETED_COMMANDS


MAX_STEPS = 50
SOLAR_PANEL_PEAK_W = 90.0
THRUSTER_FUEL_PER_STEP = 1.5
DESAT_FUEL_COST = 4.0
STABILIZE_FUEL_COST = 2.0
STABLE_STEPS_REQUIRED = 3


FAULT_LIBRARY = (
    "solar_a_degraded",
    "solar_b_degraded",
    "battery_drain",
    "rw_fault",
    "attitude_drift",
    "thermal_fault",
    "comms_degraded",
)


@dataclass
class SpacecraftState:
    # Hidden hardware
    solar_a_health: float = 1.0
    solar_b_health: float = 1.0
    battery_health: float = 1.0
    backup_battery_health: float = 1.0
    battery_pct: float = 100.0
    battery_drain_rate: float = 0.0
    backup_battery_pct: float = 100.0
    on_backup_battery: bool = False
    battery_temp_c: float = 20.0
    power_controller_fault: bool = False
    # Attitude
    rw_degradation: float = 0.0
    rw_status: str = "nominal"
    attitude_fault: bool = False
    attitude_error: float = 0.0
    star_tracker_bias: float = 0.0
    gyro_bias: float = 0.0
    attitude_reference: str = "star_tracker"
    attitude_mode: str = "nominal"
    fuel_units: float = 50.0
    # Comms / thermal flags
    thermal_fault: bool = False
    comms_fault: bool = False
    # Subsystems
    science_a_online: bool = True
    science_b_online: bool = True
    heaters_online: bool = True
    transponder_online: bool = True
    transponder_pending_powercycle: bool = False
    safe_mode: bool = False
    # Episode tracking
    active_faults: list = field(default_factory=list)
    cleared_faults: list = field(default_factory=list)
    diagnosed_faults: set = field(default_factory=set)  # faults the agent has diagnosed
    step: int = 0
    mission_status: str = "nominal"
    last_action_result: str = "ok"
    consecutive_stable_steps: int = 0


class SpaceFaultRecoveryEnvironment(Environment):
    """Cascade-based spacecraft fault recovery environment."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._sc = SpacecraftState()
        self._rng = random.Random()
        self._episode_seed = 0

    @property
    def state(self) -> State:
        return self._state

    def reset(self, seed: Optional[int] = None) -> SpaceFaultObservation:
        self._episode_seed = seed if seed is not None else random.randint(0, 2**31 - 1)
        self._rng = random.Random(self._episode_seed)
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._sc = SpacecraftState()

        n_faults = self._rng.randint(1, 3)
        faults = self._rng.sample(FAULT_LIBRARY, n_faults)
        self._sc.active_faults = list(faults)

        for fault in faults:
            self._inject_fault(fault)

        if self._sc.active_faults:
            self._sc.mission_status = "degraded"
        self._sc.last_action_result = (
            f"episode initialized with {n_faults} fault(s); telemetry shows anomalies"
        )
        return self._build_observation()

    def _inject_fault(self, fault: str) -> None:
        sc = self._sc
        if fault == "solar_a_degraded":
            sc.solar_a_health = self._rng.uniform(0.05, 0.35)
        elif fault == "solar_b_degraded":
            sc.solar_b_health = self._rng.uniform(0.05, 0.35)
        elif fault == "battery_drain":
            sc.battery_health = self._rng.uniform(0.4, 0.7)
            sc.battery_pct = self._rng.uniform(55.0, 75.0)
            sc.power_controller_fault = True
        elif fault == "rw_fault":
            sc.rw_degradation = self._rng.uniform(0.40, 0.70)
            sc.rw_status = "degraded"
            # Vibration from degraded wheels introduces IMU noise
            sc.gyro_bias = sc.rw_degradation * 2.0
        elif fault == "attitude_drift":
            sc.attitude_fault = True
            sc.star_tracker_bias = self._rng.uniform(3.0, 8.0)
            sc.attitude_error = self._rng.uniform(2.0, 5.0)
        elif fault == "thermal_fault":
            sc.thermal_fault = True
            sc.heaters_online = False
            sc.battery_temp_c = self._rng.uniform(5.0, 15.0)
        elif fault == "comms_degraded":
            sc.comms_fault = True

    def step(self, action: SpaceFaultAction) -> SpaceFaultObservation:  # type: ignore[override]
        sc = self._sc
        if sc.mission_status in ("recovered", "lost") or sc.step >= MAX_STEPS:
            return self._build_observation(done=True, reward=0.0)
        self._state.step_count += 1
        sc.step += 1
        reward = 0.0

        command = action.command
        target = action.target

        if command not in VALID_COMMANDS:
            sc.last_action_result = f"error: unknown command '{command}'"
            reward -= 1.0
            self._tick_physics()
            reward += self._evaluate_status()
            return self._finalize(reward)

        if command in TARGETED_COMMANDS:
            valid_targets = TARGETED_COMMANDS[command]
            if target is None:
                sc.last_action_result = f"error: '{command}' requires a target"
                reward -= 1.0
                self._tick_physics()
                reward += self._evaluate_status()
                return self._finalize(reward)
            if target not in valid_targets:
                sc.last_action_result = (
                    f"error: invalid target '{target}' for '{command}' "
                    f"(expected one of {sorted(valid_targets)})"
                )
                reward -= 1.0
                self._tick_physics()
                reward += self._evaluate_status()
                return self._finalize(reward)

        reward += self._apply_command(command, target)
        self._tick_physics()
        reward += self._evaluate_status()
        return self._finalize(reward)

    def _apply_command(self, command: str, target: Optional[str]) -> float:
        sc = self._sc
        reward = 0.0

        if command == "shed_load":
            attr = f"{target}_online"
            if not getattr(sc, attr):
                sc.last_action_result = f"{target} already offline"
                reward -= 0.2
            else:
                setattr(sc, attr, False)
                sc.last_action_result = f"{target} powered off"
                if target == "transponder":
                    sc.transponder_pending_powercycle = True
                if sc.battery_pct < 60.0 or sc.power_controller_fault:
                    reward += 0.3
                else:
                    reward -= 0.1

        elif command == "restore_load":
            attr = f"{target}_online"
            if getattr(sc, attr):
                if target == "transponder" and sc.comms_fault:
                    sc.last_action_result = "comms fault active — power cycle required: shed_load then restore_load"
                    reward -= 0.1
                else:
                    sc.last_action_result = f"{target} already online"
                    reward -= 0.2
            elif sc.battery_pct < 25.0:
                sc.last_action_result = (
                    f"refused: battery {sc.battery_pct:.1f}% too low to restore {target}"
                )
                reward -= 0.5
            else:
                setattr(sc, attr, True)
                sc.last_action_result = f"{target} restored"
                reward += 0.1
                if target == "heaters" and sc.thermal_fault and sc.battery_temp_c > -2.0:
                    # Symptom fix: always clear the thermal_fault flag
                    sc.thermal_fault = False
                    if "thermal_fault" in sc.diagnosed_faults and "thermal_fault" in sc.active_faults:
                        sc.active_faults.remove("thermal_fault")
                        sc.cleared_faults.append("thermal_fault")
                        sc.last_action_result = "heaters restored; thermal fault cleared"
                        reward += 0.8
                    else:
                        sc.last_action_result = "heaters restored; temperature stabilizing"
                        reward += 0.2
                elif target == "transponder" and sc.comms_fault and sc.transponder_pending_powercycle:
                    # Symptom fix: always clear the comms_fault flag
                    sc.comms_fault = False
                    sc.transponder_pending_powercycle = False
                    if "comms_degraded" in sc.diagnosed_faults and "comms_degraded" in sc.active_faults:
                        sc.active_faults.remove("comms_degraded")
                        sc.cleared_faults.append("comms_degraded")
                        sc.last_action_result = "transponder power-cycled; comms fault cleared"
                        reward += 0.8
                    else:
                        sc.last_action_result = "transponder power-cycled; link partially restored"
                        reward += 0.2

        elif command == "switch_to_backup_battery":
            if sc.on_backup_battery:
                sc.last_action_result = "already on backup battery"
                reward -= 0.3
            else:
                sc.on_backup_battery = True
                sc.battery_pct = sc.backup_battery_pct
                sc.last_action_result = "switched to backup battery"
                if "battery_drain" in sc.active_faults:
                    reward += 0.5

        elif command == "reset_power_controller":
            if not sc.power_controller_fault:
                sc.last_action_result = "power controller nominal; reset had no effect"
                reward -= 0.2
            else:
                # Symptom fix: always clear the fault flag
                sc.power_controller_fault = False
                if "battery_drain" in sc.diagnosed_faults and "battery_drain" in sc.active_faults:
                    sc.active_faults.remove("battery_drain")
                    sc.cleared_faults.append("battery_drain")
                    sc.last_action_result = "power controller reset; drain fault cleared"
                    reward += 0.8
                else:
                    sc.last_action_result = "power controller reset; bus stabilized"
                    reward += 0.2

        elif command == "reconfigure_power":
            # Solar fault recovery: multi-step path.
            # The agent must first diagnose the panel (query_power_level(solar_X))
            # to know it's degraded, then reconfigure.
            panel = target  # "solar_a" or "solar_b"
            fault_name = f"{panel}_degraded"
            health_attr = f"{panel}_health"
            current_health = getattr(sc, health_attr)

            if fault_name not in sc.active_faults:
                sc.last_action_result = f"{panel} has no active fault to reconfigure"
                reward -= 0.3
            elif fault_name not in sc.diagnosed_faults:
                # Blind reconfigure without diagnosis: partial effect + penalty
                # The agent guessed instead of diagnosing first
                new_health = min(1.0, current_health + 0.10)
                setattr(sc, health_attr, new_health)
                sc.last_action_result = (
                    f"{panel} reconfigured blindly; marginal improvement "
                    f"(health {current_health:.2f} -> {new_health:.2f})"
                )
                reward -= 0.1  # net negative to discourage blind attempts
            else:
                # Diagnosed reconfigure: significant restoration
                new_health = min(1.0, current_health + 0.40)
                setattr(sc, health_attr, new_health)
                if new_health >= 0.50:
                    sc.active_faults.remove(fault_name)
                    sc.cleared_faults.append(fault_name)
                    sc.last_action_result = (
                        f"{panel} reconfigured successfully; fault cleared "
                        f"(health {current_health:.2f} -> {new_health:.2f})"
                    )
                    reward += 1.0
                else:
                    sc.last_action_result = (
                        f"{panel} partially reconfigured "
                        f"(health {current_health:.2f} -> {new_health:.2f}); "
                        f"repeat reconfigure may be needed"
                    )
                    reward += 0.3

        elif command == "stabilize_attitude":
            if sc.fuel_units < STABILIZE_FUEL_COST:
                sc.last_action_result = "insufficient fuel to stabilize"
                reward -= 0.3
            else:
                sc.fuel_units -= STABILIZE_FUEL_COST
                sc.attitude_error = max(0.0, sc.attitude_error - 2.5)
                if sc.attitude_mode == "tumbling" and sc.attitude_error < 3.0:
                    sc.attitude_mode = "thruster"
                sc.last_action_result = (
                    f"attitude stabilization burn; pointing error {sc.attitude_error:.2f}deg"
                )
                reward += 0.3

        elif command == "switch_to_thruster_control":
            if sc.attitude_mode == "thruster":
                sc.last_action_result = "already in thruster mode"
                reward -= 0.2
            else:
                sc.attitude_mode = "thruster"
                sc.last_action_result = "ACS switched to thruster control"
                if sc.rw_status == "failed":
                    reward += 0.5

        elif command == "desaturate_wheels":
            if sc.fuel_units < DESAT_FUEL_COST:
                sc.last_action_result = "insufficient fuel to desaturate"
                reward -= 0.3
            elif sc.rw_degradation < 0.1:
                sc.last_action_result = "wheels already nominal; desaturation unnecessary"
                reward -= 0.1
            else:
                # Symptom fix: always desaturate at full effectiveness
                sc.fuel_units -= DESAT_FUEL_COST
                sc.rw_degradation = max(0.0, sc.rw_degradation - 0.50)
                if sc.rw_degradation < 0.4 and sc.rw_status != "failed":
                    sc.rw_status = "nominal"
                    if "rw_fault" in sc.diagnosed_faults and "rw_fault" in sc.active_faults:
                        sc.active_faults.remove("rw_fault")
                        sc.cleared_faults.append("rw_fault")
                        sc.last_action_result = (
                            f"wheels desaturated; rw_fault cleared (degradation {sc.rw_degradation:.2f})"
                        )
                        reward += 0.5
                    else:
                        sc.last_action_result = (
                            f"wheels desaturated; degradation now {sc.rw_degradation:.2f}"
                        )
                        reward += 0.2
                else:
                    sc.last_action_result = (
                        f"wheels desaturated; degradation now {sc.rw_degradation:.2f}"
                    )
                    reward += 0.2

        elif command == "recalibrate_star_tracker":
            if not sc.attitude_fault:
                sc.last_action_result = "star tracker nominal; recalibration unnecessary"
                reward -= 0.2
            else:
                # Symptom fix: always clear bias and reduce error
                sc.attitude_fault = False
                sc.star_tracker_bias = 0.0
                sc.attitude_error = max(0.0, sc.attitude_error - 1.0)
                if "attitude_drift" in sc.diagnosed_faults and "attitude_drift" in sc.active_faults:
                    sc.active_faults.remove("attitude_drift")
                    sc.cleared_faults.append("attitude_drift")
                    sc.last_action_result = "star tracker recalibrated; drift fault cleared"
                    reward += 1.0
                else:
                    sc.last_action_result = "star tracker recalibrated; bias removed"
                    reward += 0.3

        elif command == "cross_validate_attitude":
            st_reading = sc.attitude_error + sc.star_tracker_bias
            gy_reading = sc.attitude_error + sc.gyro_bias
            sun_reading = sc.attitude_error
            sc.last_action_result = (
                f"diag: ST={st_reading:.2f}deg GY={gy_reading:.2f}deg "
                f"SUN={sun_reading:.2f}deg disagree={abs(st_reading - sun_reading):.2f}"
            )
            if sc.attitude_fault:
                sc.diagnosed_faults.add("attitude_drift")
            if sc.comms_fault:
                sc.diagnosed_faults.add("comms_degraded")
            reward += 0.1 if sc.attitude_fault else 0.0

        elif command == "switch_attitude_reference":
            sc.attitude_reference = target
            sc.last_action_result = f"attitude reference set to {target}"
            # Switching away from a faulty star tracker to sun_sensor during attitude_fault is helpful
            if target == "sun_sensor" and sc.attitude_fault:
                reward += 0.3
            elif target == "star_tracker" and sc.attitude_fault:
                reward -= 0.3

        elif command == "recalibrate_imu":
            if sc.gyro_bias > 0.5:
                sc.last_action_result = f"IMU recalibrated; gyro bias {sc.gyro_bias:.2f} cleared"
                sc.gyro_bias = 0.0
                reward += 0.4
            else:
                sc.gyro_bias = 0.0
                sc.last_action_result = "IMU recalibrated (bias was negligible)"
                reward += 0.05

        elif command == "query_power_level":
            if target == "battery":
                sc.last_action_result = (
                    f"diag: battery_pct={sc.battery_pct:.1f}% "
                    f"health={sc.battery_health:.2f} backup={sc.on_backup_battery} "
                    f"pcf={sc.power_controller_fault}"
                )
                if sc.power_controller_fault:
                    sc.diagnosed_faults.add("battery_drain")
            elif target == "solar_a":
                sc.last_action_result = (
                    f"diag: solar_a output={self._solar_output('a'):.1f}W "
                    f"health={sc.solar_a_health:.2f} "
                    f"(panel {'DEGRADED' if sc.solar_a_health < 0.5 else 'nominal'})"
                )
                if sc.solar_a_health < 0.5:
                    sc.diagnosed_faults.add("solar_a_degraded")
            else:
                sc.last_action_result = (
                    f"diag: solar_b output={self._solar_output('b'):.1f}W "
                    f"health={sc.solar_b_health:.2f} "
                    f"(panel {'DEGRADED' if sc.solar_b_health < 0.5 else 'nominal'})"
                )
                if sc.solar_b_health < 0.5:
                    sc.diagnosed_faults.add("solar_b_degraded")
            reward += 0.05

        elif command == "query_attitude":
            sc.last_action_result = (
                f"diag: mode={sc.attitude_mode} err={sc.attitude_error:.2f}deg "
                f"rw={sc.rw_status} ref={sc.attitude_reference} "
                f"st_bias={sc.star_tracker_bias:.1f} gyro_bias={sc.gyro_bias:.1f}"
            )
            if sc.rw_degradation >= 0.4:
                sc.diagnosed_faults.add("rw_fault")
            reward += 0.05

        elif command == "query_thermal":
            sc.last_action_result = (
                f"diag: battery_temp={sc.battery_temp_c:.1f}C "
                f"heaters={'online' if sc.heaters_online else 'offline'} "
                f"thermal_fault={sc.thermal_fault}"
            )
            if sc.thermal_fault:
                sc.diagnosed_faults.add("thermal_fault")
            reward += 0.05

        elif command == "diagnostic_scan":
            if target == "power":
                load_w = self._current_load_w()
                solar_w = self._solar_output("a") + self._solar_output("b")
                sc.last_action_result = (
                    f"diag-scan power: bat={sc.battery_pct:.1f}% "
                    f"solar={solar_w:.1f}W load={load_w:.1f}W "
                    f"net={'positive' if solar_w >= load_w else 'negative'} "
                    f"pcf={sc.power_controller_fault}"
                )
                if sc.power_controller_fault:
                    sc.diagnosed_faults.add("battery_drain")
            elif target == "attitude":
                sc.last_action_result = (
                    f"diag-scan attitude: mode={sc.attitude_mode} "
                    f"err={sc.attitude_error:.2f}deg rw_deg={sc.rw_degradation:.2f} "
                    f"st_fault={sc.attitude_fault} ref={sc.attitude_reference}"
                )
                if sc.attitude_fault:
                    sc.diagnosed_faults.add("attitude_drift")
                if sc.rw_degradation >= 0.4:
                    sc.diagnosed_faults.add("rw_fault")
            else:  # target == "comms"
                signal_db = 25.0 if sc.transponder_online else 0.0
                if sc.comms_fault and sc.transponder_online:
                    signal_db = 8.0
                sc.last_action_result = (
                    f"diag-scan comms: transponder={'online' if sc.transponder_online else 'offline'} "
                    f"signal={signal_db:.1f}dB comms_fault={sc.comms_fault} "
                    f"pending_cycle={sc.transponder_pending_powercycle}"
                )
                if sc.comms_fault:
                    sc.diagnosed_faults.add("comms_degraded")
            reward += 0.1

        elif command == "safe_mode":
            sc.safe_mode = True
            sc.science_a_online = False
            sc.science_b_online = False
            sc.transponder_online = False
            if sc.attitude_mode == "tumbling":
                sc.attitude_mode = "thruster"
            sc.attitude_error = max(0.0, sc.attitude_error - 1.0)
            sc.last_action_result = "safe mode engaged; non-essentials shed"
            reward += 0.5 if sc.mission_status == "critical" else -0.2

        elif command == "resume_nominal":
            # ── Gate 1: all injected faults must be resolved ──
            if sc.active_faults:
                sc.last_action_result = (
                    f"refused: {len(sc.active_faults)} unresolved fault(s) — "
                    f"resolve before resuming nominal"
                )
                reward -= 0.5
            # ── Gate 2: hardware boolean flags clean ──
            elif (
                sc.power_controller_fault
                or sc.attitude_fault
                or sc.thermal_fault
                or sc.comms_fault
                or sc.rw_status == "failed"
            ):
                sc.last_action_result = "refused: subsystem hardware faults still active"
                reward -= 0.5
            # ── Gate 3: rw must be nominal (not just "not failed") ──
            elif sc.rw_status != "nominal":
                sc.last_action_result = (
                    f"refused: reaction wheels {sc.rw_status} — desaturate first"
                )
                reward -= 0.3
            # ── Gate 4: state margins ──
            elif sc.attitude_error > 2.0 or sc.battery_pct < 40.0:
                sc.last_action_result = (
                    f"refused: state not stable (err={sc.attitude_error:.1f}, "
                    f"bat={sc.battery_pct:.1f}%)"
                )
                reward -= 0.3
            # ── Gate 5: comms must be live ──
            elif not sc.transponder_online:
                sc.last_action_result = "refused: transponder offline — restore comms first"
                reward -= 0.3
            # ── Gate 6: stability duration ──
            elif sc.consecutive_stable_steps < STABLE_STEPS_REQUIRED:
                sc.last_action_result = (
                    f"refused: need {STABLE_STEPS_REQUIRED} consecutive stable steps "
                    f"(have {sc.consecutive_stable_steps})"
                )
                reward -= 0.1
            else:
                sc.safe_mode = False
                sc.science_a_online = True
                sc.science_b_online = True
                sc.transponder_online = True
                sc.heaters_online = True
                sc.attitude_mode = "nominal"
                sc.mission_status = "recovered"
                sc.last_action_result = "nominal operations resumed; mission recovered"

        return reward

    def _tick_physics(self) -> None:
        sc = self._sc

        solar_w = self._solar_output("a") + self._solar_output("b")
        load_w = self._current_load_w()
        net_w = solar_w - load_w
        if sc.power_controller_fault:
            net_w -= 25.0
        delta_pct = net_w / 30.0

        prev_pct = sc.battery_pct
        max_capacity = 100.0 * (sc.backup_battery_health if sc.on_backup_battery else sc.battery_health)
        sc.battery_pct = max(0.0, min(max_capacity, sc.battery_pct + delta_pct))
        sc.battery_drain_rate = sc.battery_pct - prev_pct

        if sc.thermal_fault and not sc.heaters_online:
            sc.battery_temp_c -= 0.8
        elif sc.heaters_online:
            sc.battery_temp_c += 0.3 * (20.0 - sc.battery_temp_c) / 20.0
        if sc.battery_temp_c < 0.0:
            sc.battery_health = max(0.3, sc.battery_health - 0.01)

        if sc.rw_status != "failed":
            if sc.rw_degradation > 0.0:
                sc.rw_degradation = min(1.0, sc.rw_degradation + 0.02)
                # Increasing vibration builds gyro noise
                sc.gyro_bias = min(8.0, sc.gyro_bias + sc.rw_degradation * 0.05)
                if sc.rw_degradation >= 0.95:
                    sc.rw_status = "failed"
                    if sc.attitude_mode == "nominal":
                        sc.attitude_mode = "tumbling"
                elif sc.rw_degradation >= 0.6:
                    sc.rw_status = "degraded"

        if sc.attitude_mode == "thruster":
            sc.fuel_units = max(0.0, sc.fuel_units - THRUSTER_FUEL_PER_STEP)
            if sc.fuel_units <= 0.0:
                sc.attitude_mode = "tumbling"

        if sc.attitude_mode == "tumbling":
            sc.attitude_error += 1.5
        elif sc.attitude_mode == "thruster":
            sc.attitude_error = max(0.0, sc.attitude_error - 0.2)
        else:
            # Drift rate depends on which reference sensor is active
            if sc.attitude_reference == "star_tracker":
                drift = 0.6 if sc.attitude_fault else -0.1
            elif sc.attitude_reference == "sun_sensor":
                # Sun sensor is unaffected by attitude_fault but less precise
                drift = 0.15
            else:  # gyro
                # Gyro drift scales with accumulated bias
                drift = 0.05 * sc.gyro_bias if sc.gyro_bias > 0.5 else -0.05
            sc.attitude_error = max(0.0, sc.attitude_error + drift)

        if sc.battery_pct <= 0.5:
            sc.transponder_online = False
            sc.heaters_online = False
            if sc.attitude_mode != "tumbling":
                sc.attitude_mode = "tumbling"

        # Track consecutive steps where ALL faults cleared and state is healthy.
        # active_faults is the authoritative source — if any remain, not stable.
        stable = (
            len(sc.active_faults) == 0
            and not sc.power_controller_fault
            and not sc.attitude_fault
            and not sc.thermal_fault
            and not sc.comms_fault
            and sc.rw_status == "nominal"
            and sc.battery_pct >= 40.0
            and sc.attitude_error < 3.0
            and not sc.safe_mode
            and sc.transponder_online
        )
        sc.consecutive_stable_steps = sc.consecutive_stable_steps + 1 if stable else 0

    def _solar_output(self, panel: str) -> float:
        sc = self._sc
        health = sc.solar_a_health if panel == "a" else sc.solar_b_health
        pointing_factor = max(0.1, 1.0 - sc.attitude_error / 20.0)
        return SOLAR_PANEL_PEAK_W * health * pointing_factor

    def _current_load_w(self) -> float:
        sc = self._sc
        load = 40.0
        if sc.science_a_online:
            load += 25.0
        if sc.science_b_online:
            load += 25.0
        if sc.heaters_online and sc.battery_temp_c < 18.0:
            load += 20.0
        if sc.transponder_online:
            load += 15.0
        if sc.attitude_mode == "thruster":
            load += 10.0
        return load

    def _evaluate_status(self) -> float:
        sc = self._sc
        reward = 0.0

        # If resume_nominal already set recovered, grant the terminal bonus
        if sc.mission_status == "recovered":
            return 10.0

        lost = (
            sc.battery_pct <= 0.0
            or sc.attitude_error >= 30.0
            or (sc.attitude_mode == "tumbling" and sc.fuel_units <= 0.0 and sc.rw_status == "failed")
        )
        if lost:
            sc.mission_status = "lost"
            return -10.0

        critical = (
            sc.battery_pct < 20.0
            or sc.attitude_error >= 15.0
            or sc.attitude_mode == "tumbling"
            or sc.battery_temp_c < 0.0
        )
        degraded = (
            len(sc.active_faults) > 0
            or sc.battery_pct < 50.0
            or sc.attitude_error >= 5.0
            or sc.rw_status != "nominal"
            or sc.safe_mode
        )

        # _evaluate_status never sets "recovered" — only resume_nominal can.
        if critical:
            sc.mission_status = "critical"
            reward -= 0.5
        elif degraded:
            sc.mission_status = "degraded"
            reward -= 0.05
        else:
            # All faults cleared and margins healthy, but agent hasn't
            # called resume_nominal yet — stay "degraded" to force the
            # explicit recovery action.
            sc.mission_status = "nominal"
            reward += 0.1

        return reward

    def _finalize(self, reward: float) -> SpaceFaultObservation:
        sc = self._sc
        done = sc.mission_status in ("recovered", "lost") or sc.step >= MAX_STEPS
        if sc.step >= MAX_STEPS and sc.mission_status not in ("recovered", "lost"):
            if sc.mission_status == "critical":
                sc.mission_status = "lost"
                reward -= 5.0
            else:
                reward -= 1.0
        return self._build_observation(done=done, reward=reward)

    def _build_observation(self, done: bool = False, reward: float = 0.0) -> SpaceFaultObservation:
        sc = self._sc

        signal_db = 25.0 if sc.transponder_online else 0.0
        if sc.comms_fault and sc.transponder_online:
            signal_db = 8.0
        if sc.attitude_error > 5.0 and sc.transponder_online:
            signal_db = max(0.0, signal_db - sc.attitude_error)

        if not sc.transponder_online or signal_db <= 1.0:
            transponder_status = "offline"
            link_bandwidth = "none"
        elif sc.comms_fault or signal_db < 12.0:
            transponder_status = "degraded"
            link_bandwidth = "low"
        else:
            transponder_status = "nominal"
            link_bandwidth = "high"

        bus_voltage = 28.0
        load_w = self._current_load_w()
        bus_voltage -= max(0.0, (load_w - 100.0) / 30.0)
        if sc.battery_pct < 30.0:
            bus_voltage -= (30.0 - sc.battery_pct) / 10.0
        bus_voltage = max(18.0, bus_voltage)

        subsystems = []
        if sc.science_a_online:
            subsystems.append("science_a")
        if sc.science_b_online:
            subsystems.append("science_b")
        if sc.heaters_online:
            subsystems.append("heaters")
        if sc.transponder_online:
            subsystems.append("transponder")

        return SpaceFaultObservation(
            battery_pct=round(sc.battery_pct, 2),
            battery_drain_rate=round(sc.battery_drain_rate, 3),
            solar_a_sensor_output_w=round(self._solar_output("a"), 2),
            solar_b_sensor_output_w=round(self._solar_output("b"), 2),
            bus_voltage=round(bus_voltage, 2),
            star_tracker_deg=round(sc.attitude_error + sc.star_tracker_bias, 3),
            gyro_deg=round(sc.attitude_error + sc.gyro_bias, 3),
            sun_sensor_deg=round(sc.attitude_error, 3),
            attitude_mode=sc.attitude_mode,
            rw_status=sc.rw_status,
            fuel_units=round(sc.fuel_units, 2),
            signal_strength_db=round(signal_db, 2),
            transponder_status=transponder_status,
            link_bandwidth=link_bandwidth,
            battery_temp_c=round(sc.battery_temp_c, 2),
            heater_status="online" if sc.heaters_online else "offline",
            subsystems_online=subsystems,
            step=sc.step,
            mission_status=sc.mission_status,
            last_action_result=sc.last_action_result,
            done=done,
            reward=float(reward),
            metadata={"episode_seed": self._episode_seed},
        )


if __name__ == "__main__":
    env = SpaceFaultRecoveryEnvironment()
    obs = env.reset(seed=42)
    print(f"reset: status={obs.mission_status} seed={obs.metadata['episode_seed']}")
    print(f"  bat={obs.battery_pct}% solar=({obs.solar_a_sensor_output_w}W,"
          f"{obs.solar_b_sensor_output_w}W) st_err={obs.star_tracker_deg}deg")

    seq = [
        # 1. Diagnose everything
        ("diagnostic_scan", "power"),
        ("diagnostic_scan", "attitude"),
        ("diagnostic_scan", "comms"),
        ("query_thermal", None),
        ("query_power_level", "battery"),     # diagnose battery_drain
        ("query_power_level", "solar_a"),     # diagnose solar_a
        ("query_power_level", "solar_b"),     # diagnose solar_b
        ("cross_validate_attitude", None),    # diagnose attitude_drift + comms_degraded
        ("query_attitude", None),             # diagnose rw_fault
        # 2. Shed non-essentials
        ("shed_load", "science_a"),
        ("shed_load", "science_b"),
        # 3. Fix faults (order matters)
        ("reset_power_controller", None),
        ("reconfigure_power", "solar_a"),
        ("reconfigure_power", "solar_b"),
        ("recalibrate_star_tracker", None),
        ("desaturate_wheels", None),
        ("desaturate_wheels", None),         # may need two passes
        ("recalibrate_imu", None),
        ("restore_load", "heaters"),
        ("shed_load", "transponder"),        # power-cycle comms: shed first
        ("restore_load", "transponder"),     # then restore to clear comms_degraded
        # 4. Stabilize attitude
        ("stabilize_attitude", None),
        ("stabilize_attitude", None),
        ("stabilize_attitude", None),
        # 5. Wait for stable steps, then resume
        ("query_attitude", None),          # burn stable steps
        ("query_thermal", None),
        ("query_power_level", "battery"),
        ("resume_nominal", None),
        ("resume_nominal", None),
        ("resume_nominal", None),
        ("resume_nominal", None),
    ]

    total_reward = 0.0
    for cmd, tgt in seq:
        obs = env.step(SpaceFaultAction(command=cmd, target=tgt))
        total_reward += obs.reward
        print(
            f"step {obs.step:02d} {cmd}({tgt}): r={obs.reward:+.2f} "
            f"status={obs.mission_status} bat={obs.battery_pct}% "
            f"err={obs.star_tracker_deg}deg -> {obs.last_action_result[:80]}"
        )
        if obs.done:
            print(f"DONE @ step {obs.step}: {obs.mission_status}")
            break

    print(f"total_reward={total_reward:.2f}")
