# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Baseline scripted policies for Space Fault Recovery difficulty calibration.

Four policies (Random, Blind, Smart, Adaptive) + rollout, aggregation,
and CLI.  Run:

    python server/scripted_policy.py --seeds 100 --json-out baselines.json
    python -m server.scripted_policy --seeds 5   # quick smoke
"""

import json
import random
import statistics
import sys
import os
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Iterable, Optional, Any

# ── sys.path shim so the script runs as `python server/scripted_policy.py`
# *and* as `python -m server.scripted_policy` from the package root.
_pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

try:
    from ..models import SpaceFaultAction, VALID_COMMANDS, TARGETED_COMMANDS
    from .space_fault_recovery_environment import (
        SpaceFaultRecoveryEnvironment,
        MAX_STEPS,
    )
except ImportError:
    from models import SpaceFaultAction, VALID_COMMANDS, TARGETED_COMMANDS  # type: ignore[no-redef]
    from server.space_fault_recovery_environment import (  # type: ignore[no-redef]
        SpaceFaultRecoveryEnvironment,
        MAX_STEPS,
    )


# ── Shared macro sequences ────────────────────────────────────────────────────
# These are the canonical definitions; test_difficulty.py imports from here.

BLIND_MACRO: list[tuple[str, Optional[str]]] = [
    ("shed_load", "science_a"),
    ("shed_load", "science_b"),
    ("reset_power_controller", None),
    ("reconfigure_power", "solar_a"),   # blind — no prior diagnosis
    ("reconfigure_power", "solar_b"),   # blind
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
]

SMART_MACRO: list[tuple[str, Optional[str]]] = [
    ("query_power_level", "solar_a"),
    ("query_power_level", "solar_b"),
    ("query_power_level", "battery"),        # diagnoses battery_drain
    ("diagnostic_scan", "power"),
    ("diagnostic_scan", "attitude"),
    ("diagnostic_scan", "comms"),            # diagnoses comms_degraded
    ("query_thermal", None),
    ("cross_validate_attitude", None),       # diagnoses attitude_drift + comms_degraded
    ("query_attitude", None),               # diagnoses rw_fault
    ("shed_load", "science_a"),
    ("shed_load", "science_b"),
    ("reset_power_controller", None),
    ("reconfigure_power", "solar_a"),
    ("reconfigure_power", "solar_b"),
    ("reconfigure_power", "solar_a"),        # second pass for severely degraded panels
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
]

# Cheap no-op filler used when a macro finishes early.
_FILLER = ("query_thermal", None)


# ── Policy interface ──────────────────────────────────────────────────────────

class Policy:
    """Abstract base for all scripted policies."""

    name: str = "base"

    def reset(self, seed: int) -> None:
        """Called once before each episode; seed is the episode seed."""

    def __call__(self, obs: Any, step_idx: int, rng: random.Random) -> SpaceFaultAction:
        raise NotImplementedError


# ── Random policy ─────────────────────────────────────────────────────────────

class RandomPolicy(Policy):
    """Uniformly random command selection."""

    name = "random"

    def __call__(self, obs: Any, step_idx: int, rng: random.Random) -> SpaceFaultAction:
        cmd = rng.choice(sorted(VALID_COMMANDS))
        target: Optional[str] = None
        if cmd in TARGETED_COMMANDS:
            target = rng.choice(sorted(TARGETED_COMMANDS[cmd]))
        return SpaceFaultAction(command=cmd, target=target)


# ── Blind macro policy ────────────────────────────────────────────────────────

class BlindMacroPolicy(Policy):
    """Replays BLIND_MACRO; pads with cheap filler once exhausted."""

    name = "blind"

    def reset(self, seed: int) -> None:
        self._queue: list[tuple[str, Optional[str]]] = list(BLIND_MACRO)

    def __call__(self, obs: Any, step_idx: int, rng: random.Random) -> SpaceFaultAction:
        if self._queue:
            cmd, tgt = self._queue.pop(0)
        else:
            cmd, tgt = _FILLER
        return SpaceFaultAction(command=cmd, target=tgt)


# ── Smart macro policy ────────────────────────────────────────────────────────

class SmartMacroPolicy(Policy):
    """Replays SMART_MACRO; pads with cheap filler once exhausted."""

    name = "smart"

    def reset(self, seed: int) -> None:
        self._queue: list[tuple[str, Optional[str]]] = list(SMART_MACRO)

    def __call__(self, obs: Any, step_idx: int, rng: random.Random) -> SpaceFaultAction:
        if self._queue:
            cmd, tgt = self._queue.pop(0)
        else:
            cmd, tgt = _FILLER
        return SpaceFaultAction(command=cmd, target=tgt)


# ── Adaptive policy ───────────────────────────────────────────────────────────

# Per-fault repair plan: list of (command, target) steps to fix that fault.
_FAULT_REPAIRS: dict[str, list[tuple[str, Optional[str]]]] = {
    "solar_a_degraded": [
        ("reconfigure_power", "solar_a"),
        ("reconfigure_power", "solar_a"),
    ],
    "solar_b_degraded": [
        ("reconfigure_power", "solar_b"),
        ("reconfigure_power", "solar_b"),
    ],
    "battery_drain": [
        ("shed_load", "science_a"),
        ("shed_load", "science_b"),
        ("reset_power_controller", None),
    ],
    "rw_fault": [
        ("desaturate_wheels", None),
        ("desaturate_wheels", None),
    ],
    "attitude_drift": [
        ("recalibrate_star_tracker", None),
        ("recalibrate_imu", None),
    ],
    "thermal_fault": [
        ("restore_load", "heaters"),
    ],
    "comms_degraded": [
        ("shed_load", "transponder"),
        ("restore_load", "transponder"),
    ],
}

# Full Phase-A diagnostic sweep (runs once per episode).
_DIAGNOSE_STEPS: list[tuple[str, Optional[str]]] = [
    ("query_power_level", "battery"),
    ("query_power_level", "solar_a"),
    ("query_power_level", "solar_b"),
    ("diagnostic_scan", "power"),
    ("diagnostic_scan", "attitude"),
    ("diagnostic_scan", "comms"),
    ("query_attitude", None),
    ("query_thermal", None),
    ("cross_validate_attitude", None),
]
_DIAGNOSE_LENGTH = len(_DIAGNOSE_STEPS)   # 9 steps (indices 0-8)

_MAX_RESUME_RETRIES = 3


class AdaptivePolicy(Policy):
    """Three-phase observation-driven policy.

    Phase A (steps 0-8):  Diagnostic sweep.
    Phase B (steps 9..): Targeted repairs derived from diagnosed faults,
                         observation-gated attitude burns, then subsystem restores.
    Phase C (last 3+):   resume_nominal with bounded retry.
    """

    name = "adaptive"

    def __init__(self, env_ref: SpaceFaultRecoveryEnvironment):
        # We need a reference to read env._sc.diagnosed_faults after Phase A.
        # This is the same hidden-state peek test_difficulty.py uses for active_faults.
        self._env = env_ref

    def reset(self, seed: int) -> None:
        self._phase: str = "diagnose"
        self._queue: list[tuple[str, Optional[str]]] = list(_DIAGNOSE_STEPS)
        self._resume_retries: int = 0

    def __call__(self, obs: Any, step_idx: int, rng: random.Random) -> SpaceFaultAction:
        # ── Phase A complete → build repair queue ──────────────────────────
        if self._phase == "diagnose" and not self._queue:
            self._phase = "repair"
            detected = set(self._env._sc.diagnosed_faults)
            repair_steps: list[tuple[str, Optional[str]]] = []
            for fault in sorted(detected):
                repair_steps.extend(_FAULT_REPAIRS.get(fault, []))
            # Post-repair: restore science if battery is healthy (checked live in __call__)
            repair_steps.append(("_restore_science_conditional", None))
            # Stabilise attitude while pointing error > 1.0
            repair_steps.extend([
                ("_stabilize_if_needed", None),
                ("_stabilize_if_needed", None),
                ("_stabilize_if_needed", None),
            ])
            self._queue = repair_steps

        # ── Drain the queue ────────────────────────────────────────────────
        while self._queue:
            cmd, tgt = self._queue[0]

            # Virtual command: conditionally restore science subsystems.
            if cmd == "_restore_science_conditional":
                self._queue.pop(0)
                if obs.battery_pct >= 30.0:
                    return SpaceFaultAction(command="restore_load", target="science_a")
                # Battery too low — skip but emit a cheap diagnostic
                return SpaceFaultAction(command="query_power_level", target="battery")

            # Virtual command: stabilise only when error is non-trivial.
            if cmd == "_stabilize_if_needed":
                self._queue.pop(0)
                if obs.sun_sensor_deg > 1.0:
                    return SpaceFaultAction(command="stabilize_attitude", target=None)
                # Already stable — slide in a diagnostic filler instead.
                return SpaceFaultAction(command="query_attitude", target=None)

            self._queue.pop(0)
            return SpaceFaultAction(command=cmd, target=tgt)

        # ── Phase C: resume_nominal with bounded retry ─────────────────────
        if self._resume_retries < _MAX_RESUME_RETRIES:
            self._resume_retries += 1
            # On retry > 1, re-inspect hidden state for missed faults and push repairs.
            if self._resume_retries > 1:
                remaining = list(self._env._sc.active_faults)
                extra: list[tuple[str, Optional[str]]] = []
                for fault in sorted(remaining):
                    extra.extend(_FAULT_REPAIRS.get(fault, []))
                if extra:
                    self._queue = extra
                    cmd2, tgt2 = self._queue.pop(0)
                    return SpaceFaultAction(command=cmd2, target=tgt2)
            return SpaceFaultAction(command="resume_nominal", target=None)

        # Fallback: cheap filler.
        return SpaceFaultAction(command=_FILLER[0], target=_FILLER[1])


# ── EpisodeResult dataclass ───────────────────────────────────────────────────

@dataclass
class EpisodeResult:
    policy: str
    seed: int
    fault_combo: tuple  # sorted tuple of injected fault names
    terminal_status: str   # "recovered" | "lost" | "timeout"
    total_return: float
    steps_to_terminal: int
    final_active_faults: tuple  # faults still unresolved at episode end


# ── Rollout function ──────────────────────────────────────────────────────────

def run_episode(
    env: SpaceFaultRecoveryEnvironment,
    policy: Policy,
    seed: int,
) -> EpisodeResult:
    """Run one episode and return a structured result."""
    obs = env.reset(seed=seed)
    fault_combo = tuple(sorted(env._sc.active_faults))   # snapshot at reset
    policy.reset(seed)
    rng = random.Random(seed + 10_000)

    total_return = 0.0
    steps = 0

    while not obs.done and steps < MAX_STEPS:
        action = policy(obs, steps, rng)
        obs = env.step(action)
        total_return += float(obs.reward)
        steps += 1

    if obs.mission_status == "recovered":
        terminal = "recovered"
    elif obs.mission_status == "lost":
        terminal = "lost" if steps < MAX_STEPS else "timeout"
    else:
        terminal = "timeout"

    return EpisodeResult(
        policy=policy.name,
        seed=seed,
        fault_combo=fault_combo,
        terminal_status=terminal,
        total_return=total_return,
        steps_to_terminal=steps,
        final_active_faults=tuple(sorted(env._sc.active_faults)),
    )


# ── Aggregation & reporting ───────────────────────────────────────────────────

@dataclass
class PolicyReport:
    name: str
    n_episodes: int
    recovery_rate: float           # 0.0–1.0
    mean_return: float
    std_return: float
    mean_steps: float
    terminal_breakdown: dict       # {"recovered": N, "lost": N, "timeout": N}
    fault_combo_table: dict        # combo_str -> {n, recovered, mean_return}


def summarize(results: list[EpisodeResult], name: str) -> PolicyReport:
    """Aggregate a list of same-policy EpisodeResult into a PolicyReport."""
    n = len(results)
    recovered = [r for r in results if r.terminal_status == "recovered"]
    breakdown: dict[str, int] = {"recovered": 0, "lost": 0, "timeout": 0}
    for r in results:
        breakdown[r.terminal_status] = breakdown.get(r.terminal_status, 0) + 1

    returns = [r.total_return for r in results]
    steps = [r.steps_to_terminal for r in results]

    # Per fault-combo breakdown
    combo_groups: dict[tuple, list[EpisodeResult]] = defaultdict(list)
    for r in results:
        combo_groups[r.fault_combo].append(r)

    combo_table: dict[str, dict] = {}
    for combo, group in sorted(combo_groups.items()):
        combo_str = "+".join(combo) if combo else "(none)"
        n_c = len(group)
        rec_c = sum(1 for r in group if r.terminal_status == "recovered")
        mean_ret_c = statistics.mean(r.total_return for r in group)
        combo_table[combo_str] = {
            "n": n_c,
            "recovered": rec_c,
            "recovery_rate": rec_c / n_c,
            "mean_return": round(mean_ret_c, 2),
        }

    return PolicyReport(
        name=name,
        n_episodes=n,
        recovery_rate=len(recovered) / n if n else 0.0,
        mean_return=statistics.mean(returns) if returns else 0.0,
        std_return=statistics.stdev(returns) if len(returns) > 1 else 0.0,
        mean_steps=statistics.mean(steps) if steps else 0.0,
        terminal_breakdown=breakdown,
        fault_combo_table=combo_table,
    )


def print_report(reports: list[PolicyReport]) -> None:
    """Print a human-readable comparison table to stdout."""
    COL = 12
    print()
    print("=" * 72)
    print("  SPACE FAULT RECOVERY — DIFFICULTY CALIBRATION REPORT")
    print("=" * 72)

    # Header
    headers = ["Policy", "N", "Recovery%", "MeanRet", "StdRet", "MeanSteps", "Lost", "Timeout"]
    print("  " + "  ".join(h.ljust(COL) for h in headers))
    print("  " + "-" * (len(headers) * (COL + 2)))

    for r in reports:
        row = [
            r.name.ljust(COL),
            str(r.n_episodes).ljust(COL),
            f"{r.recovery_rate * 100:.1f}%".ljust(COL),
            f"{r.mean_return:.2f}".ljust(COL),
            f"{r.std_return:.2f}".ljust(COL),
            f"{r.mean_steps:.1f}".ljust(COL),
            str(r.terminal_breakdown.get("lost", 0)).ljust(COL),
            str(r.terminal_breakdown.get("timeout", 0)).ljust(COL),
        ]
        print("  " + "  ".join(row))

    # Per-policy fault combo breakdown
    for r in reports:
        print()
        print(f"  ── {r.name.upper()} fault-combo breakdown ──")
        combo_hdr = ["Fault Combo", "N", "Recovery%", "MeanRet"]
        print("    " + "  ".join(h.ljust(20) for h in combo_hdr))
        print("    " + "-" * 80)
        for combo_str, stats_d in sorted(r.fault_combo_table.items()):
            row2 = [
                combo_str.ljust(20),
                str(stats_d["n"]).ljust(20),
                f"{stats_d['recovery_rate'] * 100:.1f}%".ljust(20),
                f"{stats_d['mean_return']:.2f}".ljust(20),
            ]
            print("    " + "  ".join(row2))

    print()
    print("=" * 72)
    print("  HEURISTIC THRESHOLDS")
    print("=" * 72)
    by_name = {r.name: r for r in reports}
    _check("random",   by_name, max_recovery=0.05,  label="< 5%  recovery")
    _check("blind",    by_name, max_recovery=0.30,  label="< 30% recovery")
    _check("smart",    by_name, min_recovery=0.50,  label="> 50% recovery")
    _check("adaptive", by_name, min_recovery=0.45,  label="> 45% recovery")
    print()


def _check(
    name: str,
    by_name: dict,
    min_recovery: float = 0.0,
    max_recovery: float = 1.0,
    label: str = "",
) -> None:
    if name not in by_name:
        return
    r = by_name[name]
    rate = r.recovery_rate
    ok = min_recovery <= rate <= max_recovery
    symbol = "✓" if ok else "⚠"
    print(f"  {symbol} {name:10s}  {rate * 100:.1f}% recovered  ({label})")


def dump_json(
    reports: list[PolicyReport],
    results: list[EpisodeResult],
    path: Path,
) -> None:
    """Write full data (per-episode + aggregated reports) to a JSON file."""
    payload = {
        "reports": [
            {
                "name": r.name,
                "n_episodes": r.n_episodes,
                "recovery_rate": r.recovery_rate,
                "mean_return": r.mean_return,
                "std_return": r.std_return,
                "mean_steps": r.mean_steps,
                "terminal_breakdown": r.terminal_breakdown,
                "fault_combo_table": r.fault_combo_table,
            }
            for r in reports
        ],
        "episodes": [
            {
                "policy": e.policy,
                "seed": e.seed,
                "fault_combo": list(e.fault_combo),
                "terminal_status": e.terminal_status,
                "total_return": e.total_return,
                "steps_to_terminal": e.steps_to_terminal,
                "final_active_faults": list(e.final_active_faults),
            }
            for e in results
        ],
    }
    path.write_text(json.dumps(payload, indent=2))
    print(f"  JSON written to {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

POLICY_NAMES = ["random", "blind", "smart", "adaptive"]


def build_policies(env: SpaceFaultRecoveryEnvironment) -> dict[str, Policy]:
    return {
        "random": RandomPolicy(),
        "blind": BlindMacroPolicy(),
        "smart": SmartMacroPolicy(),
        "adaptive": AdaptivePolicy(env),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Baseline scripted-policy calibration for Space Fault Recovery."
    )
    parser.add_argument(
        "--seeds", type=int, default=100,
        help="Number of episode seeds per policy (default: 100).",
    )
    parser.add_argument(
        "--policies", nargs="+", default=POLICY_NAMES,
        choices=POLICY_NAMES,
        help="Policies to evaluate (default: all four).",
    )
    parser.add_argument(
        "--json-out", type=Path, default=None,
        dest="json_out",
        help="Optional path to write full JSON results.",
    )
    args = parser.parse_args()

    env = SpaceFaultRecoveryEnvironment()
    policies = build_policies(env)

    all_results: list[EpisodeResult] = []
    reports: list[PolicyReport] = []

    for pname in args.policies:
        policy = policies[pname]
        episode_results: list[EpisodeResult] = []
        print(f"  Running {pname} × {args.seeds} episodes …", end="", flush=True)
        for seed in range(args.seeds):
            episode_results.append(run_episode(env, policy, seed))
        print(f"  done  ({sum(1 for r in episode_results if r.terminal_status == 'recovered')} recovered)")
        all_results.extend(episode_results)
        reports.append(summarize(episode_results, pname))

    print_report(reports)

    if args.json_out is not None:
        dump_json(reports, all_results, args.json_out)


if __name__ == "__main__":
    main()
