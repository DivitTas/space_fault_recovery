"""Quick CI smoke-test: heuristic recovery-rate thresholds for three baseline policies.

Does NOT import AdaptivePolicy — that's covered by scripted_policy.py's own CLI.

Run:
    python server/test_difficulty.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.scripted_policy import (  # noqa: E402
    BLIND_MACRO,
    SMART_MACRO,
    RandomPolicy,
    BlindMacroPolicy,
    SmartMacroPolicy,
    run_episode,
)
from server.space_fault_recovery_environment import SpaceFaultRecoveryEnvironment  # noqa: E402

N = 100

env = SpaceFaultRecoveryEnvironment()
policies = {
    "random": RandomPolicy(),
    "blind":  BlindMacroPolicy(),
    "smart":  SmartMacroPolicy(),
}

results = {}
for pname, policy in policies.items():
    episodes = [run_episode(env, policy, seed) for seed in range(N)]
    results[pname] = episodes

blind_recovered  = sum(1 for r in results["blind"]  if r.terminal_status == "recovered")
smart_recovered  = sum(1 for r in results["smart"]  if r.terminal_status == "recovered")
random_recovered = sum(1 for r in results["random"] if r.terminal_status == "recovered")

# Verify that smart macro never recovers with active faults still set.
for r in results["smart"]:
    if r.terminal_status == "recovered":
        assert not r.final_active_faults, (
            f"seed {r.seed}: recovered with active_faults={r.final_active_faults}"
        )

print(f"=== Difficulty Report ({N} episodes each) ===")
print(f"Blind macro (no diagnosis):  {blind_recovered}/{N} recovered")
print(f"Smart macro (with diagnosis): {smart_recovered}/{N} recovered")
print(f"Random policy:               {random_recovered}/{N} recovered")
print()

ok = True
if blind_recovered > 30:
    print("⚠ BLIND MACRO TOO EASY — should be <30%")
    ok = False
else:
    print("✓ Blind macro properly difficult")

if smart_recovered < 50:
    print("⚠ SMART MACRO TOO HARD — environment may be unlearnable")
    ok = False
else:
    print(f"✓ Smart macro recovers {smart_recovered}% — good signal for RL")

if random_recovered > 5:
    print(f"⚠ Random policy recovers {random_recovered}% — still too easy")
    ok = False
else:
    print("✓ Random policy properly fails")

if not ok:
    sys.exit(1)
