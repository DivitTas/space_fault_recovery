"""Quick difficulty check: blind macro, random policy, and fault verification."""
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import SpaceFaultAction, SpaceFaultObservation, VALID_COMMANDS, TARGETED_COMMANDS
from server.space_fault_recovery_environment import SpaceFaultRecoveryEnvironment

N = 100

# ── Blind macro (no diagnosis, just fire-and-forget) ──
BLIND_MACRO = [
    ("shed_load", "science_a"),
    ("shed_load", "science_b"),
    ("reset_power_controller", None),
    ("reconfigure_power", "solar_a"),  # blind — no prior diagnosis
    ("reconfigure_power", "solar_b"),  # blind
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

# ── Smart macro (diagnoses first, then fixes) ──
SMART_MACRO = [
    ("query_power_level", "solar_a"),
    ("query_power_level", "solar_b"),
    ("query_power_level", "battery"),        # diagnoses battery_drain
    ("diagnostic_scan", "power"),
    ("diagnostic_scan", "attitude"),
    ("diagnostic_scan", "comms"),             # diagnoses comms_degraded
    ("query_thermal", None),
    ("cross_validate_attitude", None),        # diagnoses attitude_drift + comms_degraded
    ("query_attitude", None),                 # diagnoses rw_fault
    ("shed_load", "science_a"),
    ("shed_load", "science_b"),
    ("reset_power_controller", None),
    ("reconfigure_power", "solar_a"),
    ("reconfigure_power", "solar_b"),
    ("reconfigure_power", "solar_a"),         # second pass for severely degraded panels
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


def run_macro(env, macro, seed):
    obs = env.reset(seed=seed)
    for cmd, tgt in macro:
        obs = env.step(SpaceFaultAction(command=cmd, target=tgt))
        if obs.done:
            break
    return obs


def run_random(env, seed):
    rng = random.Random(seed + 10000)
    obs = env.reset(seed=seed)
    for _ in range(50):
        cmd = rng.choice(list(VALID_COMMANDS))
        target = None
        if cmd in TARGETED_COMMANDS:
            target = rng.choice(list(TARGETED_COMMANDS[cmd]))
        obs = env.step(SpaceFaultAction(command=cmd, target=target))
        if obs.done:
            break
    return obs


env = SpaceFaultRecoveryEnvironment()

# Test 1: Blind macro
blind_recovered = 0
blind_faults_remaining = 0
for seed in range(N):
    obs = run_macro(env, BLIND_MACRO, seed)
    if obs.mission_status == "recovered":
        blind_recovered += 1
        # Check hidden state
        sc = env._sc
        if sc.active_faults:
            blind_faults_remaining += 1

# Test 2: Smart macro
smart_recovered = 0
for seed in range(N):
    obs = run_macro(env, SMART_MACRO, seed)
    if obs.mission_status == "recovered":
        smart_recovered += 1
        sc = env._sc
        assert not sc.active_faults, f"seed {seed}: recovered with {sc.active_faults}"

# Test 3: Random policy
random_recovered = 0
for seed in range(N):
    obs = run_random(env, seed)
    if obs.mission_status == "recovered":
        random_recovered += 1

print(f"=== Difficulty Report ({N} episodes each) ===")
print(f"Blind macro (no diagnosis):  {blind_recovered}/{N} recovered  (faults remaining in {blind_faults_remaining})")
print(f"Smart macro (with diagnosis): {smart_recovered}/{N} recovered")
print(f"Random policy:               {random_recovered}/{N} recovered")
print()
if blind_recovered > 30:
    print("⚠ BLIND MACRO TOO EASY — should be <30%")
elif blind_recovered > 10:
    print("⚠ Blind macro moderate — acceptable but could be tighter")
else:
    print("✓ Blind macro properly difficult")

if smart_recovered < 50:
    print("⚠ SMART MACRO TOO HARD — environment may be unlearnable")
else:
    print(f"✓ Smart macro recovers {smart_recovered}% — good signal for RL")

if random_recovered > 5:
    print(f"⚠ Random policy recovers {random_recovered}% — still too easy")
else:
    print("✓ Random policy properly fails")
