#%%
"""
CHN Sleep Phase (Autonomous Retrieval) - Larger Networks (N=300-1000)

Prerequisites: Run write_chn_larger.py first.
"""

#%% Imports
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import setup_sleep_experiment, run_cpp, build, DATA_DIR

SCRIPT_DIR = Path(__file__).parent
LOG_FILE = SCRIPT_DIR / "timing_log.txt"

#%% Configuration
NB_REPETITION = 2  # Must match write_chn_larger.py

EXPERIMENT_NAME = "capacity_scaling_larger"

# Sleep parameters
SLEEP_PARAMS = {
    "beta": 0.1,
    "delta": 0.01,
    "noise_dynamics": 1,
    "stddev_dynamics": 0.01,
    "max_queries": 200,
    "stop_on_spurious": 1,
    "stop_on_all_found": 1,
    "save_trajectories": 0,
}

# Sweep configurations (must match write_chn_larger.py)
SWEEP_SUFFIXES = ["small", "large"]

#%% Build
print("=" * 70)
print("CHN SLEEP PHASE - Larger Networks")
print("=" * 70)
build()

#%% Run sleep sweeps
total_time = 0.0
timing_info = []

for suffix in SWEEP_SUFFIXES:
    trained_dir = DATA_DIR / "trained_networks" / f"{EXPERIMENT_NAME}_{suffix}"

    if not trained_dir.exists():
        print(f"\nWARNING: {trained_dir} not found, skipping.")
        continue

    sim_dirs = [d for d in trained_dir.iterdir() if d.is_dir() and d.name.startswith("sim_nb_")]
    n_networks = len(sim_dirs)

    print(f"\n--- {suffix}: {n_networks} networks ---")

    config = setup_sleep_experiment(
        name=f"{EXPERIMENT_NAME}_{suffix}_sleep",
        trained_networks_dir=trained_dir,
        params=SLEEP_PARAMS
    )

    t0 = time.time()
    run_cpp("sleep", config)
    elapsed = time.time() - t0
    total_time += elapsed

    timing_info.append((suffix, elapsed, n_networks))
    print(f"  Time: {elapsed/60:.1f} min ({elapsed/n_networks:.2f}s per network)")

#%% Write timing log
with open(LOG_FILE, "a") as f:
    f.write(f"\n=== SLEEP PHASE ({time.strftime('%Y-%m-%d %H:%M')}) ===\n")
    f.write(f"Repetitions: {NB_REPETITION}\n")
    for suffix, elapsed, n_nets in timing_info:
        per_sweep = elapsed / NB_REPETITION
        f.write(f"  {suffix}: {elapsed/60:.1f} min total, {per_sweep/60:.1f} min/sweep\n")
    f.write(f"Total: {total_time/60:.1f} min, {total_time/NB_REPETITION/60:.1f} min/sweep\n")
    f.write(f"Estimated for 10 sweeps: {total_time/NB_REPETITION*10/60:.1f} min\n")

#%% Summary
print("\n" + "=" * 70)
print("SLEEP PHASE COMPLETE")
print(f"Total time: {total_time/60:.1f} min ({total_time/NB_REPETITION/60:.1f} min/sweep)")
print(f"Timing log: {LOG_FILE}")
print("=" * 70)
