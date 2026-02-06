#%%
"""
CHN Training (Write) + Sleep Phase - Larger Networks (N=300-1000)

Combined script: runs write phase first, then sleep phase.
Launch once and leave your computer.
"""

#%% Imports
import time
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import setup_write_experiment, setup_sleep_experiment, run_cpp, build, DATA_DIR

SCRIPT_DIR = Path(__file__).parent
LOG_FILE = SCRIPT_DIR / "timing_log.txt"

#%% Configuration
NB_REPETITION = 2  

EXPERIMENT_NAME = "SR_correlation_0.5"

# Training parameters
TRAIN_PARAMS = {
    "leak": 5.0,
    "drive_target": 6.0,
    "learning_rate": 0.0001,
    "epsilon_learning": 0.001,
    "max_iter": 1000000,
    "momentum_coef": 0.9,
    "sparsity": 0.5,
}

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

# Sweep configurations: (suffix, network_sizes, pattern_range)
SWEEP_CONFIGS = [
    ("compatible_large", [100 , 200], range(1, 25)),  # 15-30 patterns
]

CORRELATIONS = [0.5]

#%% ========== WRITE PHASE ==========
print("\n" + "=" * 70)
print("PHASE 1: WRITE (Training)")
print("=" * 70)

write_total_time = 0.0
write_timing_info = []

for suffix, sizes, patterns in SWEEP_CONFIGS:
    name = f"{EXPERIMENT_NAME}_{suffix}"
    n_networks = len(sizes) * len(patterns) * len(CORRELATIONS) * NB_REPETITION

    print(f"\n--- {suffix}: N={sizes}, K={patterns[0]}-{patterns[-1]}, {n_networks} networks ---")

    config = setup_write_experiment(
        name=name,
        params=TRAIN_PARAMS,
        varying_params={
            "network_size": sizes,
            "num_patterns": list(patterns),
            "rho": CORRELATIONS,
            "seed": list(range(NB_REPETITION)),
        },
        native_pattern_generation=True
    )

    t0 = time.time()
    run_cpp("write", config)
    elapsed = time.time() - t0
    write_total_time += elapsed

    write_timing_info.append((suffix, elapsed, n_networks))
    print(f"  Time: {elapsed/60:.1f} min ({elapsed/n_networks:.2f}s per network)")

print("\n" + "-" * 70)
print(f"WRITE PHASE COMPLETE - Total: {write_total_time/60:.1f} min")
print("-" * 70)

#%% ========== SLEEP PHASE ==========
print("\n" + "=" * 70)
print("PHASE 2: SLEEP (Autonomous Retrieval)")
print("=" * 70)

sleep_total_time = 0.0
sleep_timing_info = []

for suffix, sizes, patterns in SWEEP_CONFIGS:
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
    sleep_total_time += elapsed

    sleep_timing_info.append((suffix, elapsed, n_networks))
    print(f"  Time: {elapsed/60:.1f} min ({elapsed/n_networks:.2f}s per network)")

#%% Write timing log
with open(LOG_FILE, "a") as f:
    f.write(f"\n=== COMBINED WRITE+SLEEP ({time.strftime('%Y-%m-%d %H:%M')}) ===\n")
    f.write(f"Repetitions: {NB_REPETITION}\n")
    f.write("\n-- Write Phase --\n")
    for suffix, elapsed, n_nets in write_timing_info:
        per_sweep = elapsed / NB_REPETITION
        f.write(f"  {suffix}: {elapsed/60:.1f} min total, {per_sweep/60:.1f} min/sweep\n")
    f.write(f"Write total: {write_total_time/60:.1f} min\n")
    f.write("\n-- Sleep Phase --\n")
    for suffix, elapsed, n_nets in sleep_timing_info:
        per_sweep = elapsed / NB_REPETITION
        f.write(f"  {suffix}: {elapsed/60:.1f} min total, {per_sweep/60:.1f} min/sweep\n")
    f.write(f"Sleep total: {sleep_total_time/60:.1f} min\n")
    total_time = write_total_time + sleep_total_time
    f.write(f"\nGrand total: {total_time/60:.1f} min\n")
    f.write(f"Estimated for 10 sweeps: {total_time/NB_REPETITION*10/60:.1f} min\n")

#%% Summary
total_time = write_total_time + sleep_total_time
print("\n" + "=" * 70)
print("ALL PHASES COMPLETE")
print("=" * 70)
print(f"Write time:  {write_total_time/60:.1f} min")
print(f"Sleep time:  {sleep_total_time/60:.1f} min")
print(f"Total time:  {total_time/60:.1f} min")
print(f"Timing log:  {LOG_FILE}")
print("=" * 70)
