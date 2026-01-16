#%%
"""
CHN Training (Write Phase) - Larger Networks (N=300-1000)
"""

#%% Imports
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import setup_write_experiment, run_cpp, build, DATA_DIR

SCRIPT_DIR = Path(__file__).parent
LOG_FILE = SCRIPT_DIR / "timing_log.txt"

#%% Configuration
NB_REPETITION = 2  # TODO: increase to 10 for final runs

EXPERIMENT_NAME = "capacity_scaling_larger"

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

# Sweep configurations: (suffix, network_sizes, pattern_range)
SWEEP_CONFIGS = [
    ("small", [300, 350, 400, 450, 500], range(15, 31)),  # 15-30 patterns
    ("large", [1000], range(25, 40)),                      # 25-39 patterns
]

CORRELATIONS = [0.9, 0.75, 0.25, 0.0]

#%% Build
print("=" * 70)
print("CHN WRITE PHASE - Larger Networks")
print("=" * 70)
build()

#%% Run training sweeps
total_time = 0.0
timing_info = []

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
    total_time += elapsed

    timing_info.append((suffix, elapsed, n_networks))
    print(f"  Time: {elapsed/60:.1f} min ({elapsed/n_networks:.2f}s per network)")

#%% Write timing log
with open(LOG_FILE, "a") as f:
    f.write(f"\n=== WRITE PHASE ({time.strftime('%Y-%m-%d %H:%M')}) ===\n")
    f.write(f"Repetitions: {NB_REPETITION}\n")
    for suffix, elapsed, n_nets in timing_info:
        per_sweep = elapsed / NB_REPETITION
        f.write(f"  {suffix}: {elapsed/60:.1f} min total, {per_sweep/60:.1f} min/sweep\n")
    f.write(f"Total: {total_time/60:.1f} min, {total_time/NB_REPETITION/60:.1f} min/sweep\n")
    f.write(f"Estimated for 10 sweeps: {total_time/NB_REPETITION*10/60:.1f} min\n")

#%% Summary
print("\n" + "=" * 70)
print("WRITE PHASE COMPLETE")
print(f"Total time: {total_time/60:.1f} min ({total_time/NB_REPETITION/60:.1f} min/sweep)")
print(f"Timing log: {LOG_FILE}")
print("=" * 70)
