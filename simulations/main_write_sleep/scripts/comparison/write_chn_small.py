#%%
"""
CHN Training (Write Phase) - C++ Backend for Small Networks

Trains Continuous Hopfield Networks using gradient descent with momentum.
Uses C++ backend which is efficient for smaller networks (N < 300).

After running, use sleep_chn_small.py for the sleep phase.
"""

#%% Imports
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import (
    setup_write_experiment,
    run_cpp,
    build,
    DATA_DIR
)

#%% Configuration - Repetitions and Network Parameters
NB_REPETITION = 2
REPETITIONS = list(range(1, NB_REPETITION + 1))

NETWORK_SIZES = np.linspace(25, 250, 20, dtype=int).tolist()
NUM_PATTERNS = list(range(1, 26))
CORRELATIONS = [0.9, 0.75, 0.5, 0.25, 0.0]  # Avoid 1.0 (causes infinite loop)
SPARSITY = 0.5

#%% Configuration - Training Parameters
DRIVE_TARGET = 6.0
LEARNING_RATE = 0.00001
MAX_ITER = 1000000
MOMENTUM_COEF = 0.9
LEAK = 1.0

EXPERIMENT_NAME = "comparison_chn_cpp"

#%% Build C++ Executables
print("=" * 70)
print("CHN WRITE PHASE - C++ Backend (Small Networks)")
print("=" * 70)

print("\nBuilding C++ executables...")
build()
print("Build complete!")

#%% Print Configuration Summary
total_networks = len(NETWORK_SIZES) * len(NUM_PATTERNS) * len(CORRELATIONS) * NB_REPETITION

print(f"\nConfiguration:")
print(f"  Repetitions: {NB_REPETITION}")
print(f"  Network sizes: {NETWORK_SIZES[0]} to {NETWORK_SIZES[-1]}")
print(f"  Pattern counts: {NUM_PATTERNS[0]} to {NUM_PATTERNS[-1]}")
print(f"  Correlations: {CORRELATIONS}")
print(f"  Sparsity: {SPARSITY}")
print(f"  Total networks: {total_networks}")
print("=" * 70)

#%% Setup Experiment Configuration
write_config = setup_write_experiment(
    name=EXPERIMENT_NAME,
    params={
        "leak": LEAK,
        "drive_target": DRIVE_TARGET,
        "learning_rate": LEARNING_RATE,
        "max_iter": MAX_ITER,
        "momentum_coef": MOMENTUM_COEF,
        "sparsity": SPARSITY,
    },
    varying_params={
        "network_size": NETWORK_SIZES,
        "num_patterns": NUM_PATTERNS,
        "rho": CORRELATIONS,
        "seed": REPETITIONS,
    },
    native_pattern_generation=True
)

print(f"Configuration saved to: {write_config}")

#%% Run Training
print("\nStarting training (C++ will parallelize across threads)...")
run_cpp("write", write_config)

#%% Summary
print("\n" + "=" * 70)
print("WRITE PHASE COMPLETE")
print("=" * 70)
print(f"\nTrained networks saved to: {DATA_DIR / 'trained_networks' / EXPERIMENT_NAME}")
print(f"Next: Run sleep_chn_small.py")
print("=" * 70)
