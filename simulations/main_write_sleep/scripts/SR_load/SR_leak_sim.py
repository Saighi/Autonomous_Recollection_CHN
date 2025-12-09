# %% [markdown]
# # Spontaneous Recovery Simulation - Leak Parameter Sweep
#
# This script trains networks and runs sleep simulations to test spontaneous
# recovery across varying network sizes, pattern counts, and leak parameters.
#
# After running this script, use SR_leak_viz.py to visualize the results.

# %% Imports
import numpy as np
from pathlib import Path
import sys

# Add scripts directory to path (parent.parent = scripts/)
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import (
    setup_write_experiment,
    setup_sleep_experiment,
    run_cpp,
    build,
    DATA_DIR
)

# %%=========================================================================
# CONFIGURATION SECTION
# ==========================================================================

# Network and pattern parameters
NB_REPETITION = 20 
REPETITIONS = [i for i in range(1,NB_REPETITION+1)]
NETWORK_SIZES = np.linspace(25, 250, 20, dtype=int)  # 20 values from 25 to 250
NUM_PATTERNS = np.arange(1, 26)  # 1 to 25 patterns
LEAK_VALUES = [0.25, 0.5, 1.0, 1.5, 2.0]  # Leak parameter sweep
SPARSITY = 0.5  # 50% active units
RHO = 0.5  # Pattern correlation

# Training parameters
DRIVE_TARGET = 6.0
LEARNING_RATE = 0.0001
MAX_ITER = 100000
MOMENTUM_COEF = 0.9
DISTANCE_NOISE_LEVEL = 0.0

# Sleep parameters
BETA = 0.1  # Inhibitory plasticity rate
DELTA = 0.01  # Integration timestep
NOISE_DYNAMICS = 1  # Enable stochastic noise
STDDEV_DYNAMICS = 0.01  # Noise standard deviation
INIT_DRIVE = 0.5  # Initial state
MAX_QUERIES = 200  # Maximum retrieval attempts
STOP_ON_SPURIOUS = 1  # Stop when spurious pattern encountered
STOP_ON_ALL_FOUND = 1  # Stop when all patterns found

# Experiment names
EXPERIMENT_NAME = "SR_leak_sweep"
SLEEP_NAME = "SR_leak_sleep"

# %% [markdown]
# ## Phase 1: Build C++ Executables

# %% Build
print("="*70)
print("BUILDING C++ EXECUTABLES")
print("="*70)
build()
print("Build complete!\n")

# %% [markdown]
# ## Phase 2: Training Phase (Write)

# %% Setup write experiment
total_networks = len(NETWORK_SIZES) * len(NUM_PATTERNS) * len(LEAK_VALUES) * NB_REPETITION
print("="*70)
print("TRAINING PHASE")
print("="*70)
print(f"Number of repetitions: {NB_REPETITION}")
print(f"Network sizes: {len(NETWORK_SIZES)} values from {NETWORK_SIZES[0]} to {NETWORK_SIZES[-1]}")
print(f"Pattern counts: {len(NUM_PATTERNS)} values from {NUM_PATTERNS[0]} to {NUM_PATTERNS[-1]}")
print(f"Leak values: {LEAK_VALUES}")
print(f"Total networks to train: {total_networks}")
print("="*70 + "\n")

write_config = setup_write_experiment(
    name=EXPERIMENT_NAME,
    params={
        "drive_target": DRIVE_TARGET,
        "learning_rate": LEARNING_RATE,
        "max_iter": MAX_ITER,
        "momentum_coef": MOMENTUM_COEF,
        "distance_noise_level": DISTANCE_NOISE_LEVEL,
        "use_old_patterns": 1.0,  # explicit old generator (balanced flips), requires sparsity=0.5
    },
    varying_params={
        "nb_repetition" : REPETITIONS,
        "network_size": NETWORK_SIZES.tolist(),
        "num_patterns": NUM_PATTERNS.tolist(),
        "sparsity": [SPARSITY],
        "rho": [RHO],
        "leak": LEAK_VALUES,
    },
    native_pattern_generation=True
)

print(f"Configuration saved to: {write_config}\n")

# %% Run training
print("Starting training (this may take a while)...")
print("C++ will parallelize across up to 20 threads\n")
run_cpp("write", write_config)
print("\nTraining complete!")

# %% [markdown]
# ## Phase 3: Sleep Phase (Spontaneous Recovery)

# %% Setup sleep experiment
print("\n" + "="*70)
print("SLEEP PHASE")
print("="*70)
print(f"Running sleep simulations on {total_networks} trained networks")
print(f"Beta (inhibitory plasticity): {BETA}")
print(f"Delta (timestep): {DELTA}")
print(f"Max queries: {MAX_QUERIES}")
print(f"Stop on spurious: {STOP_ON_SPURIOUS}")
print(f"Stop on all found: {STOP_ON_ALL_FOUND}")
print("="*70 + "\n")

sleep_config = setup_sleep_experiment(
    name=SLEEP_NAME,
    trained_networks_dir=DATA_DIR / "trained_networks" / EXPERIMENT_NAME,
    params={
        "beta": BETA,
        "delta": DELTA,
        "noise_dynamics": NOISE_DYNAMICS,
        "stddev_dynamics": STDDEV_DYNAMICS,
        "max_queries": MAX_QUERIES,
        "stop_on_spurious": STOP_ON_SPURIOUS,
        "stop_on_all_found": STOP_ON_ALL_FOUND,
        "save_trajectories": 0,
    }
)

print(f"Configuration saved to: {sleep_config}\n")

# %% Run sleep simulations
print("Starting sleep simulations...")
run_cpp("sleep", sleep_config)
print("\nSleep simulations complete!")

# %% Summary
print("\n" + "="*70)
print("SIMULATION COMPLETE!")
print("="*70)
print(f"\nResults saved to: {DATA_DIR / 'sleep_results' / SLEEP_NAME}")
print(f"\nTo visualize results, run: python SR_viz.py")
print("="*70 + "\n")

# %%
