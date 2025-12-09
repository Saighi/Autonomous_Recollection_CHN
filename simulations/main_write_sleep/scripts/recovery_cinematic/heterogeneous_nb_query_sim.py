# %% [markdown]
# # Heterogeneous Sparsity Query Count - Simulation Script
#
# This script generates data for analyzing how pattern sparsity affects
# the number of queries needed to recover each pattern during sleep.
#
# Key features:
# - Uses C++ native heterogeneous pattern generation
# - Sweeps across network sizes (200, 250, 300) and pattern counts (5, 8, 11)
# - 200 repetitions per configuration for statistical robustness
#
# Run visualization with: scripts/viz/heterogeneous_nb_query_viz.py

# %% Imports
from pathlib import Path
import sys

# Add scripts directory to path
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

# Network and pattern parameters - systematic sweep
NETWORK_SIZES = [200, 250, 300]
NUM_PATTERNS_LIST = [5, 8, 11]
NB_REPETITION = 200
REPETITIONS = list(range(1, NB_REPETITION + 1))

# Pattern generation parameters
MEAN_SPARSITY = 0.5     # Center of sparsity distribution (P(0) convention)
SPARSITY_WIDTH = 0.4    # Full width: sparsities in [0.3, 0.7]
RHO = 0.3               # Pattern correlation

# Training parameters
LEAK = 1.0
DRIVE_TARGET = 6.0
LEARNING_RATE = 0.0001
MAX_ITER = 100000
MOMENTUM_COEF = 0.9

# Sleep parameters
BETA = 0.1               # Inhibitory plasticity rate
DELTA = 0.01             # Integration timestep
MAX_QUERIES = 200        # Number of retrieval attempts
NOISE_DYNAMICS = 1       # Enable stochastic noise
STDDEV_DYNAMICS = 0.01   # Noise standard deviation
STOP_ON_SPURIOUS = 1     # Stop when spurious pattern encountered
STOP_ON_ALL_FOUND = 1    # Stop when all patterns recovered

# Experiment names
EXPERIMENT_NAME = "heterogeneous_nb_query"
SLEEP_NAME = "heterogeneous_nb_query_sleep"

# %% [markdown]
# ## Phase 1: Build C++ Executables

# %% Build
print("="*70)
print("BUILDING C++ EXECUTABLES")
print("="*70)
build()
print("Build complete!\n")

# %% [markdown]
# ## Phase 2: Training Phase (Write) with C++ Native Pattern Generation

# %% Setup and run training
total_networks = len(NETWORK_SIZES) * len(NUM_PATTERNS_LIST) * NB_REPETITION
print("="*70)
print("TRAINING PHASE (C++ NATIVE HETEROGENEOUS GENERATION)")
print("="*70)
print(f"Network sizes: {NETWORK_SIZES}")
print(f"Pattern counts: {NUM_PATTERNS_LIST}")
print(f"Repetitions per configuration: {NB_REPETITION}")
print(f"Total networks to train: {total_networks}")
print(f"\nPattern generation params:")
print(f"  Mean sparsity (P(0)): {MEAN_SPARSITY}")
print(f"  Sparsity width: {SPARSITY_WIDTH}")
print(f"  Expected sparsity range: [{MEAN_SPARSITY - SPARSITY_WIDTH/2:.2f}, {MEAN_SPARSITY + SPARSITY_WIDTH/2:.2f}]")
print(f"  Pattern correlation (rho): {RHO}")
print("="*70 + "\n")

# Use native pattern generation (C++ generates patterns with metadata)
write_config = setup_write_experiment(
    name=EXPERIMENT_NAME,
    patterns=None,  # No patterns from Python - C++ will generate them
    pattern_metadata=None,  # No metadata from Python - C++ will generate it
    params={
        # Training parameters
        "leak": LEAK,
        "drive_target": DRIVE_TARGET,
        "learning_rate": LEARNING_RATE,
        "max_iter": MAX_ITER,
        "momentum_coef": MOMENTUM_COEF,
        # Native pattern generation parameters
        "use_heterogeneous_sparsity": 1,  # Enable heterogeneous mode
        "mean_sparsity": MEAN_SPARSITY,
        "sparsity_width": SPARSITY_WIDTH,
    },
    varying_params={
        "network_size": NETWORK_SIZES,
        "num_patterns": NUM_PATTERNS_LIST,
        "rho": [RHO],
        "nb_repetition": REPETITIONS,
    },
    native_pattern_generation=True,  # Enable C++ native generation
)

print(f"Configuration saved to: {write_config}\n")
print("Starting training with C++ native pattern generation...")
print("All simulations will run in parallel...")
run_cpp("write", write_config)
print("\nTraining complete!")

# %% [markdown]
# ## Phase 3: Sleep Phase

# %% Setup and run sleep
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
print("Starting sleep simulations...")
run_cpp("sleep", sleep_config)
print("\nSleep simulations complete!")

# %% Summary
print("\n" + "="*70)
print("SIMULATION COMPLETE!")
print("="*70)
print(f"\nTrained networks: {DATA_DIR / 'trained_networks' / EXPERIMENT_NAME}")
print(f"Sleep results: {DATA_DIR / 'sleep_results' / SLEEP_NAME}")
print(f"\nTo visualize results, run:")
print(f"  python scripts/viz/heterogeneous_nb_query_viz.py")
print("="*70 + "\n")

# %%
