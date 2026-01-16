# %% [markdown]
# # Spontaneous Recovery Simulation - Heterogeneous Sparsity Sweep
#
# This script trains networks and runs sleep simulations to test spontaneous
# recovery across varying network sizes, pattern counts, and heterogeneity levels.
#
# Unlike SR_sparsity_sim which varies fixed sparsity, this varies the heterogeneity
# (sparsity_width) parameter while keeping mean_sparsity fixed at 0.5.
#
# After running this script, use visualization scripts to analyze the results.

# %% Imports
import numpy as np
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

# Network and pattern parameters
NB_REPETITION = 30
REPETITIONS = [i for i in range(1, NB_REPETITION + 1)]
NETWORK_SIZES = np.linspace(25, 250, 20, dtype=int)  # 20 values from 25 to 250
NUM_PATTERNS = np.arange(1, 26)  # 1 to 25 patterns
LEAK_VALUE = 1.0  # Fixed leak parameter
MEAN_SPARSITY = 0.5  # Fixed mean sparsity (P(0) = 0.5)
SPARSITY_WIDTHS = np.array([0.0, 0.2, 0.4, 0.6, 0.8])  # Heterogeneity levels to sweep
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
INIT_DRIVE = 0.5  # Initial state (unused in C++ now; kept for backward compatibility)
MAX_QUERIES = 200  # Maximum retrieval attempts
STOP_ON_SPURIOUS = 1  # Stop when spurious pattern encountered
STOP_ON_ALL_FOUND = 1  # Stop when all patterns found

# Experiment names
EXPERIMENT_NAME = "SR_heterogeneous_sparsity_sweep_small"
SLEEP_NAME = "SR_heterogeneous_sparsity_sleep_small"

# %% [markdown]
# ## Phase 1: Build C++ Executables

# %% Build
print("="*70)
print("BUILDING C++ EXECUTABLES")
print("="*70)
build()
print("Build complete!\n")

# %% [markdown]
# ## Phase 2: Training Phase (Write) with Heterogeneous Pattern Generation

# %% Setup write experiment
total_networks = len(NETWORK_SIZES) * len(NUM_PATTERNS) * len(SPARSITY_WIDTHS) * NB_REPETITION
print("="*70)
print("TRAINING PHASE (HETEROGENEOUS SPARSITY)")
print("="*70)
print(f"Number of repetitions: {NB_REPETITION}")
print(f"Network sizes: {len(NETWORK_SIZES)} values from {NETWORK_SIZES[0]} to {NETWORK_SIZES[-1]}")
print(f"Pattern counts: {len(NUM_PATTERNS)} values from {NUM_PATTERNS[0]} to {NUM_PATTERNS[-1]}")
print(f"Mean sparsity (P(0)): {MEAN_SPARSITY} (fixed)")
print(f"Heterogeneity widths: {len(SPARSITY_WIDTHS)} values {SPARSITY_WIDTHS.tolist()}")
print(f"  - 0.0 = uniform (all patterns same sparsity)")
print(f"  - 0.4 = highly heterogeneous (patterns vary widely)")
print(f"Total networks to train: {total_networks}")
print("="*70 + "\n")

write_config = setup_write_experiment(
    name=EXPERIMENT_NAME,
    patterns=None,  # C++ native generation
    params={
        "drive_target": DRIVE_TARGET,
        "learning_rate": LEARNING_RATE,
        "max_iter": MAX_ITER,
        "momentum_coef": MOMENTUM_COEF,
        "distance_noise_level": DISTANCE_NOISE_LEVEL,
        "leak": LEAK_VALUE,
        # Heterogeneous sparsity parameters
        "use_heterogeneous_sparsity": 1,  # Enable heterogeneous mode
        "mean_sparsity": MEAN_SPARSITY,
    },
    varying_params={
        "nb_repetition": REPETITIONS,
        "network_size": NETWORK_SIZES.tolist(),
        "num_patterns": NUM_PATTERNS.tolist(),
        "sparsity_width": SPARSITY_WIDTHS.tolist(),  # Vary heterogeneity
        "rho": [RHO],
    },
    native_pattern_generation=True  # C++ generates patterns
)

print(f"Configuration saved to: {write_config}\n")

# %% Run training
print("Starting training (this may take a while)...")
print("C++ will parallelize across up to 20 threads")
print("Each simulation generates unique heterogeneous pattern sets\n")
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
print(f"\nTrained networks: {DATA_DIR / 'trained_networks' / EXPERIMENT_NAME}")
print(f"Sleep results: {DATA_DIR / 'sleep_results' / SLEEP_NAME}")
print(f"\nEach network has heterogeneous patterns with varying sparsities.")
print(f"Pattern metadata (per-pattern sparsities) saved in each simulation directory.")
print(f"\nTo visualize results, use SR_viz.py or custom analysis scripts.")
print("="*70 + "\n")

# %%
