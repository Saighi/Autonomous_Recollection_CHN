# %% [markdown]
# # CHN Training and Sleep Simulation for Comparison Study
#
# This script trains Continuous Hopfield Networks (CHN) using gradient descent
# and then runs sleep consolidation to test Autonomous Retrieval (AR) capacity.
#
# ## Methodology
#
# **Training (Write Phase):**
# - Gradient descent with momentum adjusts weights to create attractor basins
# - Target drive of 6.0 gives activation ~0.997 for active, ~0.003 for inactive
# - AVX2-optimized for performance
#
# **Sleep Phase:**
# - Networks start from neutral state (0.5)
# - Diagonal inhibitory plasticity enables sequential pattern retrieval
# - Success = all patterns retrieved before spurious attractor
#
# ## Key Finding
#
# In load regimes where AR succeeds >= 90%, networks can retrieve patterns
# from partial cues with only **10% informed units**. This makes AR success
# directly comparable to DHN partial cue experiments.
#
# After running this script, run run_dhn_sim.py for DHN comparison,
# then viz_comparison.py for visualization.

# %% Imports
import numpy as np
from pathlib import Path
import sys

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

# Repetitions (2 for testing, 20 for final results)
NB_REPETITION = 2
REPETITIONS = list(range(NB_REPETITION))

# Network and pattern parameters
NETWORK_SIZES = list(range(100, 1001, 100))  # [100, 200, ..., 1000]
NUM_PATTERNS = list(range(10, 101, 5))       # [10, 15, ..., 100]
CORRELATIONS = [0.1, 0.25, 0.5, 0.75, 1.0]   # Pattern correlations
SPARSITY = 0.5                                # 50% active units

# Training parameters
DRIVE_TARGET = 6.0
LEARNING_RATE = 0.001
MAX_ITER = 100000
MOMENTUM_COEF = 0.9
LEAK = 1.0

# Sleep parameters
BETA = 0.1           # Inhibitory plasticity rate
DELTA = 0.01         # Integration timestep
NOISE_DYNAMICS = 1   # Enable stochastic noise
STDDEV_DYNAMICS = 0.01
MAX_QUERIES = 200    # Maximum retrieval attempts
STOP_ON_SPURIOUS = 0 # Don't stop early (need full data)
STOP_ON_ALL_FOUND = 0

# Experiment names
WRITE_NAME = "comparison_chn"
SLEEP_NAME = "comparison_chn_sleep"

# %% [markdown]
# ## Phase 1: Build C++ Executables

# %% Build
print("=" * 70)
print("BUILDING C++ EXECUTABLES")
print("=" * 70)
build()
print("Build complete!\n")

# %% [markdown]
# ## Phase 2: Training Phase

# %% Setup write experiment
total_networks = len(NETWORK_SIZES) * len(NUM_PATTERNS) * len(CORRELATIONS) * NB_REPETITION

print("=" * 70)
print("CHN TRAINING PHASE")
print("=" * 70)
print(f"Repetitions: {NB_REPETITION}")
print(f"Network sizes: {NETWORK_SIZES[0]} to {NETWORK_SIZES[-1]} ({len(NETWORK_SIZES)} values)")
print(f"Pattern counts: {NUM_PATTERNS[0]} to {NUM_PATTERNS[-1]} ({len(NUM_PATTERNS)} values)")
print(f"Correlations: {CORRELATIONS}")
print(f"Sparsity: {SPARSITY}")
print(f"Leak: {LEAK}")
print(f"Total networks to train: {total_networks}")
print("=" * 70 + "\n")

write_config = setup_write_experiment(
    name=WRITE_NAME,
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

print(f"Configuration saved to: {write_config}\n")

# %% Run training
print("Starting CHN training (this may take a while)...")
print("C++ will parallelize across up to 20 threads\n")
run_cpp("write", write_config)
print("\nTraining complete!")

# %% [markdown]
# ## Phase 3: Sleep Phase (Autonomous Retrieval)

# %% Setup sleep experiment
print("\n" + "=" * 70)
print("CHN SLEEP PHASE (AUTONOMOUS RETRIEVAL)")
print("=" * 70)
print(f"Running sleep on {total_networks} trained networks")
print(f"Beta (inhibitory plasticity): {BETA}")
print(f"Delta (timestep): {DELTA}")
print(f"Max queries: {MAX_QUERIES}")
print("=" * 70 + "\n")

sleep_config = setup_sleep_experiment(
    name=SLEEP_NAME,
    trained_networks_dir=DATA_DIR / "trained_networks" / WRITE_NAME,
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
print("\n" + "=" * 70)
print("CHN SIMULATION COMPLETE!")
print("=" * 70)
print(f"\nTrained networks: {DATA_DIR / 'trained_networks' / WRITE_NAME}")
print(f"Sleep results: {DATA_DIR / 'sleep_results' / SLEEP_NAME}")
print(f"\nNext: Run run_dhn_sim.py for DHN comparison")
print("=" * 70 + "\n")

# %%
