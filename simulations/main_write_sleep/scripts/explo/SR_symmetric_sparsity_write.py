# %% [markdown]
# # Spontaneous Recovery - Symmetric Transfer with Sparsity Sweep (Write Phase)
#
# This script trains networks using:
# - Symmetric transfer function (sigmoid(x) - 0.5)
# - NEW pattern generator (parent + redraw)
# - Multiple sparsity levels: 0.1, 0.3, 0.5, 0.7, 0.9
# - 1 repetition per (network size, num patterns) combination
#
# After running this script, use SR_symmetric_sparsity_sleep.py to run sleep simulations.

# %% Imports
import numpy as np
from pathlib import Path
import sys

# Add scripts directory to path (parent.parent = scripts/)
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import (
    setup_write_experiment,
    run_cpp,
    build,
    DATA_DIR
)

# %%=========================================================================
# CONFIGURATION SECTION
# ==========================================================================

# Network and pattern parameters
NB_REPETITION = 1
REPETITIONS = [1]
NETWORK_SIZES = np.linspace(25, 250, 20, dtype=int)  # 20 values from 25 to 250
NUM_PATTERNS = np.arange(1, 26)  # 1 to 25 patterns
SPARSITY_VALUES = [0.1, 0.3, 0.5, 0.7, 0.9]  # Sparsity parameter sweep
RHO = 0.5  # Pattern correlation

# Training parameters
LEAK = 1.0  # Fixed leak value
DRIVE_TARGET = 6.0
LEARNING_RATE = 0.0001
MAX_ITER = 100000
MOMENTUM_COEF = 0.9
DISTANCE_NOISE_LEVEL = 0.0
SYMMETRIC_TRANSFER = 1.0  # Use symmetric transfer function

# Experiment name
EXPERIMENT_NAME = "SR_symmetric_sparsity"

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
total_networks = len(NETWORK_SIZES) * len(NUM_PATTERNS) * len(SPARSITY_VALUES) * NB_REPETITION
print("="*70)
print("TRAINING PHASE - SYMMETRIC TRANSFER + NEW PATTERN GENERATION")
print("="*70)
print(f"Number of repetitions: {NB_REPETITION}")
print(f"Network sizes: {len(NETWORK_SIZES)} values from {NETWORK_SIZES[0]} to {NETWORK_SIZES[-1]}")
print(f"Pattern counts: {len(NUM_PATTERNS)} values from {NUM_PATTERNS[0]} to {NUM_PATTERNS[-1]}")
print(f"Sparsity values: {SPARSITY_VALUES}")
print(f"Transfer function: Symmetric (sigmoid(x) - 0.5)")
print(f"Pattern generator: NEW (parent + redraw)")
print(f"Total networks to train: {total_networks}")
print("="*70 + "\n")

write_config = setup_write_experiment(
    name=EXPERIMENT_NAME,
    params={
        "leak": LEAK,
        "drive_target": DRIVE_TARGET,
        "learning_rate": LEARNING_RATE,
        "max_iter": MAX_ITER,
        "momentum_coef": MOMENTUM_COEF,
        "distance_noise_level": DISTANCE_NOISE_LEVEL,
        "symmetric_transfer": SYMMETRIC_TRANSFER,
        "use_old_patterns": 0.0,  # Use NEW pattern generator
    },
    varying_params={
        "nb_repetition": REPETITIONS,
        "network_size": NETWORK_SIZES.tolist(),
        "num_patterns": NUM_PATTERNS.tolist(),
        "sparsity": SPARSITY_VALUES,
        "rho": [RHO],
    },
    native_pattern_generation=True
)

print(f"Configuration saved to: {write_config}\n")

# %% Run training
print("Starting training (this may take a while)...")
print("C++ will parallelize across up to 20 threads\n")
run_cpp("write", write_config)
print("\nTraining complete!")

# %% Summary
print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)
print(f"\nTrained networks saved to: {DATA_DIR / 'trained_networks' / EXPERIMENT_NAME}")
print(f"\nNext step: Run SR_symmetric_sparsity_sleep.py to perform sleep simulations")
print("="*70 + "\n")

# %%
