# %% [markdown]
# # DHN Training and Query Simulation for Comparison Study
#
# This script trains Discrete Hopfield Networks using Hebbian and Storkey
# learning rules, then tests partial cue retrieval to compare with CHN AR.
#
# ## Learning Rules
#
# **Hebbian (learning_rule=0):**
# ```
# W_ij = (1/N) * sum_mu(xi_i^mu * xi_j^mu)
# ```
# - Simple outer-product rule
# - Theoretical capacity: ~0.138*N patterns
#
# **Storkey (learning_rule=1):**
# ```
# W_ij^new = W_ij^old + (1/N) * [xi_i*xi_j - xi_i*h_j - h_i*xi_j]
# ```
# Where h_i = sum_{k!=i} W_ik * xi_k (local field before storing)
# - Subtracts crosstalk to reduce interference
# - Theoretical capacity: ~0.42*N patterns (3x better than Hebbian)
#
# ## Partial Cue Query
#
# For each pattern:
# 1. Keep informed_fraction of units with pattern values
# 2. Set remaining units to random {-1, +1}
# 3. Run asynchronous dynamics (10 sweeps)
# 4. Check if result matches pattern (or inverse)
#
# Run run_chn_sim.py first, then this script, then viz_comparison.py.

# %% Imports
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import (
    setup_dhn_train_experiment,
    setup_dhn_query_experiment,
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

# Network and pattern parameters (MUST MATCH run_chn_sim.py!)
NETWORK_SIZES = list(range(100, 1001, 100))  # [100, 200, ..., 1000]
NUM_PATTERNS = list(range(10, 101, 5))       # [10, 15, ..., 100]
CORRELATIONS = [0.1, 0.25, 0.5, 0.75, 1.0]   # Pattern correlations
SPARSITY = 0.5                                # 50% active units

# Informed fractions for partial cue queries
INFORMED_FRACTIONS = [0.9, 0.5, 0.2, 0.1]

# Query parameters
NB_DYNAMICS_STEPS = 10  # Async update sweeps per query

# Experiment names
HEBBIAN_TRAIN_NAME = "comparison_dhn_hebbian"
STORKEY_TRAIN_NAME = "comparison_dhn_storkey"
HEBBIAN_QUERY_NAME = "comparison_dhn_hebbian_query"
STORKEY_QUERY_NAME = "comparison_dhn_storkey_query"

# %% [markdown]
# ## Phase 1: Build C++ Executables

# %% Build
print("=" * 70)
print("BUILDING C++ EXECUTABLES")
print("=" * 70)
build()
print("Build complete!\n")

# %% [markdown]
# ## Phase 2: Train Hebbian Networks

# %% Train Hebbian
sims_per_rule = len(NETWORK_SIZES) * len(NUM_PATTERNS) * len(CORRELATIONS) * NB_REPETITION

print("=" * 70)
print("DHN HEBBIAN TRAINING")
print("=" * 70)
print(f"Repetitions: {NB_REPETITION}")
print(f"Network sizes: {NETWORK_SIZES[0]} to {NETWORK_SIZES[-1]} ({len(NETWORK_SIZES)} values)")
print(f"Pattern counts: {NUM_PATTERNS[0]} to {NUM_PATTERNS[-1]} ({len(NUM_PATTERNS)} values)")
print(f"Correlations: {CORRELATIONS}")
print(f"Sparsity: {SPARSITY}")
print(f"Learning rule: Hebbian (outer product)")
print(f"Total networks: {sims_per_rule}")
print("=" * 70 + "\n")

hebbian_config = setup_dhn_train_experiment(
    name=HEBBIAN_TRAIN_NAME,
    params={
        "sparsity": SPARSITY,
        "learning_rule": 0,  # 0 = Hebbian
    },
    varying_params={
        "network_size": NETWORK_SIZES,
        "num_patterns": NUM_PATTERNS,
        "rho": CORRELATIONS,
        "seed": REPETITIONS,
    }
)

print(f"Configuration saved to: {hebbian_config}\n")
print("Starting Hebbian training...")
run_cpp("dhn_train", hebbian_config)
print("\nHebbian training complete!")

# %% [markdown]
# ## Phase 3: Train Storkey Networks

# %% Train Storkey
print("\n" + "=" * 70)
print("DHN STORKEY TRAINING")
print("=" * 70)
print(f"Learning rule: Storkey (local field correction)")
print(f"Total networks: {sims_per_rule}")
print("=" * 70 + "\n")

storkey_config = setup_dhn_train_experiment(
    name=STORKEY_TRAIN_NAME,
    params={
        "sparsity": SPARSITY,
        "learning_rule": 1,  # 1 = Storkey
    },
    varying_params={
        "network_size": NETWORK_SIZES,
        "num_patterns": NUM_PATTERNS,
        "rho": CORRELATIONS,
        "seed": REPETITIONS,
    }
)

print(f"Configuration saved to: {storkey_config}\n")
print("Starting Storkey training...")
run_cpp("dhn_train", storkey_config)
print("\nStorkey training complete!")

# %% [markdown]
# ## Phase 4: Query Hebbian Networks

# %% Query Hebbian
query_sims = sims_per_rule * len(INFORMED_FRACTIONS)

print("\n" + "=" * 70)
print("DHN HEBBIAN QUERY (PARTIAL CUES)")
print("=" * 70)
print(f"Informed fractions: {INFORMED_FRACTIONS}")
print(f"Dynamics steps per query: {NB_DYNAMICS_STEPS}")
print(f"Total query simulations: {query_sims}")
print("=" * 70 + "\n")

hebbian_query_config = setup_dhn_query_experiment(
    name=HEBBIAN_QUERY_NAME,
    trained_networks_dir=DATA_DIR / "trained_networks" / HEBBIAN_TRAIN_NAME,
    params={
        "nb_dynamics_steps": NB_DYNAMICS_STEPS,
    },
    varying_params={
        "informed_fraction": INFORMED_FRACTIONS,
    }
)

print(f"Configuration saved to: {hebbian_query_config}\n")
print("Starting Hebbian queries...")
run_cpp("dhn_query", hebbian_query_config)
print("\nHebbian queries complete!")

# %% [markdown]
# ## Phase 5: Query Storkey Networks

# %% Query Storkey
print("\n" + "=" * 70)
print("DHN STORKEY QUERY (PARTIAL CUES)")
print("=" * 70)
print(f"Total query simulations: {query_sims}")
print("=" * 70 + "\n")

storkey_query_config = setup_dhn_query_experiment(
    name=STORKEY_QUERY_NAME,
    trained_networks_dir=DATA_DIR / "trained_networks" / STORKEY_TRAIN_NAME,
    params={
        "nb_dynamics_steps": NB_DYNAMICS_STEPS,
    },
    varying_params={
        "informed_fraction": INFORMED_FRACTIONS,
    }
)

print(f"Configuration saved to: {storkey_query_config}\n")
print("Starting Storkey queries...")
run_cpp("dhn_query", storkey_query_config)
print("\nStorkey queries complete!")

# %% Summary
print("\n" + "=" * 70)
print("DHN SIMULATION COMPLETE!")
print("=" * 70)
print(f"\nHebbian networks: {DATA_DIR / 'trained_networks' / HEBBIAN_TRAIN_NAME}")
print(f"Storkey networks: {DATA_DIR / 'trained_networks' / STORKEY_TRAIN_NAME}")
print(f"Hebbian queries: {DATA_DIR / 'query_results' / HEBBIAN_QUERY_NAME}")
print(f"Storkey queries: {DATA_DIR / 'query_results' / STORKEY_QUERY_NAME}")
print(f"\nNext: Run viz_comparison.py to generate figures")
print("=" * 70 + "\n")

# %%
