# %% [markdown]
# # DHN Full Pipeline - Training and Query
#
# This notebook combines the write (training) and query phases for DHN comparison.
#
# **Phase 1 - Training:**
# - **Hebbian**: $W_{ij} = \frac{1}{N} \sum_\mu \xi_i^\mu \xi_j^\mu$
# - **Storkey**: mitigate crosstalk
#
# **Phase 2 - Query:**
# - Test partial cue retrieval with various informed fractions
# - Synchronous dynamics until convergence

# %% [markdown]
# ## Imports

# %%
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

# %% [markdown]
# ## Configuration - Write Phase
#
# Define the parameter space for the DHN training experiments.
# These parameters should match CHN experiments for fair comparison.

# %%
# Repetitions
NB_REPETITION = 10
REPETITIONS = list(range(NB_REPETITION))

# Network and pattern parameters
NETWORK_SIZES = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
NUM_PATTERNS = list(range(1, 100, 2))  # 1, 3, 5, ..., 99
CORRELATIONS = [0.8, 0.6, 0.4, 0.2, 0.0]  # Avoid 1.0 (causes infinite loop)
SPARSITY = 0.5

# Experiment names
HEBBIAN_NAME = "comparison_dhn_hebbian"
STORKEY_NAME = "comparison_dhn_storkey"

# %% [markdown]
# ## Configuration - Query Phase
#
# Define query parameters for testing partial cue retrieval.

# %%
# Informed fractions for partial cue queries
# Higher values = easier retrieval (more information provided)
INFORMED_FRACTIONS = [0.9, 0.75, 0.5, 0.25, 0.1]

# Query parameters
MAX_DYNAMICS_STEPS = 10  # Max synchronous steps before stopping
USE_SYNCHRONOUS = 1      # 1 = synchronous (faster, with convergence detection)

# Query experiment names
HEBBIAN_QUERY_NAME = "comparison_dhn_hebbian_query"
STORKEY_QUERY_NAME = "comparison_dhn_storkey_query"

# %%
# Calculate total simulations
sims_per_rule = len(NETWORK_SIZES) * len(NUM_PATTERNS) * len(CORRELATIONS) * NB_REPETITION

print("=" * 70)
print("DHN FULL PIPELINE - Configuration Summary")
print("=" * 70)
print(f"\nWRITE PHASE:")
print(f"  Repetitions:     {NB_REPETITION}")
print(f"  Network sizes:   {NETWORK_SIZES[0]} to {NETWORK_SIZES[-1]} ({len(NETWORK_SIZES)} values)")
print(f"  Pattern counts:  {NUM_PATTERNS[0]} to {NUM_PATTERNS[-1]} ({len(NUM_PATTERNS)} values)")
print(f"  Correlations:    {CORRELATIONS}")
print(f"  Sparsity:        {SPARSITY}")
print("-" * 70)
print(f"  Networks per rule: {sims_per_rule:,}")
print(f"  Total networks:    {sims_per_rule * 2:,}")
print(f"\nQUERY PHASE:")
print(f"  Informed fractions:  {INFORMED_FRACTIONS}")
print(f"  Update mode:         {'Synchronous' if USE_SYNCHRONOUS else 'Asynchronous'}")
print(f"  Max dynamics steps:  {MAX_DYNAMICS_STEPS}")
print(f"  Queries per network: {len(INFORMED_FRACTIONS)}")
print("=" * 70)

# %% [markdown]
# ## Build C++ Executables
#
# Compile the C++ simulation binaries before running experiments.

# %%
print("Building C++ executables...")
build_success = build()
if build_success:
    print("Build complete!")
else:
    print("Build failed! Check compilation errors.")

# %% [markdown]
# ---
# # PHASE 1: WRITE (Training)
# ---

# %% [markdown]
# ## Train Hebbian Networks
#
# Hebbian learning rule: $W_{ij} = \frac{1}{N} \sum_\mu \xi_i^\mu \xi_j^\mu$
#
# This is the classic Hopfield network learning rule with capacity ~0.138*N patterns.

# %%
print("\n" + "=" * 70)
print("TRAINING HEBBIAN NETWORKS")
print("=" * 70)

hebbian_config = setup_dhn_train_experiment(
    name=HEBBIAN_NAME,
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

print(f"Configuration saved to: {hebbian_config}")
print("\nStarting Hebbian training...")
run_cpp("dhn_train", hebbian_config)
print("Hebbian training complete!")

# %% [markdown]
# ## Train Storkey Networks
#
# Storkey learning rule subtracts crosstalk terms for improved capacity (~0.42*N patterns).
#
# Reference: Storkey, A.J. (1997) "Increasing the capacity of a Hopfield network
# without sacrificing functionality"

# %%
print("\n" + "=" * 70)
print("TRAINING STORKEY NETWORKS")
print("=" * 70)

storkey_config = setup_dhn_train_experiment(
    name=STORKEY_NAME,
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

print(f"Configuration saved to: {storkey_config}")
print("\nStarting Storkey training...")
run_cpp("dhn_train", storkey_config)
print("Storkey training complete!")

# %% [markdown]
# ## Write Phase Summary

# %%
print("\n" + "=" * 70)
print("WRITE PHASE COMPLETE")
print("=" * 70)
print(f"\nHebbian networks: {DATA_DIR / 'trained_networks' / HEBBIAN_NAME}")
print(f"Storkey networks: {DATA_DIR / 'trained_networks' / STORKEY_NAME}")

# %% [markdown]
# ---
# # PHASE 2: QUERY (Partial Cue Retrieval)
# ---

# %% [markdown]
# ## Check Prerequisites
#
# Verify that trained networks exist before running queries.

# %%
hebbian_dir = DATA_DIR / "trained_networks" / HEBBIAN_NAME
storkey_dir = DATA_DIR / "trained_networks" / STORKEY_NAME

# Check prerequisites
prereqs_ok = True

if not hebbian_dir.exists():
    print(f"ERROR: Hebbian networks not found at {hebbian_dir}")
    print("Please run the training cells first.")
    prereqs_ok = False
else:
    hebbian_sims = len([d for d in hebbian_dir.iterdir()
                        if d.is_dir() and d.name.startswith("sim_nb_")])
    print(f"Found {hebbian_sims} Hebbian networks")

if not storkey_dir.exists():
    print(f"ERROR: Storkey networks not found at {storkey_dir}")
    print("Please run the training cells first.")
    prereqs_ok = False
else:
    storkey_sims = len([d for d in storkey_dir.iterdir()
                        if d.is_dir() and d.name.startswith("sim_nb_")])
    print(f"Found {storkey_sims} Storkey networks")

if prereqs_ok:
    print("\nPrerequisites OK - ready for query phase")

# %% [markdown]
# ## Query Hebbian Networks
#
# Test partial cue retrieval on networks trained with Hebbian learning rule.

# %%
if prereqs_ok:
    print("\n" + "=" * 70)
    print("QUERYING HEBBIAN NETWORKS")
    print("=" * 70)

    hebbian_query_config = setup_dhn_query_experiment(
        name=HEBBIAN_QUERY_NAME,
        trained_networks_dir=hebbian_dir,
        params={
            "nb_dynamics_steps": MAX_DYNAMICS_STEPS,
            "use_synchronous": USE_SYNCHRONOUS,
        },
        varying_params={
            "informed_fraction": INFORMED_FRACTIONS,
        }
    )

    print(f"Configuration saved to: {hebbian_query_config}")
    print("\nStarting Hebbian queries...")
    run_cpp("dhn_query", hebbian_query_config)
    print("Hebbian queries complete!")
else:
    print("Skipping Hebbian queries - prerequisites not met")

# %% [markdown]
# ## Query Storkey Networks
#
# Test partial cue retrieval on networks trained with Storkey learning rule.

# %%
if prereqs_ok:
    print("\n" + "=" * 70)
    print("QUERYING STORKEY NETWORKS")
    print("=" * 70)

    storkey_query_config = setup_dhn_query_experiment(
        name=STORKEY_QUERY_NAME,
        trained_networks_dir=storkey_dir,
        params={
            "nb_dynamics_steps": MAX_DYNAMICS_STEPS,
            "use_synchronous": USE_SYNCHRONOUS,
        },
        varying_params={
            "informed_fraction": INFORMED_FRACTIONS,
        }
    )

    print(f"Configuration saved to: {storkey_query_config}")
    print("\nStarting Storkey queries...")
    run_cpp("dhn_query", storkey_query_config)
    print("Storkey queries complete!")
else:
    print("Skipping Storkey queries - prerequisites not met")

# %% [markdown]
# ## Final Summary

# %%
if prereqs_ok:
    print("\n" + "=" * 70)
    print("FULL PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\nTrained Networks:")
    print(f"  Hebbian: {DATA_DIR / 'trained_networks' / HEBBIAN_NAME}")
    print(f"  Storkey: {DATA_DIR / 'trained_networks' / STORKEY_NAME}")
    print(f"\nQuery Results:")
    print(f"  Hebbian: {DATA_DIR / 'query_results' / HEBBIAN_QUERY_NAME}")
    print(f"  Storkey: {DATA_DIR / 'query_results' / STORKEY_QUERY_NAME}")
    print(f"\nNext: Run viz_dhn_query.py to generate figures")
    print("=" * 70)

# %%
