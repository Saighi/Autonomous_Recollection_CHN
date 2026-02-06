# %% [markdown]
# # DHN Training (Write Phase) - Hebbian and Storkey Learning Rules
#
# This notebook trains Discrete Hopfield Networks using both learning rules for comparison:
# - **Hebbian**: $W_{ij} = \frac{1}{N} \sum_\mu \xi_i^\mu \xi_j^\mu$
# - **Storkey**: mitigate crosstalk
#
# After running, use `query_dhn.py` for partial cue retrieval tests.

# %% [markdown]
# ## Imports

# %%
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import (
    setup_dhn_train_experiment,
    run_cpp,
    build,
    DATA_DIR
)

# %% [markdown]
# ## Configuration
#
# Define the parameter space for the DHN training experiments.
# These parameters should match CHN experiments for fair comparison.

# %%
# Repetitions
NB_REPETITION = 1
REPETITIONS = list(range(NB_REPETITION))

# Network and pattern parameters
NETWORK_SIZES = [ 100 , 200 , 300 , 400, 500 , 600, 700, 800, 900 , 1000
]
NUM_PATTERNS = list(range(1, 100, 2))  # 1 to 99 patterns
CORRELATIONS = [0.8, 0.6, 0.4, 0.2, 0.0]  # Avoid 1.0 (causes infinite loop)
SPARSITY = 0.5

# Experiment names
HEBBIAN_NAME = "comparison_dhn_hebbian"
STORKEY_NAME = "comparison_dhn_storkey"

# %%
# Calculate total simulations
sims_per_rule = len(NETWORK_SIZES) * len(NUM_PATTERNS) * len(CORRELATIONS) * NB_REPETITION

print("=" * 70)
print("DHN WRITE PHASE - Configuration Summary")
print("=" * 70)
print(f"  Repetitions:     {NB_REPETITION}")
print(f"  Network sizes:   {NETWORK_SIZES[0]} to {NETWORK_SIZES[-1]} ({len(NETWORK_SIZES)} values)")
print(f"  Pattern counts:  {NUM_PATTERNS[0]} to {NUM_PATTERNS[-1]} ({len(NUM_PATTERNS)} values)")
print(f"  Correlations:    {CORRELATIONS}")
print(f"  Sparsity:        {SPARSITY}")
print("-" * 70)
print(f"  Networks per rule: {sims_per_rule:,}")
print(f"  Total networks:    {sims_per_rule * 2:,}")
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
# ## Summary
#
# Training complete. Output locations:

# %%
print("\n" + "=" * 70)
print("WRITE PHASE COMPLETE")
print("=" * 70)
print(f"\nHebbian networks: {DATA_DIR / 'trained_networks' / HEBBIAN_NAME}")
print(f"Storkey networks: {DATA_DIR / 'trained_networks' / STORKEY_NAME}")
print(f"\nNext: Run query_dhn.py or query_dhn_notebook.py")
print("=" * 70)
