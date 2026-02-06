# %% [markdown]
# # DHN Query Phase - Partial Cue Retrieval Tests
#
# Tests pattern retrieval from partial cues on trained DHN networks.
#
# For each stored pattern:
# 1. Keep `informed_fraction` of units with pattern values
# 2. Set remaining units to random {-1, +1}
# 3. Run **synchronous** dynamics until convergence (no unit changes)
# 4. Check if result matches the original pattern
#
# **Prerequisites:** Run `write_dhn.py` or `write_dhn_notebook.py` first.

# %% [markdown]
# ## Imports

# %%
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import (
    setup_dhn_query_experiment,
    run_cpp,
    build,
    DATA_DIR
)

# %% [markdown]
# ## Configuration
#
# Define query parameters for testing partial cue retrieval.

# %%
# Informed fractions for partial cue queries
# Higher values = easier retrieval (more information provided)
INFORMED_FRACTIONS = [0.9, 0.75, 0.5, 0.25, 0.1]

# Query parameters
MAX_DYNAMICS_STEPS = 10  # Max synchronous steps before stopping
USE_SYNCHRONOUS = 1      # 1 = synchronous (faster, with convergence detection)

# Experiment names (must match write_dhn names)
HEBBIAN_TRAIN_NAME = "comparison_dhn_hebbian"
STORKEY_TRAIN_NAME = "comparison_dhn_storkey"
HEBBIAN_QUERY_NAME = "comparison_dhn_hebbian_query"
STORKEY_QUERY_NAME = "comparison_dhn_storkey_query"

# %% [markdown]
# ## Build C++ Executables

# %%
print("Building C++ executables...")
build_success = build()
if build_success:
    print("Build complete!")
else:
    print("Build failed! Check compilation errors.")

# %% [markdown]
# ## Check Prerequisites
#
# Verify that trained networks exist before running queries.

# %%
hebbian_dir = DATA_DIR / "trained_networks" / HEBBIAN_TRAIN_NAME
storkey_dir = DATA_DIR / "trained_networks" / STORKEY_TRAIN_NAME

# Check prerequisites
prereqs_ok = True

if not hebbian_dir.exists():
    print(f"ERROR: Hebbian networks not found at {hebbian_dir}")
    print("Please run write_dhn.py first.")
    prereqs_ok = False
else:
    hebbian_sims = len([d for d in hebbian_dir.iterdir()
                        if d.is_dir() and d.name.startswith("sim_nb_")])
    print(f"Found {hebbian_sims} Hebbian networks")

if not storkey_dir.exists():
    print(f"ERROR: Storkey networks not found at {storkey_dir}")
    print("Please run write_dhn.py first.")
    prereqs_ok = False
else:
    storkey_sims = len([d for d in storkey_dir.iterdir()
                        if d.is_dir() and d.name.startswith("sim_nb_")])
    print(f"Found {storkey_sims} Storkey networks")

if prereqs_ok:
    print("\n" + "=" * 70)
    print("Configuration Summary")
    print("=" * 70)
    print(f"  Informed fractions:       {INFORMED_FRACTIONS}")
    print(f"  Update mode:              {'Synchronous' if USE_SYNCHRONOUS else 'Asynchronous'}")
    print(f"  Max dynamics steps:       {MAX_DYNAMICS_STEPS}")
    print(f"  Queries per network:      {len(INFORMED_FRACTIONS)}")
    print("=" * 70)

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
# ## Summary
#
# Query phase complete. Output locations:

# %%
if prereqs_ok:
    print("\n" + "=" * 70)
    print("QUERY PHASE COMPLETE")
    print("=" * 70)
    print(f"\nHebbian query results: {DATA_DIR / 'query_results' / HEBBIAN_QUERY_NAME}")
    print(f"Storkey query results: {DATA_DIR / 'query_results' / STORKEY_QUERY_NAME}")
    print(f"\nNext: Run viz_comparison.py to generate figures")
    print("=" * 70)
