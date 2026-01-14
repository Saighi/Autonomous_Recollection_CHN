#!/usr/bin/env python3
"""
DHN Query Phase - Partial Cue Retrieval Tests

Tests pattern retrieval from partial cues on trained DHN networks.
For each pattern:
1. Keep informed_fraction of units with pattern values
2. Set remaining units to random {-1, +1}
3. Run asynchronous dynamics
4. Check if result matches pattern

Usage:
    python query_dhn.py

Prerequisites: Run write_dhn.py first to generate trained networks.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import (
    setup_dhn_query_experiment,
    run_cpp,
    build,
    DATA_DIR
)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Informed fractions for partial cue queries
INFORMED_FRACTIONS = [0.9, 0.5, 0.2, 0.1]

# Query parameters
NB_DYNAMICS_STEPS = 10  # Async update sweeps per query

# Experiment names
HEBBIAN_TRAIN_NAME = "comparison_dhn_hebbian"
STORKEY_TRAIN_NAME = "comparison_dhn_storkey"
HEBBIAN_QUERY_NAME = "comparison_dhn_hebbian_query"
STORKEY_QUERY_NAME = "comparison_dhn_storkey_query"


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("DHN QUERY PHASE - Partial Cue Retrieval")
    print("=" * 70)

    # Build C++ executables
    print("\nBuilding C++ executables...")
    build()
    print("Build complete!")

    hebbian_dir = DATA_DIR / "trained_networks" / HEBBIAN_TRAIN_NAME
    storkey_dir = DATA_DIR / "trained_networks" / STORKEY_TRAIN_NAME

    # Check prerequisites
    if not hebbian_dir.exists():
        print(f"\nERROR: Hebbian networks not found at {hebbian_dir}")
        print("Please run write_dhn.py first.")
        sys.exit(1)

    if not storkey_dir.exists():
        print(f"\nERROR: Storkey networks not found at {storkey_dir}")
        print("Please run write_dhn.py first.")
        sys.exit(1)

    # Count networks
    hebbian_sims = len([d for d in hebbian_dir.iterdir()
                        if d.is_dir() and d.name.startswith("sim_nb_")])
    storkey_sims = len([d for d in storkey_dir.iterdir()
                        if d.is_dir() and d.name.startswith("sim_nb_")])

    print(f"\nConfiguration:")
    print(f"  Informed fractions: {INFORMED_FRACTIONS}")
    print(f"  Dynamics steps per query: {NB_DYNAMICS_STEPS}")
    print(f"  Hebbian networks: {hebbian_sims}")
    print(f"  Storkey networks: {storkey_sims}")
    print(f"  Queries per network: {len(INFORMED_FRACTIONS)}")
    print("=" * 70)

    # =========================================================================
    # QUERY HEBBIAN NETWORKS
    # =========================================================================
    print("\n" + "=" * 70)
    print("QUERYING HEBBIAN NETWORKS")
    print("=" * 70)

    hebbian_query_config = setup_dhn_query_experiment(
        name=HEBBIAN_QUERY_NAME,
        trained_networks_dir=hebbian_dir,
        params={
            "nb_dynamics_steps": NB_DYNAMICS_STEPS,
        },
        varying_params={
            "informed_fraction": INFORMED_FRACTIONS,
        }
    )

    print(f"Configuration saved to: {hebbian_query_config}")
    print("\nStarting Hebbian queries...")
    run_cpp("dhn_query", hebbian_query_config)
    print("Hebbian queries complete!")

    # =========================================================================
    # QUERY STORKEY NETWORKS
    # =========================================================================
    print("\n" + "=" * 70)
    print("QUERYING STORKEY NETWORKS")
    print("=" * 70)

    storkey_query_config = setup_dhn_query_experiment(
        name=STORKEY_QUERY_NAME,
        trained_networks_dir=storkey_dir,
        params={
            "nb_dynamics_steps": NB_DYNAMICS_STEPS,
        },
        varying_params={
            "informed_fraction": INFORMED_FRACTIONS,
        }
    )

    print(f"Configuration saved to: {storkey_query_config}")
    print("\nStarting Storkey queries...")
    run_cpp("dhn_query", storkey_query_config)
    print("Storkey queries complete!")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("QUERY PHASE COMPLETE")
    print("=" * 70)
    print(f"\nHebbian query results: {DATA_DIR / 'query_results' / HEBBIAN_QUERY_NAME}")
    print(f"Storkey query results: {DATA_DIR / 'query_results' / STORKEY_QUERY_NAME}")
    print(f"\nNext: Run viz_comparison.py to generate figures")
    print("=" * 70)
