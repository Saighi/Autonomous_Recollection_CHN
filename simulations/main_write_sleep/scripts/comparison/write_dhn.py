#!/usr/bin/env python3
"""
DHN Training (Write Phase) - Hebbian and Storkey Learning Rules

Trains Discrete Hopfield Networks using both learning rules for comparison:
- Hebbian: W_ij = (1/N) * sum_mu(xi_i^mu * xi_j^mu)
- Storkey: Subtracts crosstalk for higher capacity (~0.42*N vs ~0.138*N)

Usage:
    python write_dhn.py

After running, use query_dhn.py for partial cue retrieval tests.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import (
    setup_dhn_train_experiment,
    run_cpp,
    build,
    DATA_DIR
)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Repetitions
NB_REPETITION = 2
REPETITIONS = list(range(NB_REPETITION))

# Network and pattern parameters (must match CHN for fair comparison)
NETWORK_SIZES = list(range(100, 1001, 100))  # [100, 200, ..., 1000]
NUM_PATTERNS = list(range(10, 101, 5))       # [10, 15, ..., 100]
CORRELATIONS = [0.9, 0.75, 0.5, 0.25, 0.0]   # Avoid 1.0 (causes infinite loop)
SPARSITY = 0.5

# Experiment names
HEBBIAN_NAME = "comparison_dhn_hebbian"
STORKEY_NAME = "comparison_dhn_storkey"


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("DHN WRITE PHASE - Hebbian and Storkey Training")
    print("=" * 70)

    # Build C++ executables
    print("\nBuilding C++ executables...")
    build()
    print("Build complete!")

    sims_per_rule = len(NETWORK_SIZES) * len(NUM_PATTERNS) * len(CORRELATIONS) * NB_REPETITION

    print(f"\nConfiguration:")
    print(f"  Repetitions: {NB_REPETITION}")
    print(f"  Network sizes: {NETWORK_SIZES[0]} to {NETWORK_SIZES[-1]}")
    print(f"  Pattern counts: {NUM_PATTERNS[0]} to {NUM_PATTERNS[-1]}")
    print(f"  Correlations: {CORRELATIONS}")
    print(f"  Sparsity: {SPARSITY}")
    print(f"  Networks per rule: {sims_per_rule}")
    print(f"  Total networks: {sims_per_rule * 2}")
    print("=" * 70)

    # =========================================================================
    # TRAIN HEBBIAN NETWORKS
    # =========================================================================
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

    # =========================================================================
    # TRAIN STORKEY NETWORKS
    # =========================================================================
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

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("WRITE PHASE COMPLETE")
    print("=" * 70)
    print(f"\nHebbian networks: {DATA_DIR / 'trained_networks' / HEBBIAN_NAME}")
    print(f"Storkey networks: {DATA_DIR / 'trained_networks' / STORKEY_NAME}")
    print(f"\nNext: Run query_dhn.py")
    print("=" * 70)
