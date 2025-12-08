# %% [markdown]
# # Spontaneous Recovery - Symmetric Transfer with Sparsity Sweep (Sleep Phase)
#
# This script runs sleep simulations on networks trained with SR_symmetric_sparsity_write.py
#
# Sleep configuration:
# - Full matrix inhibition (use_full_inhibition = 1)
# - Beta (inhibitory plasticity) sweep: [0.001, 0.005, 0.01, 0.05, 0.1]
# - Max queries: 50
# - Stochastic noise with stddev = 0.2
# - Stop on spurious patterns or when all patterns recovered

# %% Imports
import numpy as np
from pathlib import Path
import sys

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    setup_sleep_experiment,
    run_cpp,
    DATA_DIR
)

# %%=========================================================================
# CONFIGURATION SECTION
# ==========================================================================

# Sleep parameters
BETA_VALUES = [0.001, 0.005, 0.01, 0.05, 0.1]  # Inhibitory plasticity rate sweep
DELTA = 0.01  # Integration timestep
NOISE_DYNAMICS = 1  # Enable stochastic noise
STDDEV_DYNAMICS = 0.2  # Noise standard deviation
INIT_DRIVE = 0.5  # Initial state
MAX_QUERIES = 50  # Maximum retrieval attempts
STOP_ON_SPURIOUS = 1  # Stop when spurious pattern encountered
STOP_ON_ALL_FOUND = 1  # Stop when all patterns found
USE_FULL_INHIBITION = 1  # Use full matrix inhibition (not just diagonal)
USE_INHIBITION_PLASTICITY = 1  # Enable inhibition potentiation

# Experiment names
WRITE_EXPERIMENT_NAME = "SR_symmetric_sparsity"
SLEEP_NAME = "SR_symmetric_sparsity_sleep"

# %% [markdown]
# ## Sleep Phase (Spontaneous Recovery)

# %% Setup sleep experiment
trained_networks_dir = DATA_DIR / "trained_networks" / WRITE_EXPERIMENT_NAME

if not trained_networks_dir.exists():
    print("ERROR: Trained networks directory not found!")
    print(f"Expected: {trained_networks_dir}")
    print("\nPlease run SR_symmetric_sparsity_write.py first.")
    sys.exit(1)

# Count networks
network_dirs = list(trained_networks_dir.glob("sim_nb_*"))
total_networks = len(network_dirs)

print("="*70)
print("SLEEP PHASE - FULL MATRIX INHIBITION")
print("="*70)
print(f"Running sleep simulations on {total_networks} trained networks")
print(f"Source: {WRITE_EXPERIMENT_NAME}")
print(f"\nSleep configuration:")
print(f"  Beta values (inhibitory plasticity): {BETA_VALUES}")
print(f"  Delta (timestep): {DELTA}")
print(f"  Max queries: {MAX_QUERIES}")
print(f"  Noise dynamics: Enabled (stddev = {STDDEV_DYNAMICS})")
print(f"  Inhibition type: Full matrix")
print(f"  Stop on spurious: {STOP_ON_SPURIOUS}")
print(f"  Stop on all found: {STOP_ON_ALL_FOUND}")
print("="*70 + "\n")

sleep_config = setup_sleep_experiment(
    name=SLEEP_NAME,
    trained_networks_dir=trained_networks_dir,
    params={
        "delta": DELTA,
        "noise_dynamics": NOISE_DYNAMICS,
        "stddev_dynamics": STDDEV_DYNAMICS,
        "max_queries": MAX_QUERIES,
        "stop_on_spurious": STOP_ON_SPURIOUS,
        "stop_on_all_found": STOP_ON_ALL_FOUND,
        "save_trajectories": 0,
        "use_inhibition_plasticity": USE_INHIBITION_PLASTICITY,
        "use_full_inhibition": USE_FULL_INHIBITION,
    },
    varying_params={
        "beta": BETA_VALUES,
    }
)

print(f"Configuration saved to: {sleep_config}\n")

# %% Run sleep simulations
print("Starting sleep simulations...")
print("This will run each network with all beta values in parallel\n")
run_cpp("sleep", sleep_config)
print("\nSleep simulations complete!")

# %% Summary
print("\n" + "="*70)
print("SIMULATION COMPLETE!")
print("="*70)
print(f"\nResults saved to: {DATA_DIR / 'sleep_results' / SLEEP_NAME}")
print(f"\nTotal simulations: {total_networks} networks × {len(BETA_VALUES)} beta values")
print("="*70 + "\n")

# %%
