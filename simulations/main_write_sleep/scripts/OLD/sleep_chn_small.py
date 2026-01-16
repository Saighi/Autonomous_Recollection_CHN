#%%
"""
CHN Sleep Phase (Autonomous Retrieval) - C++ Backend for Small Networks

Runs sleep consolidation on trained networks to test Autonomous Retrieval capacity.
Uses C++ backend for efficient simulation on smaller ne tworks.

Prerequisites: Run write_chn_small.py first to generate trained networks.
"""

#%% Imports
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import (
    setup_sleep_experiment,
    run_cpp,
    build,
    DATA_DIR
)

#%% Configuration - Sleep Parameters
BETA = 0.1              # Inhibitory plasticity rate
DELTA = 0.01            # Integration timestep
NOISE_DYNAMICS = 1      # Enable stochastic noise
STDDEV_DYNAMICS = 0.01  # Noise standard deviation
MAX_QUERIES = 200       # Maximum retrieval attempts
STOP_ON_SPURIOUS = 1    # Stop after spurious (0=no, 1=yes)
STOP_ON_ALL_FOUND = 1   # Stop after all found (0=no, 1=yes)

#%% Configuration - Experiment Names
TRAINED_NETWORKS_NAME = "comparison_chn_cpp"
SLEEP_RESULTS_NAME = "comparison_chn_cpp_sleep"

#%% Build C++ Executables
print("=" * 70)
print("CHN SLEEP PHASE - C++ Backend (Small Networks)")
print("=" * 70)

print("\nBuilding C++ executables...")
build()
print("Build complete!")

#%% Verify Trained Networks Exist
trained_dir = DATA_DIR / "trained_networks" / TRAINED_NETWORKS_NAME

if not trained_dir.exists():
    raise FileNotFoundError(
        f"Trained networks not found at {trained_dir}\n"
        "Please run write_chn_small.py first."
    )

sim_dirs = [d for d in trained_dir.iterdir() if d.is_dir() and d.name.startswith("sim_nb_")]
print(f"\nFound {len(sim_dirs)} trained networks")

#%% Print Configuration Summary
print(f"\nConfiguration:")
print(f"  Beta (inhibitory plasticity): {BETA}")
print(f"  Delta (timestep): {DELTA}")
print(f"  Noise stddev: {STDDEV_DYNAMICS}")
print(f"  Max queries: {MAX_QUERIES}")
print(f"  Networks to process: {len(sim_dirs)}")
print("=" * 70)

#%% Setup Experiment Configuration
sleep_config = setup_sleep_experiment(
    name=SLEEP_RESULTS_NAME,
    trained_networks_dir=trained_dir,
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

print(f"Configuration saved to: {sleep_config}")

#%% Run Sleep Simulations
print("\nStarting sleep simulations...")
run_cpp("sleep", sleep_config)

#%% Summary
print("\n" + "=" * 70)
print("SLEEP PHASE COMPLETE")
print("=" * 70)
print(f"\nResults saved to: {DATA_DIR / 'sleep_results' / SLEEP_RESULTS_NAME}")
print("=" * 70)
