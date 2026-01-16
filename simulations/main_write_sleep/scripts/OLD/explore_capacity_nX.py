#%%
"""
Explore AR Capacity for N=300 Networks

Tests different training parameters (learning_rate, epsilon) and sleep parameters (beta)
to find optimal configuration for maximizing AR capacity.

Parameters tested:
- learning_rate (alpha): 0.0001 vs 0.00001
- epsilon: derived from alpha vs explicit lower values
- beta: 0.1, 0.05, 0.01

For rho=0.5, sparsity=0.5, N=300
"""

#%% Imports
import sys
from pathlib import Path
import numpy as np
import time

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import (
    setup_write_experiment,
    setup_sleep_experiment,
    run_cpp,
    build,
    load_final_results,
    DATA_DIR
)

#%% Configuration
NETWORK_SIZE = 1000
SPARSITY = 0.5
RHO = 0.5
LEAK = 5.0
DRIVE_TARGET = 6.0
MOMENTUM_COEF = 0.9

# Pattern counts to test (will stop early if capacity found)
PATTERN_COUNTS = list(range(25, 40, 2))  # [5, 10, 15, 20, 25, 30]

# Configurations to test: (name, learning_rate, epsilon_learning, beta)
# epsilon_learning = None means use default (learning_rate / 1e6)
CONFIGS = [
    # Lower beta variants

    ("alpha1e-4_eps1e-12_beta0.1", 0.0001, 0.001, 0.1),
]

# Sleep parameters (fixed)
DELTA = 0.01
NOISE_DYNAMICS = 1
STDDEV_DYNAMICS = 0.01
MAX_QUERIES = 200

#%% Build
print("=" * 70)
print("EXPLORING AR CAPACITY - N=300, rho=0.5")
print("=" * 70)

build()
print("Build complete!\n")

#%% Helper to run one configuration
def test_config(name, learning_rate, epsilon_learning, beta, pattern_counts):
    """Test a configuration across pattern counts. Returns max capacity."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"  alpha={learning_rate}, epsilon={epsilon_learning or f'{learning_rate/1e6:.0e} (default)'}, beta={beta}")
    print(f"{'='*60}")

    max_capacity = 0

    for num_patterns in pattern_counts:
        exp_name = f"explore_{name}_k{num_patterns}"

        # Setup training params
        train_params = {
            "leak": LEAK,
            "drive_target": DRIVE_TARGET,
            "learning_rate": learning_rate,
            "max_iter": 1000000,  # High enough for convergence
            "momentum_coef": MOMENTUM_COEF,
            "sparsity": SPARSITY,
        }
        if epsilon_learning is not None:
            train_params["epsilon_learning"] = epsilon_learning

        # Run training
        print(f"\n  K={num_patterns}: Training...", end=" ", flush=True)
        t0 = time.time()

        write_config = setup_write_experiment(
            name=exp_name,
            params=train_params,
            varying_params={
                "network_size": [NETWORK_SIZE],
                "num_patterns": [num_patterns],
                "rho": [RHO],
                "seed": [0, 1],  # 2 repetitions
            },
            native_pattern_generation=True
        )
        run_cpp("write", write_config, verbose=False)
        train_time = time.time() - t0
        print(f"({train_time:.1f}s)", end=" ", flush=True)

        # Run sleep
        print("Sleeping...", end=" ", flush=True)
        t0 = time.time()

        trained_dir = DATA_DIR / "trained_networks" / exp_name
        sleep_config = setup_sleep_experiment(
            name=f"{exp_name}_sleep",
            trained_networks_dir=trained_dir,
            params={
                "beta": beta,
                "delta": DELTA,
                "noise_dynamics": NOISE_DYNAMICS,
                "stddev_dynamics": STDDEV_DYNAMICS,
                "max_queries": MAX_QUERIES,
                "stop_on_spurious": 1,
                "stop_on_all_found": 1,
                "save_trajectories": 0,
            }
        )
        run_cpp("sleep", sleep_config, verbose=False)
        sleep_time = time.time() - t0
        print(f"({sleep_time:.1f}s)", end=" ", flush=True)

        # Check results
        sleep_dir = DATA_DIR / "sleep_results" / f"{exp_name}_sleep"
        results = load_final_results(sleep_dir)

        if len(results) == 0:
            print("NO RESULTS!")
            continue

        ar_success_rate = results['all_recovered_before_spurious'].mean()
        print(f"AR={ar_success_rate*100:.0f}%", end="")

        if ar_success_rate >= 0.9:
            max_capacity = num_patterns
            print(" [PASS]")
        else:
            print(" [FAIL - stopping]")
            break  # Stop testing higher pattern counts

    return max_capacity

#%% Run all configurations
results_summary = []

for name, alpha, epsilon, beta in CONFIGS:
    capacity = test_config(name, alpha, epsilon, beta, PATTERN_COUNTS)
    results_summary.append({
        'name': name,
        'alpha': alpha,
        'epsilon': epsilon if epsilon else alpha/1e6,
        'beta': beta,
        'capacity': capacity
    })

#%% Summary
print("\n" + "=" * 70)
print("SUMMARY: AR Capacity at 90% threshold (N=300, rho=0.5)")
print("=" * 70)
print(f"{'Config':<35} {'Alpha':<12} {'Epsilon':<12} {'Beta':<8} {'Capacity':<10}")
print("-" * 70)

for r in sorted(results_summary, key=lambda x: -x['capacity']):
    print(f"{r['name']:<35} {r['alpha']:<12.0e} {r['epsilon']:<12.0e} {r['beta']:<8} {r['capacity']:<10}")

best = max(results_summary, key=lambda x: x['capacity'])
print("\n" + "=" * 70)
print(f"BEST CONFIG: {best['name']}")
print(f"  Capacity: {best['capacity']} patterns at 90% AR success")
print(f"  alpha={best['alpha']}, epsilon={best['epsilon']:.0e}, beta={best['beta']}")
print("=" * 70)
