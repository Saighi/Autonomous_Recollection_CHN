#%%
"""
Explore AR Capacity for N=300 Networks - Symmetric Transfer, No Beta

Tests symmetric transfer function (centered at 0 instead of 0.5) without
inhibitory plasticity (beta=0) to see if AR can work without potentiation.

Parameters tested:
- learning_rate (alpha): 0.0001 vs 0.00001
- epsilon: derived from alpha vs explicit lower values
- beta = 0 (no potentiation)
- symmetric_transfer = True

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
NETWORK_SIZE = 300
SPARSITY = 0.5
RHO = 0.5
LEAK = 1.0
DRIVE_TARGET = 6.0
MOMENTUM_COEF = 0.9

# Pattern counts to test (will stop early if capacity found)
PATTERN_COUNTS = list(range(5, 31, 5))  # [5, 10, 15, 20, 25, 30]

# Configurations to test: (name, learning_rate, epsilon_learning)
# All use symmetric_transfer=True and beta=0
CONFIGS = [
    # Standard alpha values
    ("sym_alpha1e-4", 0.0001, None),
    ("sym_alpha1e-5", 0.00001, None),

    # Lower epsilon with alpha=0.0001
    ("sym_alpha1e-4_eps1e-11", 0.0001, 1e-11),
    ("sym_alpha1e-4_eps1e-12", 0.0001, 1e-12),

    # Even lower alpha
    ("sym_alpha1e-6", 0.000001, None),
]

# Sleep parameters (fixed)
BETA = 0.0  # NO potentiation
DELTA = 0.01
NOISE_DYNAMICS = 1
STDDEV_DYNAMICS = 0.01
MAX_QUERIES = 200

#%% Build
print("=" * 70)
print("EXPLORING AR CAPACITY - SYMMETRIC TRANSFER, NO BETA")
print("N=300, rho=0.5, beta=0")
print("=" * 70)

build()
print("Build complete!\n")

#%% Helper to run one configuration
def test_config(name, learning_rate, epsilon_learning, pattern_counts):
    """Test a configuration across pattern counts. Returns max capacity."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"  alpha={learning_rate}, epsilon={epsilon_learning or f'{learning_rate/1e6:.0e} (default)'}")
    print(f"  symmetric_transfer=True, beta=0")
    print(f"{'='*60}")

    max_capacity = 0

    for num_patterns in pattern_counts:
        exp_name = f"explore_{name}_k{num_patterns}"

        # Setup training params with symmetric transfer
        train_params = {
            "leak": LEAK,
            "drive_target": DRIVE_TARGET,
            "learning_rate": learning_rate,
            "max_iter": 1000000,  # High enough for convergence
            "momentum_coef": MOMENTUM_COEF,
            "sparsity": SPARSITY,
            "symmetric_transfer": 1.0,  # Enable symmetric transfer
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

        # Run sleep with beta=0 and symmetric transfer
        print("Sleeping...", end=" ", flush=True)
        t0 = time.time()

        trained_dir = DATA_DIR / "trained_networks" / exp_name
        sleep_config = setup_sleep_experiment(
            name=f"{exp_name}_sleep",
            trained_networks_dir=trained_dir,
            params={
                "beta": BETA,  # No potentiation
                "delta": DELTA,
                "noise_dynamics": NOISE_DYNAMICS,
                "stddev_dynamics": STDDEV_DYNAMICS,
                "max_queries": MAX_QUERIES,
                "stop_on_spurious": 1,
                "stop_on_all_found": 1,
                "save_trajectories": 0,
                "symmetric_transfer": 1.0,  # Must match training
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
        avg_found = results['nb_fnd_pat'].mean() if 'nb_fnd_pat' in results.columns else 0
        avg_spurious = results['nb_spurious'].mean() if 'nb_spurious' in results.columns else 0

        print(f"AR={ar_success_rate*100:.0f}% (found={avg_found:.1f}, spur={avg_spurious:.1f})", end="")

        if ar_success_rate >= 0.9:
            max_capacity = num_patterns
            print(" [PASS]")
        else:
            print(" [FAIL - stopping]")
            break  # Stop testing higher pattern counts

    return max_capacity

#%% Run all configurations
results_summary = []

for name, alpha, epsilon in CONFIGS:
    capacity = test_config(name, alpha, epsilon, PATTERN_COUNTS)
    results_summary.append({
        'name': name,
        'alpha': alpha,
        'epsilon': epsilon if epsilon else alpha/1e6,
        'capacity': capacity
    })

#%% Summary
print("\n" + "=" * 70)
print("SUMMARY: AR Capacity - Symmetric Transfer, No Beta (N=300, rho=0.5)")
print("=" * 70)
print(f"{'Config':<30} {'Alpha':<12} {'Epsilon':<12} {'Capacity':<10}")
print("-" * 70)

for r in sorted(results_summary, key=lambda x: -x['capacity']):
    print(f"{r['name']:<30} {r['alpha']:<12.0e} {r['epsilon']:<12.0e} {r['capacity']:<10}")

best = max(results_summary, key=lambda x: x['capacity'])
print("\n" + "=" * 70)
if best['capacity'] > 0:
    print(f"BEST CONFIG: {best['name']}")
    print(f"  Capacity: {best['capacity']} patterns at 90% AR success")
    print(f"  alpha={best['alpha']}, epsilon={best['epsilon']:.0e}")
else:
    print("NO CONFIGURATION ACHIEVED 90% AR SUCCESS")
    print("Symmetric transfer without beta potentiation may not support AR")
print("=" * 70)
