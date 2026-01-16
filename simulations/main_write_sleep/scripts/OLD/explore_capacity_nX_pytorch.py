#%%
"""
Explore AR Capacity for Networks - PyTorch GPU Backend

Tests different training parameters (learning_rate, epsilon) and sleep parameters (beta)
to find optimal configuration for maximizing AR capacity.

Parameters tested:
- learning_rate (alpha): 0.0001 vs 0.00001
- epsilon: derived from alpha vs explicit lower values
- beta: 0.1, 0.05, 0.01

For rho=0.5, sparsity=0.5
"""

#%% Imports
import sys
from pathlib import Path
import numpy as np
import time
import torch
from typing import Dict, List, Tuple, Optional

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

from pytorch_chn import (
    ContinuousHopfieldNetwork,
    train_patterns_sgd,
    run_sleep_phase,
    generate_patterns,
    get_device,
    check_cuda
)

PROJECT_DIR = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_DIR / "data"

#%% Configuration
NETWORK_SIZE = 1000
SPARSITY = 0.5
RHO = 0.5
LEAK = 5.0
DRIVE_TARGET = 6.0
MOMENTUM_COEF = 0.9

# Pattern counts to test (will stop early if capacity found)
PATTERN_COUNTS = list(range(25, 40, 2))  # [25, 27, 29, 31, 33, 35, 37, 39]

# Configurations to test: (name, learning_rate, epsilon_learning, beta)
# epsilon_learning = None means use default (learning_rate / 1e6)
CONFIGS = [
    ("alpha1e-4_eps1e-12_beta0.1", 0.0001, 0.001, 0.1),
]

# Sleep parameters (fixed)
DELTA = 0.01
NOISE_DYNAMICS = 1
STDDEV_DYNAMICS = 0.01
MAX_QUERIES = 200

#%% Build message
print("=" * 70)
print("EXPLORING AR CAPACITY - PyTorch GPU Backend")
print(f"N={NETWORK_SIZE}, rho={RHO}, leak={LEAK}")
print("=" * 70)

device = get_device()
print(f"\nDevice: {device}")
if device.type == "cuda":
    check_cuda()
print()

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

        # Compute tolerance matching C++ behavior
        tolerance = epsilon_learning if epsilon_learning is not None else learning_rate / 1e6

        print(f"\n  K={num_patterns}: Training...", end=" ", flush=True)
        t0 = time.time()

        ar_successes = 0
        total_runs = 0

        # 2 repetitions (seeds 0 and 1)
        for seed in [0, 1]:
            # Generate patterns
            patterns = generate_patterns(
                k=num_patterns,
                n=NETWORK_SIZE,
                sparsity=SPARSITY,
                rho=RHO,
                device=str(device),
                seed=seed
            )

            # Initialize network
            network = ContinuousHopfieldNetwork(
                n_neurons=NETWORK_SIZE,
                leak=LEAK,
                delta=DELTA,
                device=str(device)
            )

            # Train with SGD + momentum (matches C++)
            W_trained, converged, history = train_patterns_sgd(
                W=network.W,
                patterns=patterns,
                target_drive=DRIVE_TARGET,
                learning_rate=learning_rate,
                momentum=MOMENTUM_COEF,
                max_iter=1000000,
                tolerance=tolerance,
                leak=LEAK,
                verbose=False
            )

            network.W = W_trained

            train_time = time.time() - t0
            print(f"({train_time:.1f}s)", end=" ", flush=True)

            # Run sleep
            print("Sleeping...", end=" ", flush=True)
            t0 = time.time()

            results = run_sleep_phase(
                network=network,
                patterns=patterns,
                max_queries=MAX_QUERIES,
                beta=beta,
                delta=DELTA,
                noise_stddev=STDDEV_DYNAMICS,
                max_steps_per_query=1000,
                stop_on_spurious=True,
                stop_on_all_found=True,
                verbose=False
            )

            sleep_time = time.time() - t0
            print(f"({sleep_time:.1f}s)", end=" ", flush=True)

            if results.all_recovered_before_spurious:
                ar_successes += 1
            total_runs += 1

        ar_success_rate = ar_successes / total_runs
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
print(f"SUMMARY: AR Capacity at 90% threshold (N={NETWORK_SIZE}, rho={RHO})")
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
