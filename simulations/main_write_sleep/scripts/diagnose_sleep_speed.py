#!/usr/bin/env python3
"""
Sleep Phase Speed Diagnosis

Analyzes sleep phase performance with detailed timing per query.
Tests N=500, K=30 with different parameters.
"""

import torch
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from pytorch_chn import (
    ContinuousHopfieldNetwork,
    train_patterns_adam,
    generate_patterns,
    get_device,
    check_cuda
)
from pytorch_chn.sleep import detect_pattern_match

# =============================================================================
# CONFIGURATION
# =============================================================================

NETWORK_SIZE = 500
NUM_PATTERNS = 30
SPARSITY = 0.5
RHO = 0.5
SEED = 42

# Training params
DRIVE_TARGET = 6.0
LEARNING_RATE = 0.01
MAX_TRAIN_ITER = 10000
TOLERANCE = 0.1
LEAK = 1.0

# Sleep params to test
MAX_QUERIES = 300
STOP_ON_SPURIOUS = True
STOP_ON_ALL_FOUND = True


def run_sleep_verbose(
    network: ContinuousHopfieldNetwork,
    patterns: torch.Tensor,
    max_queries: int,
    beta: float,
    delta: float,
    noise_stddev: float = 0.01,
    max_steps_per_query: int = 1000,
    convergence_tol: float = 1e-4,
    match_threshold: float = 0.8,
    stop_on_spurious: bool = True,
    stop_on_all_found: bool = True
):
    """Run sleep with detailed per-query logging."""

    network.delta = delta
    network.reset_inhibition()

    K = patterns.shape[0]
    found_patterns = set()
    n_spurious = 0
    total_steps = 0

    print(f"\n{'='*70}")
    print(f"Sleep Phase: beta={beta}, delta={delta}, max_steps={max_steps_per_query}")
    print(f"{'='*70}")
    print(f"{'Query':>5} | {'Steps':>6} | {'Time':>8} | {'Result':>12} | {'Found':>8} | {'Spurious':>8}")
    print(f"{'-'*70}")

    start_time = time.time()

    for q in range(max_queries):
        query_start = time.time()

        # Reset to neutral
        network.reset_to_neutral()

        # Run dynamics until convergence
        prev_v = network.v.clone()
        steps = 0

        for step in range(max_steps_per_query):
            network.depressed_step(noise_stddev)
            steps += 1

            if (step + 1) % 10 == 0:
                v_change = (network.v - prev_v).abs().max().item()
                if v_change < convergence_tol:
                    break
                prev_v = network.v.clone()

        total_steps += steps

        # Check pattern match
        matched = detect_pattern_match(network.v, patterns, match_threshold)

        query_time = time.time() - query_start

        if matched >= 0:
            is_new = matched not in found_patterns
            found_patterns.add(matched)
            result = f"P{matched}" + (" (new)" if is_new else " (dup)")
        else:
            n_spurious += 1
            result = "SPURIOUS"

        # Print progress
        print(f"{q+1:>5} | {steps:>6} | {query_time*1000:>6.1f}ms | {result:>12} | {len(found_patterns):>3}/{K:<4} | {n_spurious:>8}")

        # Potentiate inhibition
        network.pot_inhib_diag(beta)

        # Check stopping conditions
        if stop_on_spurious and n_spurious > 0:
            print(f"\n>>> Stopped: spurious attractor found")
            break

        if stop_on_all_found and len(found_patterns) == K:
            print(f"\n>>> Stopped: all patterns found!")
            break

    total_time = time.time() - start_time

    print(f"{'-'*70}")
    print(f"SUMMARY:")
    print(f"  Total queries: {q+1}")
    print(f"  Total steps: {total_steps}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Avg steps/query: {total_steps/(q+1):.1f}")
    print(f"  Avg time/query: {total_time/(q+1)*1000:.1f}ms")
    print(f"  Patterns found: {len(found_patterns)}/{K}")
    print(f"  Spurious: {n_spurious}")
    print(f"  AR Success: {len(found_patterns) == K and n_spurious == 0}")

    return {
        'queries': q + 1,
        'total_steps': total_steps,
        'total_time': total_time,
        'found': len(found_patterns),
        'spurious': n_spurious,
        'ar_success': len(found_patterns) == K and n_spurious == 0
    }


def main():
    print("=" * 70)
    print("SLEEP PHASE SPEED DIAGNOSIS")
    print("=" * 70)

    device = get_device()
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        check_cuda()

    print(f"\nNetwork: N={NETWORK_SIZE}, K={NUM_PATTERNS}, sparsity={SPARSITY}, rho={RHO}")

    # Generate patterns
    print("\nGenerating patterns...")
    patterns = generate_patterns(
        k=NUM_PATTERNS,
        n=NETWORK_SIZE,
        sparsity=SPARSITY,
        rho=RHO,
        device=str(device),
        seed=SEED
    )
    print(f"Patterns shape: {patterns.shape}")

    # Train network
    print("\nTraining network...")
    train_start = time.time()

    network = ContinuousHopfieldNetwork(
        n_neurons=NETWORK_SIZE,
        leak=LEAK,
        device=str(device)
    )

    W_trained, converged, history = train_patterns_adam(
        W=network.W,
        patterns=patterns,
        target_drive=DRIVE_TARGET,
        learning_rate=LEARNING_RATE,
        max_iter=MAX_TRAIN_ITER,
        tolerance=TOLERANCE,
        leak=LEAK,
        verbose=False
    )
    network.W = W_trained

    train_time = time.time() - train_start
    print(f"Training: {len(history)} iterations, {train_time:.2f}s, converged={converged}")

    # Test different parameter combinations
    test_configs = [
        # (beta, delta, max_steps)
        (0.1, 0.01, 1000),   # Current default
        (0.1, 0.05, 200),    # Larger delta, fewer steps
        (0.1, 0.1, 100),     # Even larger delta
        (0.5, 0.01, 1000),   # Higher beta
        (0.5, 0.05, 200),    # Higher beta + larger delta
        (1.0, 0.05, 200),    # Very high beta
    ]

    results = []

    for beta, delta, max_steps in test_configs:
        # Reset network weights (keep trained W, reset inhibition)
        network.W = W_trained.clone()

        result = run_sleep_verbose(
            network=network,
            patterns=patterns,
            max_queries=MAX_QUERIES,
            beta=beta,
            delta=delta,
            max_steps_per_query=max_steps,
            stop_on_spurious=STOP_ON_SPURIOUS,
            stop_on_all_found=STOP_ON_ALL_FOUND
        )
        result['beta'] = beta
        result['delta'] = delta
        result['max_steps'] = max_steps
        results.append(result)

    # Summary comparison
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Config':>25} | {'Queries':>7} | {'Time':>8} | {'Found':>6} | {'Spur':>5} | {'AR':>5}")
    print("-" * 70)

    for r in results:
        config = f"b={r['beta']}, d={r['delta']}, s={r['max_steps']}"
        ar = "YES" if r['ar_success'] else "NO"
        print(f"{config:>25} | {r['queries']:>7} | {r['total_time']:>6.2f}s | {r['found']:>6} | {r['spurious']:>5} | {ar:>5}")

    print("=" * 70)


if __name__ == "__main__":
    main()
