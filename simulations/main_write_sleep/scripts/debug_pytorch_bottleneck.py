#!/usr/bin/env python3
"""
Debug script to identify bottlenecks in PyTorch CHN simulation.

Tests pattern generation and training phases separately with timing
and timeout detection to find where the simulation gets stuck.

Usage:
    python debug_pytorch_bottleneck.py
"""

import torch
import numpy as np
import time
import csv
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import sys

# Add scripts directory to path
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from pytorch_chn import get_device, check_cuda


# =============================================================================
# CONFIGURATION
# =============================================================================

# Test parameters
NETWORK_SIZES = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
PATTERN_COUNTS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
RHO = 0.9
SPARSITY = 0.5
SEED = 42

# Timeout thresholds
PATTERN_GEN_TIMEOUT = 10.0  # seconds
TRAINING_TIMEOUT = 30.0     # seconds

# Training parameters (matching run_chn_sim_pytorch.py)
DRIVE_TARGET = 6.0
LEARNING_RATE = 0.01
MAX_ITER = 10000
TOLERANCE = 0.1
LEAK = 1.0


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PatternGenStats:
    """Statistics from pattern generation."""
    patterns_generated: int = 0
    total_attempts: int = 0
    collisions: int = 0
    time_elapsed: float = 0.0
    timed_out: bool = False
    per_pattern_times: List[float] = field(default_factory=list)

    @property
    def collision_rate(self) -> float:
        if self.total_attempts == 0:
            return 0.0
        return self.collisions / self.total_attempts

    def __str__(self) -> str:
        if self.timed_out:
            return (f"TIMEOUT after {self.time_elapsed:.1f}s "
                    f"({self.patterns_generated} patterns, "
                    f"{self.collisions} collisions, "
                    f"collision_rate={self.collision_rate*100:.0f}%)")
        return (f"{self.time_elapsed:.3f}s "
                f"({self.patterns_generated} patterns, "
                f"{self.collisions} collisions)")


@dataclass
class TrainingStats:
    """Statistics from training phase."""
    iterations: int = 0
    time_elapsed: float = 0.0
    converged: bool = False
    final_error: float = 0.0
    timed_out: bool = False
    plateau_detected: bool = False
    error_history: List[float] = field(default_factory=list)

    def __str__(self) -> str:
        if self.timed_out:
            return (f"TIMEOUT after {self.time_elapsed:.1f}s "
                    f"({self.iterations} iters, error={self.final_error:.4f})")
        status = "converged" if self.converged else "not_converged"
        extra = ", PLATEAU" if self.plateau_detected else ""
        return (f"{self.time_elapsed:.3f}s "
                f"({self.iterations} iters, {status}, "
                f"error={self.final_error:.4f}{extra})")


@dataclass
class TestResult:
    """Result from testing a single configuration."""
    network_size: int
    num_patterns: int
    rho: float
    sparsity: float
    pattern_gen: PatternGenStats = field(default_factory=PatternGenStats)
    training: Optional[TrainingStats] = None
    status: str = "PENDING"


# =============================================================================
# INSTRUMENTED PATTERN GENERATION
# =============================================================================

def generate_patterns_instrumented(
    k: int,
    n: int,
    sparsity: float = 0.5,
    rho: float = 0.5,
    seed: int = 42,
    timeout: float = PATTERN_GEN_TIMEOUT,
    device: str = "cuda"
) -> Tuple[Optional[torch.Tensor], PatternGenStats]:
    """Generate patterns with timing and collision tracking.

    Args:
        k: Number of patterns to generate
        n: Network size (neurons)
        sparsity: Fraction of active units
        rho: Pattern correlation
        seed: Random seed
        timeout: Maximum time in seconds
        device: Device for output tensor

    Returns:
        Tuple of (patterns tensor or None if timed out, stats)
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    stats = PatternGenStats()
    start_time = time.time()

    # Number of active units
    nb_winners = max(1, int(sparsity * n))

    # Base pattern: first nb_winners are active
    base = np.zeros(n, dtype=bool)
    base[:nb_winners] = True

    # Number of flips based on correlation
    num_flips = int((1.0 - rho) * nb_winners)

    patterns = []
    seen = set()

    while len(patterns) < k:
        # Check timeout
        elapsed = time.time() - start_time
        if elapsed > timeout:
            stats.time_elapsed = elapsed
            stats.timed_out = True
            stats.patterns_generated = len(patterns)
            return None, stats

        pattern_start = time.time()
        stats.total_attempts += 1

        pattern = base.copy()

        # Perform balanced flips
        for _ in range(num_flips):
            ones = np.where(pattern)[0]
            zeros = np.where(~pattern)[0]
            if len(ones) > 0 and len(zeros) > 0:
                pattern[np.random.choice(ones)] = False
                pattern[np.random.choice(zeros)] = True

        # Check uniqueness
        key = tuple(pattern.tolist())
        if key not in seen:
            seen.add(key)
            patterns.append(pattern)
            stats.per_pattern_times.append(time.time() - pattern_start)
        else:
            stats.collisions += 1

    # Success
    stats.time_elapsed = time.time() - start_time
    stats.patterns_generated = len(patterns)

    patterns_np = np.array(patterns, dtype=np.float32)
    patterns_torch = torch.from_numpy(patterns_np).to(device)

    return patterns_torch, stats


# =============================================================================
# INSTRUMENTED TRAINING
# =============================================================================

def train_patterns_instrumented(
    W: torch.Tensor,
    patterns: torch.Tensor,
    target_drive: float = DRIVE_TARGET,
    learning_rate: float = LEARNING_RATE,
    max_iter: int = MAX_ITER,
    tolerance: float = TOLERANCE,
    leak: float = LEAK,
    timeout: float = TRAINING_TIMEOUT
) -> Tuple[torch.Tensor, TrainingStats]:
    """Train network with timing and plateau detection.

    Args:
        W: Initial weight matrix
        patterns: Binary patterns [K, N]
        target_drive: Target drive magnitude
        learning_rate: Adam learning rate
        max_iter: Maximum iterations
        tolerance: Convergence threshold
        leak: Leak rate
        timeout: Maximum time in seconds

    Returns:
        Tuple of (trained weights, stats)
    """
    stats = TrainingStats()
    start_time = time.time()

    # Setup Adam optimizer
    W = W.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([W], lr=learning_rate)

    # Precompute target drives
    target_drives = (2 * patterns - 1) * target_drive

    # For plateau detection
    plateau_window = 1000
    plateau_threshold = 0.001  # error must decrease by this much

    for i in range(max_iter):
        # Check timeout
        elapsed = time.time() - start_time
        if elapsed > timeout:
            stats.time_elapsed = elapsed
            stats.timed_out = True
            stats.iterations = i
            stats.final_error = stats.error_history[-1] if stats.error_history else float('inf')
            return W.detach(), stats

        optimizer.zero_grad()

        # Compute target rates
        with torch.no_grad():
            target_rates = torch.sigmoid(target_drives)

        # Predicted equilibrium
        u_hat = (target_rates @ W) / leak

        # Loss
        loss = ((target_drives - u_hat) ** 2).mean()

        loss.backward()
        optimizer.step()

        # Enforce constraints
        with torch.no_grad():
            W.data = (W.data + W.data.T) / 2
            W.data.fill_diagonal_(0)

        # Track error
        max_error = (target_drives - u_hat).abs().max().item()
        stats.error_history.append(max_error)

        # Check convergence
        if max_error < tolerance:
            stats.converged = True
            stats.iterations = i + 1
            stats.time_elapsed = time.time() - start_time
            stats.final_error = max_error
            return W.detach(), stats

        # Check for plateau
        if i >= plateau_window:
            old_error = stats.error_history[i - plateau_window]
            if old_error - max_error < plateau_threshold:
                stats.plateau_detected = True

    # Did not converge
    stats.iterations = max_iter
    stats.time_elapsed = time.time() - start_time
    stats.final_error = stats.error_history[-1] if stats.error_history else float('inf')
    return W.detach(), stats


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_single_config(
    network_size: int,
    num_patterns: int,
    rho: float = RHO,
    sparsity: float = SPARSITY,
    seed: int = SEED,
    device: str = "cuda"
) -> TestResult:
    """Test a single configuration and return detailed results."""
    result = TestResult(
        network_size=network_size,
        num_patterns=num_patterns,
        rho=rho,
        sparsity=sparsity
    )

    # Phase 1: Pattern Generation
    patterns, pattern_stats = generate_patterns_instrumented(
        k=num_patterns,
        n=network_size,
        sparsity=sparsity,
        rho=rho,
        seed=seed,
        device=device
    )
    result.pattern_gen = pattern_stats

    if pattern_stats.timed_out:
        result.status = "PATTERN_GEN_TIMEOUT"
        return result

    # Phase 2: Training
    W = torch.zeros(network_size, network_size, device=device)
    W_trained, train_stats = train_patterns_instrumented(
        W=W,
        patterns=patterns,
        target_drive=DRIVE_TARGET,
        learning_rate=LEARNING_RATE,
        max_iter=MAX_ITER,
        tolerance=TOLERANCE,
        leak=LEAK
    )
    result.training = train_stats

    if train_stats.timed_out:
        result.status = "TRAINING_TIMEOUT"
    elif train_stats.plateau_detected and not train_stats.converged:
        result.status = "TRAINING_PLATEAU"
    elif not train_stats.converged:
        result.status = "TRAINING_NOT_CONVERGED"
    else:
        result.status = "OK"

    return result


def run_sweep(
    network_sizes: List[int] = NETWORK_SIZES,
    pattern_counts: List[int] = PATTERN_COUNTS,
    rho: float = RHO,
    sparsity: float = SPARSITY,
    output_csv: Optional[str] = None,
    device: str = "cuda"
) -> List[TestResult]:
    """Run sequential sweep over all configurations."""

    results = []
    total_tests = len(network_sizes) * len(pattern_counts)
    test_num = 0

    for n in network_sizes:
        for k in pattern_counts:
            test_num += 1
            print(f"\n[{test_num}/{total_tests}] Testing N={n}, K={k}, rho={rho}:")

            result = test_single_config(
                network_size=n,
                num_patterns=k,
                rho=rho,
                sparsity=sparsity,
                seed=SEED,
                device=device
            )
            results.append(result)

            # Print pattern generation results
            print(f"  Pattern generation: {result.pattern_gen}")

            # Print training results if available
            if result.training is not None:
                print(f"  Training: {result.training}")

            # Print status
            status_color = "" if result.status == "OK" else ">>> "
            print(f"  {status_color}Status: {result.status}")

            # Early warning for slow pattern generation
            if result.pattern_gen.collision_rate > 0.5 and not result.pattern_gen.timed_out:
                print(f"  [WARNING] High collision rate: {result.pattern_gen.collision_rate*100:.0f}%")

    # Save to CSV if requested
    if output_csv:
        save_results_csv(results, output_csv)
        print(f"\nResults saved to: {output_csv}")

    return results


def save_results_csv(results: List[TestResult], filepath: str):
    """Save results to CSV file."""
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'network_size', 'num_patterns', 'rho', 'sparsity',
            'pattern_gen_time', 'pattern_gen_attempts', 'pattern_gen_collisions',
            'pattern_gen_ok', 'collision_rate',
            'train_time', 'train_iters', 'train_converged',
            'train_final_error', 'train_plateau', 'train_ok',
            'status'
        ])

        for r in results:
            pg = r.pattern_gen
            tr = r.training

            writer.writerow([
                r.network_size,
                r.num_patterns,
                r.rho,
                r.sparsity,
                f"{pg.time_elapsed:.4f}",
                pg.total_attempts,
                pg.collisions,
                not pg.timed_out,
                f"{pg.collision_rate:.4f}",
                f"{tr.time_elapsed:.4f}" if tr else "",
                tr.iterations if tr else "",
                tr.converged if tr else "",
                f"{tr.final_error:.6f}" if tr else "",
                tr.plateau_detected if tr else "",
                not tr.timed_out if tr else "",
                r.status
            ])


def print_summary(results: List[TestResult]):
    """Print summary of all results."""
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Count statuses
    status_counts = {}
    for r in results:
        status_counts[r.status] = status_counts.get(r.status, 0) + 1

    print("\nStatus counts:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")

    # Find boundary cases
    print("\nFirst failures by network size:")
    by_n = {}
    for r in results:
        if r.network_size not in by_n:
            by_n[r.network_size] = []
        by_n[r.network_size].append(r)

    for n in sorted(by_n.keys()):
        failures = [r for r in by_n[n] if r.status != "OK"]
        if failures:
            first_fail = min(failures, key=lambda x: x.num_patterns)
            print(f"  N={n}: first failure at K={first_fail.num_patterns} ({first_fail.status})")
        else:
            print(f"  N={n}: all OK")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("CHN PYTORCH BOTTLENECK DEBUGGER")
    print("=" * 70)

    # Check device
    device = get_device()
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        check_cuda()

    # Configuration
    print(f"\nConfiguration:")
    print(f"  Network sizes: {NETWORK_SIZES[0]} to {NETWORK_SIZES[-1]}")
    print(f"  Pattern counts: {PATTERN_COUNTS[0]} to {PATTERN_COUNTS[-1]}")
    print(f"  Correlation (rho): {RHO}")
    print(f"  Sparsity: {SPARSITY}")
    print(f"  Pattern gen timeout: {PATTERN_GEN_TIMEOUT}s")
    print(f"  Training timeout: {TRAINING_TIMEOUT}s")

    total_tests = len(NETWORK_SIZES) * len(PATTERN_COUNTS)
    print(f"\nTotal configurations to test: {total_tests}")

    # Run sweep
    output_csv = SCRIPT_DIR / "debug_timing_results.csv"
    results = run_sweep(
        network_sizes=NETWORK_SIZES,
        pattern_counts=PATTERN_COUNTS,
        rho=RHO,
        sparsity=SPARSITY,
        output_csv=str(output_csv),
        device=str(device)
    )

    # Print summary
    print_summary(results)

    print("\n" + "=" * 70)
    print("DEBUG COMPLETE")
    print("=" * 70)
