#!/usr/bin/env python3
"""
CHN Sleep Phase (Autonomous Retrieval) - PyTorch GPU Backend for Large Networks

Runs sleep consolidation on trained networks to test Autonomous Retrieval capacity.
Networks start from neutral state (0.5) and diagonal inhibitory plasticity enables
sequential pattern retrieval.

Usage:
    python sleep_chn_large.py

Prerequisites: Run write_chn_large.py first to generate trained networks.
"""

import torch
import pandas as pd
import time
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass
import sys

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# Add scripts directory to path
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

from pytorch_chn import (
    ContinuousHopfieldNetwork,
    run_sleep_phase,
    get_device,
    check_cuda
)

# Base paths
PROJECT_DIR = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_DIR / "data"

# =============================================================================
# CONFIGURATION
# =============================================================================

# Sleep parameters
BETA = 0.15               # Inhibitory plasticity rate
DELTA = 0.05             # Integration timestep (larger = faster convergence)
NOISE_DYNAMICS = 0.01    # Noise std
MAX_QUERIES = 400        # Maximum retrieval attempts
MAX_STEPS_PER_QUERY = 200  # Fewer steps needed with larger delta
STOP_ON_SPURIOUS = True
STOP_ON_ALL_FOUND = True
LEAK = 1.0

# Experiment names
TRAINED_NETWORKS_NAME = "comparison_chn_pytorch_slightly_larger"
SLEEP_RESULTS_NAME = "comparison_chn_pytorch_slightly_larger_sleep"


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def save_parameters(params: Dict, filepath: Path):
    """Save parameters in C++ compatible format."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        for key, value in params.items():
            f.write(f"{key}={value}\n")


def save_patterns(patterns: torch.Tensor, filepath: Path):
    """Save patterns in C++ compatible format."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    patterns_np = patterns.cpu().numpy()
    with open(filepath, 'w') as f:
        for pattern in patterns_np:
            f.write(' '.join(str(int(x)) for x in pattern) + '\n')


def save_results_csv(results: List[Dict], filepath: Path):
    """Save results as CSV."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(filepath, index=False)


def load_weights(filepath: Path, device: str) -> torch.Tensor:
    """Load weight matrix from C++ format."""
    rows = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                rows.append([float(x) for x in line.strip().split()])
    return torch.tensor(rows, dtype=torch.float32, device=device)


def load_patterns(filepath: Path, device: str) -> torch.Tensor:
    """Load patterns from C++ format."""
    patterns = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                patterns.append([float(x == '1') for x in line.strip().split()])
    return torch.tensor(patterns, dtype=torch.float32, device=device)


def load_parameters(filepath: Path) -> Dict[str, float]:
    """Load parameters from key=value format."""
    params = {}
    with open(filepath, 'r') as f:
        for line in f:
            if '=' in line:
                key, value = line.strip().split('=', 1)
                try:
                    params[key] = float(value)
                except ValueError:
                    params[key] = value
    return params


@dataclass
class SleepStats:
    """Statistics collected during sleep phase."""
    total: int = 0
    all_recovered: int = 0
    total_found: int = 0
    total_patterns: int = 0
    total_spurious: int = 0
    total_time: float = 0.0

    def update(self, n_found: int, n_patterns: int, n_spurious: int,
               all_before_spurious: bool, time_s: float):
        self.total += 1
        self.total_found += n_found
        self.total_patterns += n_patterns
        self.total_spurious += n_spurious
        if all_before_spurious:
            self.all_recovered += 1
        self.total_time += time_s

    @property
    def ar_success_rate(self) -> float:
        return self.all_recovered / self.total if self.total > 0 else 0

    @property
    def avg_recovery_rate(self) -> float:
        return self.total_found / self.total_patterns if self.total_patterns > 0 else 0

    def summary(self) -> str:
        return (f"AR success: {self.all_recovered}/{self.total} ({self.ar_success_rate*100:.1f}%) | "
                f"Avg recovery: {self.avg_recovery_rate*100:.1f}% | "
                f"Avg spurious: {self.total_spurious/max(1,self.total):.1f}")


def get_gpu_memory_str() -> str:
    """Get GPU memory usage string."""
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1e6
        return f"GPU mem: {used:.0f}MB"
    return ""


# =============================================================================
# SLEEP PHASE
# =============================================================================

def run_sleep_all(trained_dir: Path, output_dir: Path, device: str) -> SleepStats:
    """Run sleep phase on all trained networks."""
    sim_dirs = sorted([d for d in trained_dir.iterdir()
                       if d.is_dir() and d.name.startswith("sim_nb_")])

    if not sim_dirs:
        print(f"ERROR: No trained networks found in {trained_dir}")
        return SleepStats()

    stats = SleepStats()
    start_time = time.time()
    all_results = []

    if HAS_TQDM:
        pbar = tqdm(sim_dirs, desc="Sleep", unit="net")
    else:
        pbar = sim_dirs
        print(f"\nRunning sleep on {len(sim_dirs)} networks...")

    for i, sim_dir in enumerate(pbar):
        t0 = time.time()

        sim_id = int(sim_dir.name.split("_")[-1])
        out_sim_dir = output_dir / f"sim_nb_{sim_id}"
        out_sim_dir.mkdir(parents=True, exist_ok=True)

        # Load network
        W = load_weights(sim_dir / "weights.data", device)
        patterns = load_patterns(sim_dir / "patterns.data", device)
        params = load_parameters(sim_dir / "parameters.data")

        n_neurons = W.shape[0]
        network = ContinuousHopfieldNetwork(
            n_neurons=n_neurons,
            leak=params.get('leak', LEAK),
            delta=DELTA,
            device=device
        )
        network.W = W

        # Run sleep
        results = run_sleep_phase(
            network=network,
            patterns=patterns,
            max_queries=MAX_QUERIES,
            beta=BETA,
            delta=DELTA,
            noise_stddev=NOISE_DYNAMICS,
            max_steps_per_query=MAX_STEPS_PER_QUERY,
            stop_on_spurious=STOP_ON_SPURIOUS,
            stop_on_all_found=STOP_ON_ALL_FOUND,
            verbose=False
        )

        sleep_time = time.time() - t0
        n_patterns = patterns.shape[0]

        stats.update(
            n_found=len(results.found_patterns),
            n_patterns=n_patterns,
            n_spurious=results.n_spurious,
            all_before_spurious=results.all_recovered_before_spurious,
            time_s=sleep_time
        )

        # Build result rows
        found_so_far = set()
        spurious_so_far = 0
        rows = []

        for q, query in enumerate(results.queries):
            if query.matched_pattern >= 0:
                found_so_far.add(query.matched_pattern)
            else:
                spurious_so_far += 1

            rows.append({
                'sim_ID': sim_id,
                'query_iter': q,
                'nb_fnd_pat': len(found_so_far),
                'nb_spurious': spurious_so_far,
                'all_recovered_before_spurious': int(results.all_recovered_before_spurious),
                **params
            })

        all_results.extend(rows)

        # Save per-simulation results
        if rows:
            save_results_csv(rows, out_sim_dir / "results.data")
        save_patterns(patterns, out_sim_dir / "patterns.data")
        save_parameters(params, out_sim_dir / "parameters.data")

        if HAS_TQDM:
            pbar.set_postfix_str(
                f"N={n_neurons} K={n_patterns} | "
                f"AR={stats.ar_success_rate*100:.0f}% | {get_gpu_memory_str()}"
            )
        elif (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(sim_dirs)}] {stats.summary()}")

    if HAS_TQDM:
        pbar.close()

    # Save aggregated results
    if all_results:
        save_results_csv(all_results, output_dir / "all_simulation_data.csv")

    elapsed = time.time() - start_time
    print(f"\nSleep phase complete in {elapsed/60:.1f} minutes")
    print(f"{stats.summary()}")

    return stats


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("CHN SLEEP PHASE - PyTorch GPU (Large Networks)")
    print("=" * 70)

    device = get_device()
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        check_cuda()

    trained_dir = DATA_DIR / "trained_networks" / TRAINED_NETWORKS_NAME
    output_dir = DATA_DIR / "sleep_results" / SLEEP_RESULTS_NAME

    print(f"\nConfiguration:")
    print(f"  Beta (inhibitory plasticity): {BETA}")
    print(f"  Delta (timestep): {DELTA}")
    print(f"  Max steps per query: {MAX_STEPS_PER_QUERY}")
    print(f"  Max queries: {MAX_QUERIES}")
    print(f"\nInput: {trained_dir}")
    print(f"Output: {output_dir}")
    print("=" * 70)

    if not trained_dir.exists():
        print(f"\nERROR: Trained networks not found at {trained_dir}")
        print("Please run write_chn_large.py first.")
        sys.exit(1)

    run_sleep_all(trained_dir, output_dir, str(device))

    print("\n" + "=" * 70)
    print("SLEEP PHASE COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {output_dir}")
    print(f"Aggregated CSV: {output_dir / 'all_simulation_data.csv'}")
    print("=" * 70)
