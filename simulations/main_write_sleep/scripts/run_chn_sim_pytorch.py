# %% [markdown]
# # CHN Simulation using PyTorch GPU Backend
#
# This script mirrors run_chn_sim.py but uses PyTorch for GPU acceleration.
# Results are saved in C++ compatible format for use with existing analysis tools.

# %% Imports
import torch
import pandas as pd
from pathlib import Path
from itertools import product
from typing import Dict, List, Any
from dataclasses import dataclass, field
import json
import time
import sys

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Note: Install tqdm for progress bars: pip install tqdm")

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

from pytorch_chn import (
    ContinuousHopfieldNetwork,
    BatchedCHN,
    train_patterns_adam,
    run_sleep_phase,
    generate_patterns,
    get_device,
    check_cuda
)

# Base paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"

# %%=========================================================================
# CONFIGURATION SECTION
# ==========================================================================

# Repetitions (2 for testing, 20+ for final results)
NB_REPETITION = 2
REPETITIONS = list(range(NB_REPETITION))

# Network and pattern parameters
NETWORK_SIZES = list(range(100, 1001, 100))  # [100, 200, ..., 1000]
NUM_PATTERNS = list(range(10, 101, 5))       # [10, 15, ..., 100]
CORRELATIONS = [0.1, 0.25, 0.5, 0.75, 1.0]   # Pattern correlations
SPARSITY = 0.5                                # 50% active units

# Training parameters
DRIVE_TARGET = 6.0
LEARNING_RATE = 0.01      # Higher for Adam
MAX_ITER = 10000          # Fewer iterations needed with Adam
LEAK = 1.0
TOLERANCE = 0.1

# Sleep parameters
BETA = 0.1               # Inhibitory plasticity rate
DELTA = 0.01             # Integration timestep
NOISE_DYNAMICS = 0.01    # Noise std
MAX_QUERIES = 200        # Maximum retrieval attempts
STOP_ON_SPURIOUS = False
STOP_ON_ALL_FOUND = False

# Experiment names
WRITE_NAME = "comparison_chn_pytorch"
SLEEP_NAME = "comparison_chn_pytorch_sleep"


# %%=========================================================================
# UTILITY FUNCTIONS
# ==========================================================================

def save_parameters(params: Dict, filepath: Path):
    """Save parameters in C++ compatible format (key=value)."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        for key, value in params.items():
            f.write(f"{key}={value}\n")


def save_weights(W: torch.Tensor, filepath: Path):
    """Save weight matrix in C++ compatible format."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    W_np = W.cpu().numpy()
    with open(filepath, 'w') as f:
        for row in W_np:
            f.write(' '.join(f"{x:.10g}" for x in row) + '\n')


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


def generate_configs() -> List[Dict]:
    """Generate all parameter configurations."""
    configs = []
    sim_id = 0

    for net_size, num_pat, rho, seed in product(
        NETWORK_SIZES, NUM_PATTERNS, CORRELATIONS, REPETITIONS
    ):
        configs.append({
            'sim_id': sim_id,
            'network_size': net_size,
            'num_patterns': num_pat,
            'rho': rho,
            'seed': seed,
            'sparsity': SPARSITY,
            'leak': LEAK,
            'drive_target': DRIVE_TARGET,
        })
        sim_id += 1

    return configs


@dataclass
class TrainingStats:
    """Statistics collected during training phase."""
    total: int = 0
    converged: int = 0
    total_iterations: int = 0
    total_time: float = 0.0
    times: List[float] = field(default_factory=list)
    iterations: List[int] = field(default_factory=list)

    def update(self, converged: bool, iterations: int, time_s: float):
        self.total += 1
        if converged:
            self.converged += 1
        self.total_iterations += iterations
        self.total_time += time_s
        self.times.append(time_s)
        self.iterations.append(iterations)

    @property
    def convergence_rate(self) -> float:
        return self.converged / self.total if self.total > 0 else 0

    @property
    def avg_iterations(self) -> float:
        return self.total_iterations / self.total if self.total > 0 else 0

    @property
    def avg_time(self) -> float:
        return self.total_time / self.total if self.total > 0 else 0

    def summary(self) -> str:
        return (f"Converged: {self.converged}/{self.total} ({self.convergence_rate*100:.1f}%) | "
                f"Avg iters: {self.avg_iterations:.0f} | Avg time: {self.avg_time*1000:.1f}ms")


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

    @property
    def avg_time(self) -> float:
        return self.total_time / self.total if self.total > 0 else 0

    def summary(self) -> str:
        return (f"AR success: {self.all_recovered}/{self.total} ({self.ar_success_rate*100:.1f}%) | "
                f"Avg recovery: {self.avg_recovery_rate*100:.1f}% | "
                f"Avg spurious: {self.total_spurious/self.total:.1f} | "
                f"Avg time: {self.avg_time:.2f}s")


def get_gpu_memory_str() -> str:
    """Get GPU memory usage string."""
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1e6
        cached = torch.cuda.memory_reserved() / 1e6
        return f"GPU mem: {used:.0f}MB (cached: {cached:.0f}MB)"
    return ""


# %%=========================================================================
# TRAINING PHASE
# ==========================================================================

def run_training_phase(configs: List[Dict], output_dir: Path, device: str) -> TrainingStats:
    """Train networks for all configurations.

    Args:
        configs: List of configuration dicts
        output_dir: Directory to save trained networks
        device: 'cuda' or 'cpu'

    Returns:
        TrainingStats with collected statistics
    """
    stats = TrainingStats()
    start_time = time.time()

    # Create progress bar
    if HAS_TQDM:
        pbar = tqdm(configs, desc="Training", unit="net",
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}')
    else:
        pbar = configs
        print(f"\nTraining {len(configs)} networks...")

    for i, cfg in enumerate(pbar):
        t0 = time.time()

        sim_dir = output_dir / f"sim_nb_{cfg['sim_id']}"
        sim_dir.mkdir(parents=True, exist_ok=True)

        # Generate patterns
        patterns = generate_patterns(
            k=cfg['num_patterns'],
            n=cfg['network_size'],
            sparsity=cfg['sparsity'],
            rho=cfg['rho'],
            device=device,
            seed=cfg['seed']
        )

        # Initialize network
        network = ContinuousHopfieldNetwork(
            n_neurons=cfg['network_size'],
            leak=cfg['leak'],
            device=device
        )

        # Train with Adam
        W_trained, converged, history = train_patterns_adam(
            W=network.W,
            patterns=patterns,
            target_drive=cfg['drive_target'],
            learning_rate=LEARNING_RATE,
            max_iter=MAX_ITER,
            tolerance=TOLERANCE,
            leak=cfg['leak'],
            verbose=False
        )

        network.W = W_trained
        train_time = time.time() - t0

        # Update stats
        stats.update(converged, len(history), train_time)

        # Save outputs
        save_weights(network.W, sim_dir / "weights.data")
        save_patterns(patterns, sim_dir / "patterns.data")

        # Save parameters
        params = {
            **cfg,
            'converged': int(converged),
            'nb_winners': int(cfg['sparsity'] * cfg['network_size']),
            'train_iterations': len(history),
            'train_time': train_time
        }
        save_parameters(params, sim_dir / "parameters.data")

        # Update progress bar
        if HAS_TQDM:
            pbar.set_postfix_str(
                f"N={cfg['network_size']} K={cfg['num_patterns']} | "
                f"conv={stats.convergence_rate*100:.0f}% iters={len(history)} | "
                f"{get_gpu_memory_str()}"
            )
        elif (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (len(configs) - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{len(configs)}] {rate:.1f}/s, ETA: {eta/60:.1f}min | {stats.summary()}")

    if HAS_TQDM:
        pbar.close()

    # Print final summary
    elapsed = time.time() - start_time
    print(f"\n  Training complete in {elapsed/60:.1f} minutes")
    print(f"  {stats.summary()}")
    print(f"  {get_gpu_memory_str()}")

    return stats


# %%=========================================================================
# SLEEP PHASE
# ==========================================================================

def run_sleep_phase_all(
    trained_networks_dir: Path,
    output_dir: Path,
    device: str
) -> SleepStats:
    """Run sleep phase on all trained networks.

    Args:
        trained_networks_dir: Directory with trained networks
        output_dir: Directory to save sleep results
        device: 'cuda' or 'cpu'

    Returns:
        SleepStats with collected statistics
    """
    # List all simulation directories
    sim_dirs = sorted([d for d in trained_networks_dir.iterdir()
                       if d.is_dir() and d.name.startswith("sim_nb_")])

    stats = SleepStats()
    start_time = time.time()
    all_results = []

    # Create progress bar
    if HAS_TQDM:
        pbar = tqdm(sim_dirs, desc="Sleep", unit="net",
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}')
    else:
        pbar = sim_dirs
        print(f"\nRunning sleep phase on {len(sim_dirs)} networks...")

    for i, sim_dir in enumerate(pbar):
        t0 = time.time()

        sim_id = int(sim_dir.name.split("_")[-1])
        out_sim_dir = output_dir / f"sim_nb_{sim_id}"
        out_sim_dir.mkdir(parents=True, exist_ok=True)

        # Load weights
        W = load_weights(sim_dir / "weights.data", device)

        # Load patterns
        patterns = load_patterns(sim_dir / "patterns.data", device)

        # Load parameters
        params = load_parameters(sim_dir / "parameters.data")

        # Create network
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
            stop_on_spurious=STOP_ON_SPURIOUS,
            stop_on_all_found=STOP_ON_ALL_FOUND,
            verbose=False
        )

        sleep_time = time.time() - t0
        n_patterns = patterns.shape[0]

        # Update stats
        stats.update(
            n_found=len(results.found_patterns),
            n_patterns=n_patterns,
            n_spurious=results.n_spurious,
            all_before_spurious=results.all_recovered_before_spurious,
            time_s=sleep_time
        )

        # Convert to rows
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

        # Copy patterns and parameters
        save_patterns(patterns, out_sim_dir / "patterns.data")
        save_parameters(params, out_sim_dir / "parameters.data")

        # Update progress bar
        if HAS_TQDM:
            pbar.set_postfix_str(
                f"N={n_neurons} K={n_patterns} | "
                f"AR={stats.ar_success_rate*100:.0f}% "
                f"found={len(results.found_patterns)}/{n_patterns} | "
                f"{get_gpu_memory_str()}"
            )
        elif (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (len(sim_dirs) - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{len(sim_dirs)}] {rate:.2f}/s, ETA: {eta/60:.1f}min | {stats.summary()}")

    if HAS_TQDM:
        pbar.close()

    # Save aggregated results
    if all_results:
        save_results_csv(all_results, output_dir / "all_simulation_data.csv")

    # Print final summary
    elapsed = time.time() - start_time
    print(f"\n  Sleep phase complete in {elapsed/60:.1f} minutes")
    print(f"  {stats.summary()}")
    print(f"  {get_gpu_memory_str()}")

    return stats


def load_weights(filepath: Path, device: str) -> torch.Tensor:
    """Load weight matrix from C++ format."""
    rows = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                rows.append([float(x) for x in line.strip().split()])
    W = torch.tensor(rows, dtype=torch.float32, device=device)
    return W


def load_patterns(filepath: Path, device: str) -> torch.Tensor:
    """Load patterns from C++ format."""
    patterns = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                patterns.append([float(x == '1') for x in line.strip().split()])
    return torch.tensor(patterns, dtype=torch.float32, device=device)


def load_parameters(filepath: Path) -> Dict[str, float]:
    """Load parameters from C++ key=value format."""
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


# %%=========================================================================
# MAIN EXECUTION
# ==========================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("CHN SIMULATION - PyTorch GPU Backend")
    print("=" * 70)

    # Check CUDA
    device = get_device()
    print(f"\nUsing device: {device}")
    if device.type == "cuda":
        check_cuda()

    # Generate configurations
    configs = generate_configs()

    total_networks = len(configs)
    print(f"\nConfiguration:")
    print(f"  Repetitions: {NB_REPETITION}")
    print(f"  Network sizes: {NETWORK_SIZES[0]} to {NETWORK_SIZES[-1]} ({len(NETWORK_SIZES)} values)")
    print(f"  Pattern counts: {NUM_PATTERNS[0]} to {NUM_PATTERNS[-1]} ({len(NUM_PATTERNS)} values)")
    print(f"  Correlations: {CORRELATIONS}")
    print(f"  Sparsity: {SPARSITY}")
    print(f"  Total networks: {total_networks}")

    # Directories
    train_output_dir = DATA_DIR / "trained_networks" / WRITE_NAME
    sleep_output_dir = DATA_DIR / "sleep_results" / SLEEP_NAME

    # Phase 1: Training
    print("\n" + "=" * 70)
    print("PHASE 1: TRAINING")
    print("=" * 70)

    run_training_phase(configs, train_output_dir, str(device))

    # Phase 2: Sleep
    print("\n" + "=" * 70)
    print("PHASE 2: SLEEP (AUTONOMOUS RETRIEVAL)")
    print("=" * 70)

    run_sleep_phase_all(train_output_dir, sleep_output_dir, str(device))

    # Summary
    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE!")
    print("=" * 70)
    print(f"\nTrained networks: {train_output_dir}")
    print(f"Sleep results: {sleep_output_dir}")
    print(f"Aggregated CSV: {sleep_output_dir / 'all_simulation_data.csv'}")
    print("=" * 70)

# %%
