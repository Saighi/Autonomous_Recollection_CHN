#%%
"""
CHN Training (Write Phase) - PyTorch Backend for Small Networks

Trains Continuous Hopfield Networks using Adam optimizer on GPU/CPU.
PyTorch version of write_chn_small.py for comparison with C++ backend.

After running, use sleep_chn_small_pytorch.py for the sleep phase.
"""

#%% Imports
import torch
import time
from pathlib import Path
from itertools import product
from typing import Dict, List
from dataclasses import dataclass
import sys
import numpy as np

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

from pytorch_chn import (
    ContinuousHopfieldNetwork,
    train_patterns_adam,
    generate_patterns,
    get_device,
    check_cuda
)

PROJECT_DIR = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_DIR / "data"

#%% Configuration - Repetitions and Network Parameters
NB_REPETITION = 2
REPETITIONS = list(range(NB_REPETITION))

NETWORK_SIZES = np.linspace(25, 250, 20, dtype=int).tolist()
NUM_PATTERNS = list(range(1, 26))
CORRELATIONS = [0.9, 0.75, 0.5, 0.25, 0.0]  # Avoid 1.0 (causes infinite loop)
SPARSITY = 0.5

#%% Configuration - Training Parameters
DRIVE_TARGET = 6.0
LEARNING_RATE = 0.01  # Adam uses higher LR than momentum-based GD
MAX_ITER = 10000
TOLERANCE = 0.05
LEAK = 1.0

EXPERIMENT_NAME = "comparison_chn_small_pytorch"

#%% Utility Functions
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

    def update(self, converged: bool, iterations: int, time_s: float):
        self.total += 1
        if converged:
            self.converged += 1
        self.total_iterations += iterations
        self.total_time += time_s

    @property
    def convergence_rate(self) -> float:
        return self.converged / self.total if self.total > 0 else 0

    @property
    def avg_iterations(self) -> float:
        return self.total_iterations / self.total if self.total > 0 else 0

    def summary(self) -> str:
        return (f"Converged: {self.converged}/{self.total} ({self.convergence_rate*100:.1f}%) | "
                f"Avg iters: {self.avg_iterations:.0f}")


def get_gpu_memory_str() -> str:
    """Get GPU memory usage string."""
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1e6
        cached = torch.cuda.memory_reserved() / 1e6
        return f"GPU mem: {used:.0f}MB (cached: {cached:.0f}MB)"
    return ""

#%% Training Function
def run_training(configs: List[Dict], output_dir: Path, device: str) -> TrainingStats:
    """Train networks for all configurations."""
    stats = TrainingStats()
    start_time = time.time()

    if HAS_TQDM:
        pbar = tqdm(configs, desc="Training", unit="net")
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

        stats.update(converged, len(history), train_time)

        # Save outputs
        save_weights(network.W, sim_dir / "weights.data")
        save_patterns(patterns, sim_dir / "patterns.data")

        params = {
            **cfg,
            'converged': int(converged),
            'nb_winners': int(cfg['sparsity'] * cfg['network_size']),
            'train_iterations': len(history),
            'train_time': train_time
        }
        save_parameters(params, sim_dir / "parameters.data")

        if HAS_TQDM:
            pbar.set_postfix_str(
                f"N={cfg['network_size']} K={cfg['num_patterns']} | "
                f"conv={stats.convergence_rate*100:.0f}% | {get_gpu_memory_str()}"
            )
        elif (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(configs)}] {stats.summary()}")

    if HAS_TQDM:
        pbar.close()

    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed/60:.1f} minutes")
    print(f"{stats.summary()}")

    return stats

#%% Initialize Device
print("=" * 70)
print("CHN WRITE PHASE - PyTorch (Small Networks)")
print("=" * 70)

device = get_device()
print(f"\nDevice: {device}")
if device.type == "cuda":
    check_cuda()

#%% Generate Configurations
configs = generate_configs()
total_networks = len(configs)

print(f"\nConfiguration:")
print(f"  Repetitions: {NB_REPETITION}")
print(f"  Network sizes: {NETWORK_SIZES[0]} to {NETWORK_SIZES[-1]}")
print(f"  Pattern counts: {NUM_PATTERNS[0]} to {NUM_PATTERNS[-1]}")
print(f"  Correlations: {CORRELATIONS}")
print(f"  Sparsity: {SPARSITY}")
print(f"  Total networks: {total_networks}")

output_dir = DATA_DIR / "trained_networks" / EXPERIMENT_NAME
print(f"\nOutput: {output_dir}")
print("=" * 70)

#%% Run Training
run_training(configs, output_dir, str(device))

#%% Summary
print("\n" + "=" * 70)
print("WRITE PHASE COMPLETE")
print("=" * 70)
print(f"\nTrained networks saved to: {output_dir}")
print(f"Next: Run sleep_chn_small_pytorch.py")
print("=" * 70)
