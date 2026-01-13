"""
PyTorch GPU-accelerated Continuous Hopfield Network module.

Provides batched network simulation for efficient parameter sweeps on GPU.
"""

import torch

from .network import ContinuousHopfieldNetwork, BatchedCHN
from .learning import train_patterns_adam, train_patterns_sgd, train_batch_networks
from .sleep import run_sleep_phase, run_sleep_phase_batched
from .patterns import generate_patterns, generate_patterns_batch


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Get the best available device.

    Args:
        prefer_cuda: If True, use CUDA if available

    Returns:
        torch.device for computations
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def check_cuda() -> bool:
    """Check if CUDA is available and print device info."""
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return True
    else:
        print("CUDA not available, using CPU")
        return False


__all__ = [
    "ContinuousHopfieldNetwork",
    "BatchedCHN",
    "train_patterns_adam",
    "train_patterns_sgd",
    "train_batch_networks",
    "run_sleep_phase",
    "run_sleep_phase_batched",
    "generate_patterns",
    "generate_patterns_batch",
    "get_device",
    "check_cuda",
]
