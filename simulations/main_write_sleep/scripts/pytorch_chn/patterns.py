"""
Pattern generation utilities for CHN simulations.

Generates sparse correlated patterns compatible with C++ implementation.
Uses {0, 1} binary encoding (not bipolar).
"""

import torch
import numpy as np
from typing import Tuple, List, Optional


def generate_patterns(
    k: int,
    n: int,
    sparsity: float = 0.5,
    rho: float = 0.5,
    device: str = "cuda",
    seed: Optional[int] = None
) -> torch.Tensor:
    """Generate k correlated sparse patterns.

    Uses the balanced-flip algorithm matching utils.py generate_patterns_old().

    Args:
        k: Number of patterns
        n: Network size (neurons)
        sparsity: Fraction of active units (1s)
        rho: Pattern correlation (1=identical, 0=maximally different)
        device: Device for output tensor
        seed: Optional random seed

    Returns:
        Patterns tensor [k, n] with values in {0, 1}, dtype=float32
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

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
        pattern = base.copy()

        # Perform balanced flips
        for _ in range(num_flips):
            ones = np.where(pattern)[0]
            zeros = np.where(~pattern)[0]
            if len(ones) > 0 and len(zeros) > 0:
                # Flip one 1 to 0
                pattern[np.random.choice(ones)] = False
                # Flip one 0 to 1
                pattern[np.random.choice(zeros)] = True

        # Check uniqueness
        key = tuple(pattern.tolist())
        if key not in seen:
            seen.add(key)
            patterns.append(pattern)

    # Convert to torch tensor
    patterns_np = np.array(patterns, dtype=np.float32)
    patterns_torch = torch.from_numpy(patterns_np)

    return patterns_torch.to(device)


def generate_patterns_batch(
    configs: List[dict],
    device: str = "cuda"
) -> List[torch.Tensor]:
    """Generate patterns for multiple configurations.

    Args:
        configs: List of dicts with keys: k, n, sparsity, rho, seed
        device: Device for tensors

    Returns:
        List of pattern tensors
    """
    patterns_list = []

    for cfg in configs:
        patterns = generate_patterns(
            k=cfg['k'],
            n=cfg['n'],
            sparsity=cfg.get('sparsity', 0.5),
            rho=cfg.get('rho', 0.5),
            device=device,
            seed=cfg.get('seed', None)
        )
        patterns_list.append(patterns)

    return patterns_list


def generate_patterns_heterogeneous(
    k: int,
    n: int,
    mean_sparsity: float = 0.5,
    sparsity_width: float = 0.2,
    rho: float = 0.5,
    device: str = "cuda",
    seed: Optional[int] = None
) -> Tuple[torch.Tensor, List[dict]]:
    """Generate patterns with heterogeneous sparsities.

    Each pattern has a different sparsity sampled from a uniform distribution.

    Args:
        k: Number of patterns
        n: Network size
        mean_sparsity: Center of sparsity distribution
        sparsity_width: Full width of uniform distribution
        rho: Pattern correlation
        device: Device for tensor
        seed: Random seed

    Returns:
        Tuple of (patterns [k, n], per_pattern_metadata)
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    half_width = sparsity_width / 2.0

    # Compute core positions for correlation
    n_core = int(rho * n)
    n_variable = n - n_core

    # Generate core pattern at mean sparsity
    core_pattern = np.random.rand(n_core) > mean_sparsity if n_core > 0 else np.array([], dtype=bool)

    patterns = []
    metadata = []
    seen = set()

    while len(patterns) < k:
        # Sample sparsity for this pattern
        s_i = mean_sparsity + np.random.uniform(-half_width, half_width)
        s_i = float(np.clip(s_i, 0.01, 0.99))

        # Generate variable portion
        variable_part = np.random.rand(n_variable) > s_i if n_variable > 0 else np.array([], dtype=bool)

        # Combine
        pattern = np.concatenate([core_pattern, variable_part])

        # Shuffle
        np.random.shuffle(pattern)

        key = tuple(pattern.tolist())
        if key not in seen:
            seen.add(key)
            patterns.append(pattern)

            # Compute actual sparsity (fraction of zeros, P(0) convention)
            nb_active = int(pattern.sum())
            actual_sparsity = 1.0 - (nb_active / n)

            metadata.append({
                'index': len(patterns) - 1,
                'sparsity': actual_sparsity,
                'nb_active': nb_active
            })

    patterns_np = np.array(patterns, dtype=np.float32)
    patterns_torch = torch.from_numpy(patterns_np).to(device)

    return patterns_torch, metadata


def patterns_to_numpy(patterns: torch.Tensor) -> np.ndarray:
    """Convert patterns tensor to numpy array."""
    return patterns.cpu().numpy().astype(bool)


def patterns_from_numpy(patterns: np.ndarray, device: str = "cuda") -> torch.Tensor:
    """Convert numpy patterns to torch tensor."""
    return torch.from_numpy(patterns.astype(np.float32)).to(device)


def save_patterns_cpp_format(patterns: torch.Tensor, filepath: str):
    """Save patterns in C++ compatible format (space-separated 0/1)."""
    patterns_np = patterns.cpu().numpy()

    with open(filepath, 'w') as f:
        for pattern in patterns_np:
            f.write(' '.join(str(int(x)) for x in pattern) + '\n')


def load_patterns_cpp_format(filepath: str, device: str = "cuda") -> torch.Tensor:
    """Load patterns from C++ format file."""
    patterns = []

    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                pattern = [float(x == '1') for x in line.strip().split()]
                patterns.append(pattern)

    patterns_np = np.array(patterns, dtype=np.float32)
    return torch.from_numpy(patterns_np).to(device)
