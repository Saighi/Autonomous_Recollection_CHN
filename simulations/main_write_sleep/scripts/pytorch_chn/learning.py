"""
Training algorithms for Continuous Hopfield Networks.

Supports:
- Adam optimizer (fast convergence)
- SGD with momentum (matches C++ implementation)
- Batched training for multiple networks on GPU
"""

import torch
from typing import Tuple, List, Optional


def compute_drive_error(
    W: torch.Tensor,
    patterns: torch.Tensor,
    target_drive: float,
    leak: float = 1.0
) -> torch.Tensor:
    """Compute drive error for all patterns.

    At equilibrium (du/dt = 0): u_eq = (W @ v) / leak
    Target drive for pattern p: target_drive for active (1), -target_drive for inactive (0)

    Args:
        W: Weight matrix [N, N]
        patterns: Binary patterns [K, N] with values in {0, 1}
        target_drive: Target drive magnitude
        leak: Leak rate

    Returns:
        Error tensor [K, N]
    """
    # Convert {0,1} patterns to target drives: 0 -> -target_drive, 1 -> +target_drive
    target_drives = (2 * patterns - 1) * target_drive  # [K, N]

    # Compute target rates from target drives
    target_rates = torch.sigmoid(target_drives)  # [K, N]

    # Predicted equilibrium drives: u_hat = (W @ v) / leak
    # For K patterns: [K, N] @ [N, N] -> [K, N]
    u_hat = (target_rates @ W) / leak

    # Error = target - predicted
    errors = target_drives - u_hat  # [K, N]

    return errors


def gda_step_batched(
    W: torch.Tensor,
    patterns: torch.Tensor,
    target_drive: float,
    learning_rate: float,
    leak: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Single GDA step processing all patterns at once.

    Computes gradients for ALL K patterns simultaneously and averages.

    Args:
        W: Weight matrix [N, N]
        patterns: Binary patterns [K, N]
        target_drive: Target drive magnitude
        learning_rate: Learning rate
        leak: Leak rate

    Returns:
        Tuple of (W_new, max_error)
    """
    K = patterns.shape[0]

    # Convert {0,1} to target drives
    target_drives = (2 * patterns - 1) * target_drive  # [K, N]
    target_rates = torch.sigmoid(target_drives)  # [K, N]

    # Predicted equilibrium
    u_hat = (target_rates @ W) / leak  # [K, N]

    # Errors
    errors = target_drives - u_hat  # [K, N]

    # Average gradient over all patterns
    # delta_W = mean_k[ outer(error_k, target_rate_k) ]
    # = (errors.T @ target_rates) / K
    delta_W = (errors.T @ target_rates) / K  # [N, N]

    # Symmetrize
    delta_W = (delta_W + delta_W.T) / 2

    # Update weights
    W_new = W + learning_rate * delta_W

    # Zero diagonal
    W_new.fill_diagonal_(0)

    max_error = errors.abs().max().item()

    return W_new, max_error


def train_patterns_adam(
    W: torch.Tensor,
    patterns: torch.Tensor,
    target_drive: float = 6.0,
    learning_rate: float = 0.01,
    betas: Tuple[float, float] = (0.9, 0.999),
    max_iter: int = 10000,
    tolerance: float = 0.1,
    leak: float = 1.0,
    verbose: bool = False,
    log_every: int = 1000
) -> Tuple[torch.Tensor, bool, List[float]]:
    """Train network using Adam optimizer.

    Faster convergence than vanilla GDA - typically 5,000-10,000 iterations
    instead of 100,000.

    Args:
        W: Initial weight matrix [N, N]
        patterns: Binary patterns [K, N] with values in {0, 1}
        target_drive: Target drive magnitude
        learning_rate: Adam learning rate
        betas: Adam momentum coefficients
        max_iter: Maximum iterations
        tolerance: Convergence threshold for max error
        leak: Leak rate
        verbose: Print progress
        log_every: Print every N iterations

    Returns:
        Tuple of (W_trained, converged, error_history)
    """
    # Make W a parameter for Adam
    W = W.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([W], lr=learning_rate, betas=betas)

    # Precompute target drives and rates
    target_drives = (2 * patterns - 1) * target_drive  # [K, N]
    K = patterns.shape[0]

    error_history = []
    converged = False

    for i in range(max_iter):
        optimizer.zero_grad()

        # Compute target rates (no grad needed)
        with torch.no_grad():
            target_rates = torch.sigmoid(target_drives)  # [K, N]

        # Predicted equilibrium
        u_hat = (target_rates @ W) / leak  # [K, N]

        # Loss = mean squared drive error
        loss = ((target_drives - u_hat) ** 2).mean()

        # Backward pass
        loss.backward()

        # Step
        optimizer.step()

        # Enforce constraints
        with torch.no_grad():
            # Symmetrize
            W.data = (W.data + W.data.T) / 2
            # Zero diagonal
            W.data.fill_diagonal_(0)

        # Track error
        max_error = (target_drives - u_hat).abs().max().item()
        error_history.append(max_error)

        if verbose and (i + 1) % log_every == 0:
            print(f"  Adam iter {i+1}: loss={loss.item():.6f}, max_error={max_error:.4f}")

        if max_error < tolerance:
            converged = True
            if verbose:
                print(f"  Adam converged at iteration {i+1}")
            break

    return W.detach(), converged, error_history


def train_patterns_sgd(
    W: torch.Tensor,
    patterns: torch.Tensor,
    target_drive: float = 6.0,
    learning_rate: float = 0.001,
    momentum: float = 0.9,
    max_iter: int = 100000,
    tolerance: float = 0.1,
    leak: float = 1.0,
    verbose: bool = False,
    log_every: int = 10000
) -> Tuple[torch.Tensor, bool, List[float]]:
    """Train network using SGD with momentum (matches C++ behavior).

    Args:
        W: Initial weight matrix [N, N]
        patterns: Binary patterns [K, N]
        target_drive: Target drive magnitude
        learning_rate: SGD learning rate
        momentum: Momentum coefficient
        max_iter: Maximum iterations
        tolerance: Convergence threshold
        leak: Leak rate
        verbose: Print progress
        log_every: Print every N iterations

    Returns:
        Tuple of (W_trained, converged, error_history)
    """
    W = W.clone()
    velocity = torch.zeros_like(W)
    K = patterns.shape[0]

    # Precompute target drives
    target_drives = (2 * patterns - 1) * target_drive  # [K, N]

    error_history = []
    converged = False

    for i in range(max_iter):
        # Target rates
        target_rates = torch.sigmoid(target_drives)  # [K, N]

        # Predicted equilibrium
        u_hat = (target_rates @ W) / leak  # [K, N]

        # Errors
        errors = target_drives - u_hat  # [K, N]

        # Average gradient
        delta_W = (errors.T @ target_rates) / K

        # Symmetrize
        delta_W = (delta_W + delta_W.T) / 2

        # Momentum update
        velocity = momentum * velocity + delta_W
        W = W + learning_rate * velocity

        # Zero diagonal
        W.fill_diagonal_(0)

        # Track error
        max_error = errors.abs().max().item()
        error_history.append(max_error)

        if verbose and (i + 1) % log_every == 0:
            print(f"  SGD iter {i+1}: max_error={max_error:.4f}")

        if max_error < tolerance:
            converged = True
            if verbose:
                print(f"  SGD converged at iteration {i+1}")
            break

    return W, converged, error_history


def train_batch_networks(
    W_batch: torch.Tensor,
    patterns_list: List[torch.Tensor],
    target_drive: float = 6.0,
    learning_rate: float = 0.01,
    max_iter: int = 10000,
    tolerance: float = 0.1,
    leak: float = 1.0,
    use_adam: bool = True,
    verbose: bool = False
) -> Tuple[torch.Tensor, List[bool], List[List[float]]]:
    """Train multiple networks in parallel on GPU.

    Each network can have different patterns but must have the same size.

    Args:
        W_batch: Batched weight matrices [B, N, N]
        patterns_list: List of B pattern tensors, each [K_b, N]
        target_drive: Target drive magnitude
        learning_rate: Learning rate
        max_iter: Maximum iterations
        tolerance: Convergence threshold
        leak: Leak rate
        use_adam: Use Adam if True, SGD+momentum if False
        verbose: Print progress

    Returns:
        Tuple of (W_trained [B, N, N], converged_list, error_histories)
    """
    B = W_batch.shape[0]
    device = W_batch.device

    # Train each network
    # For now, train sequentially (networks may have different pattern counts)
    # Future optimization: batch networks with same pattern count

    all_converged = []
    all_histories = []

    train_fn = train_patterns_adam if use_adam else train_patterns_sgd

    for b in range(B):
        W_b = W_batch[b]
        patterns_b = patterns_list[b].to(device)

        if verbose:
            print(f"Training network {b+1}/{B}...")

        W_trained, converged, history = train_fn(
            W_b, patterns_b,
            target_drive=target_drive,
            learning_rate=learning_rate,
            max_iter=max_iter,
            tolerance=tolerance,
            leak=leak,
            verbose=verbose
        )

        W_batch[b] = W_trained
        all_converged.append(converged)
        all_histories.append(history)

    return W_batch, all_converged, all_histories


def train_batch_networks_parallel(
    W_batch: torch.Tensor,
    patterns_batch: torch.Tensor,
    target_drive: float = 6.0,
    learning_rate: float = 0.01,
    max_iter: int = 10000,
    tolerance: float = 0.1,
    leak: float = 1.0,
    verbose: bool = False,
    log_every: int = 1000
) -> Tuple[torch.Tensor, torch.Tensor, List[float]]:
    """Train multiple networks truly in parallel (same pattern count required).

    All B networks must have the same number of patterns K.

    Args:
        W_batch: Weight matrices [B, N, N]
        patterns_batch: Patterns [B, K, N]
        target_drive: Target drive magnitude
        learning_rate: Learning rate
        max_iter: Maximum iterations
        tolerance: Convergence threshold
        leak: Leak rate
        verbose: Print progress
        log_every: Print every N iterations

    Returns:
        Tuple of (W_trained [B, N, N], converged [B], max_error_history)
    """
    B, N, _ = W_batch.shape
    K = patterns_batch.shape[1]

    # Make W a parameter
    W = W_batch.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([W], lr=learning_rate)

    # Precompute target drives [B, K, N]
    target_drives = (2 * patterns_batch - 1) * target_drive

    error_history = []
    converged = torch.zeros(B, dtype=torch.bool, device=W.device)

    for i in range(max_iter):
        optimizer.zero_grad()

        # Target rates [B, K, N]
        with torch.no_grad():
            target_rates = torch.sigmoid(target_drives)

        # Predicted equilibrium: u_hat[b, k] = (target_rates[b, k] @ W[b]) / leak
        # [B, K, N] @ [B, N, N] -> need bmm with broadcasting
        # Reshape: target_rates [B, K, N] -> [B*K, N]
        # W [B, N, N] -> expand to [B*K, N, N] (repeat K times per batch)
        target_rates_flat = target_rates.reshape(B * K, N)  # [B*K, N]

        # Expand W: [B, N, N] -> [B*K, N, N]
        W_expanded = W.unsqueeze(1).expand(B, K, N, N).reshape(B * K, N, N)

        # Batched matmul: [B*K, N] @ [B*K, N, N] -> [B*K, N]
        u_hat_flat = torch.bmm(target_rates_flat.unsqueeze(1), W_expanded).squeeze(1) / leak

        # Reshape back: [B*K, N] -> [B, K, N]
        u_hat = u_hat_flat.reshape(B, K, N)

        # Loss
        loss = ((target_drives - u_hat) ** 2).mean()

        loss.backward()
        optimizer.step()

        # Enforce constraints
        with torch.no_grad():
            W.data = (W.data + W.data.transpose(-2, -1)) / 2
            diag_mask = torch.eye(N, device=W.device, dtype=torch.bool)
            W.data[:, diag_mask] = 0

        # Track error per network
        with torch.no_grad():
            errors = (target_drives - u_hat).abs()  # [B, K, N]
            max_errors = errors.max(dim=2).values.max(dim=1).values  # [B]
            overall_max = max_errors.max().item()
            error_history.append(overall_max)

            # Check convergence per network
            converged = max_errors < tolerance

        if verbose and (i + 1) % log_every == 0:
            n_converged = converged.sum().item()
            print(f"  Iter {i+1}: max_error={overall_max:.4f}, converged={n_converged}/{B}")

        if converged.all():
            if verbose:
                print(f"  All networks converged at iteration {i+1}")
            break

    return W.detach(), converged, error_history
