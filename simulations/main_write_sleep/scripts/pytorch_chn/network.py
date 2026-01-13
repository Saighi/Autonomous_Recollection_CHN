"""
Continuous Hopfield Network implementation in PyTorch with CUDA support.

Uses standard sigmoid transfer function [0, 1] to match C++ implementation.
Supports both single-network and batched operations for GPU efficiency.
"""

import torch
from typing import Optional, Tuple


class ContinuousHopfieldNetwork:
    """Single Continuous Hopfield Network with standard sigmoid [0,1].

    Matches C++ implementation behavior:
    - Transfer: sigmoid(u) -> [0, 1]
    - Patterns: {0, 1}
    - Neutral state: v = 0.5

    Attributes:
        n_neurons: Number of neurons
        leak: Leak rate (default 1.0)
        delta: Integration timestep
        device: torch.device for computations
        W: Weight matrix (n x n), symmetric, zero diagonal
        inhib_diag: Diagonal inhibition vector (for sleep)
        u: Membrane potentials
        v: Firing rates [0, 1]
    """

    def __init__(
        self,
        n_neurons: int,
        leak: float = 1.0,
        delta: float = 0.01,
        device: str = "cuda"
    ):
        self.n_neurons = n_neurons
        self.leak = leak
        self.delta = delta
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Weight matrix (symmetric, zero diagonal)
        self.W = torch.zeros(n_neurons, n_neurons, device=self.device)

        # Diagonal inhibition for sleep phase
        self.inhib_diag = torch.zeros(n_neurons, device=self.device)

        # State variables
        self.u = torch.zeros(n_neurons, device=self.device)
        self.v = torch.full((n_neurons,), 0.5, device=self.device)  # Neutral state

    def transfer(self, u: torch.Tensor) -> torch.Tensor:
        """Standard sigmoid transfer function.

        Returns:
            Firing rates in range [0, 1]
        """
        return torch.sigmoid(u)

    def transfer_inverse(self, v: torch.Tensor) -> torch.Tensor:
        """Inverse of sigmoid (logit function).

        Args:
            v: Firing rates in (0, 1)

        Returns:
            Membrane potentials
        """
        v_clamped = torch.clamp(v, 1e-7, 1 - 1e-7)
        return torch.log(v_clamped / (1 - v_clamped))

    def step(self, noise_stddev: float = 0.0) -> torch.Tensor:
        """Single Euler integration step (basic dynamics).

        Args:
            noise_stddev: Standard deviation of Gaussian noise

        Returns:
            Updated firing rates v
        """
        # Compute drive from weights
        drive = self.W @ self.v

        # Add noise if specified
        if noise_stddev > 0:
            drive = drive + noise_stddev * torch.randn_like(drive)

        # du/dt = W @ v - leak * u
        du_dt = drive - self.leak * self.u
        self.u = self.u + self.delta * du_dt
        self.v = self.transfer(self.u)

        return self.v

    def depressed_step(self, noise_stddev: float = 0.0) -> torch.Tensor:
        """Single Euler step with diagonal inhibition (sleep dynamics).

        Args:
            noise_stddev: Standard deviation of Gaussian noise

        Returns:
            Updated firing rates v
        """
        # Compute drive from weights
        drive = self.W @ self.v

        # Subtract diagonal inhibition
        drive = drive - self.inhib_diag * self.v

        # Add noise if specified
        if noise_stddev > 0:
            drive = drive + noise_stddev * torch.randn_like(drive)

        # du/dt = drive - leak * u
        du_dt = drive - self.leak * self.u
        self.u = self.u + self.delta * du_dt
        self.v = self.transfer(self.u)

        return self.v

    def run(
        self,
        n_steps: int,
        noise_stddev: float = 0.0,
        use_inhibition: bool = False,
        return_trajectory: bool = False
    ) -> torch.Tensor:
        """Run dynamics for multiple steps.

        Args:
            n_steps: Number of integration steps
            noise_stddev: Noise standard deviation
            use_inhibition: If True, use depressed dynamics
            return_trajectory: If True, return full trajectory

        Returns:
            If return_trajectory: Tensor of shape (n_steps+1, n_neurons)
            Otherwise: Final firing rates v
        """
        step_fn = self.depressed_step if use_inhibition else self.step

        if return_trajectory:
            trajectory = [self.v.clone()]
            for _ in range(n_steps):
                step_fn(noise_stddev)
                trajectory.append(self.v.clone())
            return torch.stack(trajectory)
        else:
            for _ in range(n_steps):
                step_fn(noise_stddev)
            return self.v

    def reset_to_neutral(self):
        """Reset to neutral state (v=0.5, u=0)."""
        self.u = torch.zeros(self.n_neurons, device=self.device)
        self.v = torch.full((self.n_neurons,), 0.5, device=self.device)

    def set_state_from_pattern(self, pattern: torch.Tensor, drive_magnitude: float = 6.0):
        """Set state from binary pattern {0, 1}.

        Args:
            pattern: Binary pattern in {0, 1}
            drive_magnitude: Target drive (default 6.0)
        """
        # Convert {0, 1} to target drives: 0 -> -drive, 1 -> +drive
        target_drives = (2 * pattern.float() - 1) * drive_magnitude
        self.u = target_drives.to(self.device)
        self.v = self.transfer(self.u)

    def set_weights(self, W: torch.Tensor):
        """Set weight matrix (enforces symmetry and zero diagonal)."""
        self.W = W.clone().to(self.device)
        self.W = (self.W + self.W.T) / 2
        self.W.fill_diagonal_(0)

    def get_weights(self) -> torch.Tensor:
        """Get weight matrix."""
        return self.W.clone()

    def pot_inhib_diag(self, beta: float):
        """Potentiate diagonal inhibition after pattern retrieval.

        This strengthens self-inhibition proportional to current activity,
        which helps the network escape the current attractor during sleep.

        Args:
            beta: Potentiation rate
        """
        # Increase inhibition for active neurons
        self.inhib_diag = self.inhib_diag + beta * self.v

    def reset_inhibition(self):
        """Reset diagonal inhibition to zero."""
        self.inhib_diag = torch.zeros(self.n_neurons, device=self.device)

    def compute_energy(self) -> torch.Tensor:
        """Compute network energy: E = -0.5 * v^T W v"""
        return -0.5 * torch.einsum('i,ij,j->', self.v, self.W, self.v)

    def to(self, device: str) -> "ContinuousHopfieldNetwork":
        """Move network to specified device."""
        self.device = torch.device(device)
        self.W = self.W.to(self.device)
        self.u = self.u.to(self.device)
        self.v = self.v.to(self.device)
        self.inhib_diag = self.inhib_diag.to(self.device)
        return self


class BatchedCHN:
    """Batched CHN for running B networks of same size in parallel on GPU.

    This enables efficient parameter sweeps by processing multiple networks
    simultaneously with batched matrix operations.

    Attributes:
        batch_size: Number of networks (B)
        n_neurons: Network size (N)
        leak: Leak rate
        delta: Integration timestep
        device: torch.device
        W: Weight matrices [B, N, N]
        inhib_diag: Diagonal inhibition vectors [B, N]
        u: Membrane potentials [B, N]
        v: Firing rates [B, N]
    """

    def __init__(
        self,
        batch_size: int,
        n_neurons: int,
        leak: float = 1.0,
        delta: float = 0.01,
        device: str = "cuda"
    ):
        self.batch_size = batch_size
        self.n_neurons = n_neurons
        self.leak = leak
        self.delta = delta
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Batched weight matrices [B, N, N]
        self.W = torch.zeros(batch_size, n_neurons, n_neurons, device=self.device)

        # Batched diagonal inhibition [B, N]
        self.inhib_diag = torch.zeros(batch_size, n_neurons, device=self.device)

        # Batched state [B, N]
        self.u = torch.zeros(batch_size, n_neurons, device=self.device)
        self.v = torch.full((batch_size, n_neurons), 0.5, device=self.device)

    def transfer(self, u: torch.Tensor) -> torch.Tensor:
        """Standard sigmoid transfer function."""
        return torch.sigmoid(u)

    def step(self, noise_stddev: float = 0.0) -> torch.Tensor:
        """Single batched Euler step (basic dynamics).

        All B networks update simultaneously.
        """
        # Batched matrix-vector: drive[b] = W[b] @ v[b]
        drive = torch.bmm(self.W, self.v.unsqueeze(-1)).squeeze(-1)

        # Add noise
        if noise_stddev > 0:
            drive = drive + noise_stddev * torch.randn_like(drive)

        # Euler update
        du_dt = drive - self.leak * self.u
        self.u = self.u + self.delta * du_dt
        self.v = self.transfer(self.u)

        return self.v

    def depressed_step(self, noise_stddev: float = 0.0) -> torch.Tensor:
        """Single batched Euler step with diagonal inhibition."""
        # Batched matrix-vector
        drive = torch.bmm(self.W, self.v.unsqueeze(-1)).squeeze(-1)

        # Subtract diagonal inhibition
        drive = drive - self.inhib_diag * self.v

        # Add noise
        if noise_stddev > 0:
            drive = drive + noise_stddev * torch.randn_like(drive)

        # Euler update
        du_dt = drive - self.leak * self.u
        self.u = self.u + self.delta * du_dt
        self.v = self.transfer(self.u)

        return self.v

    def run(
        self,
        n_steps: int,
        noise_stddev: float = 0.0,
        use_inhibition: bool = False
    ) -> torch.Tensor:
        """Run dynamics for multiple steps on all networks."""
        step_fn = self.depressed_step if use_inhibition else self.step

        for _ in range(n_steps):
            step_fn(noise_stddev)

        return self.v

    def reset_to_neutral(self, mask: Optional[torch.Tensor] = None):
        """Reset networks to neutral state.

        Args:
            mask: Optional boolean mask [B] to reset only specific networks
        """
        if mask is None:
            self.u = torch.zeros(self.batch_size, self.n_neurons, device=self.device)
            self.v = torch.full((self.batch_size, self.n_neurons), 0.5, device=self.device)
        else:
            self.u[mask] = 0
            self.v[mask] = 0.5

    def pot_inhib_diag(self, beta: float, mask: Optional[torch.Tensor] = None):
        """Potentiate diagonal inhibition.

        Args:
            beta: Potentiation rate
            mask: Optional boolean mask [B] to update only specific networks
        """
        if mask is None:
            self.inhib_diag = self.inhib_diag + beta * self.v
        else:
            self.inhib_diag[mask] = self.inhib_diag[mask] + beta * self.v[mask]

    def reset_inhibition(self):
        """Reset all diagonal inhibition to zero."""
        self.inhib_diag = torch.zeros(self.batch_size, self.n_neurons, device=self.device)

    def set_weights(self, W: torch.Tensor, idx: Optional[int] = None):
        """Set weight matrices.

        Args:
            W: Weight matrix/matrices
               If idx is None: W should be [B, N, N] for all networks
               If idx is int: W should be [N, N] for network idx
            idx: Optional index of specific network to set
        """
        if idx is None:
            self.W = W.clone().to(self.device)
        else:
            self.W[idx] = W.clone().to(self.device)

        # Enforce symmetry and zero diagonal
        self.W = (self.W + self.W.transpose(-2, -1)) / 2
        # Zero diagonal for all networks
        diag_mask = torch.eye(self.n_neurons, device=self.device, dtype=torch.bool)
        self.W[:, diag_mask] = 0

    def get_weights(self, idx: Optional[int] = None) -> torch.Tensor:
        """Get weight matrices."""
        if idx is None:
            return self.W.clone()
        return self.W[idx].clone()

    def check_convergence(self, tolerance: float = 1e-4) -> torch.Tensor:
        """Check which networks have converged.

        Returns:
            Boolean tensor [B] indicating converged networks
        """
        # Networks are converged if du/dt is small
        drive = torch.bmm(self.W, self.v.unsqueeze(-1)).squeeze(-1)
        drive = drive - self.inhib_diag * self.v
        du_dt = drive - self.leak * self.u
        max_change = du_dt.abs().max(dim=1).values
        return max_change < tolerance

    def to(self, device: str) -> "BatchedCHN":
        """Move all tensors to specified device."""
        self.device = torch.device(device)
        self.W = self.W.to(self.device)
        self.u = self.u.to(self.device)
        self.v = self.v.to(self.device)
        self.inhib_diag = self.inhib_diag.to(self.device)
        return self
