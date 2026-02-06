# %% [markdown]
# # McCallum 1995 Original Protocol Reproduction
#
# This notebook reproduces McCallum's original pseudorehearsal experiment:
#
# **Protocol:**
# - Network size: N = 100
# - Patterns: up to 100, stored incrementally
# - Before each new pattern (M > 1):
#   - Probe network 300 times with random states
#   - Collect up to 256 unique pseudoitems (stable states)
#   - Train on pseudoitems + new pattern with delta learning
# - After each incorporation, count how many stored patterns are **stable**
#   - Stable = relaxing FROM the pattern itself returns the same pattern
#   - This is a WEAK criterion (just fixed-point check, not basin of attraction)
#
# **Output:** Plot of "Number of stable patterns" vs "Patterns learned"

# %% [markdown]
# ## Imports and Setup

# %%
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Set
from dataclasses import dataclass
import time

# Plotting style
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'figure.figsize': (10, 6),
    'figure.dpi': 100,
})

# %% [markdown]
# ## McCallum DHN Implementation

# %%
@dataclass
class McCallumDHN:
    """Discrete Hopfield Network with McCallum's delta learning."""

    size: int
    weights: np.ndarray = None

    # McCallum parameters
    eta: float = 0.1           # Learning rate
    max_epochs: float = 500    # Max training epochs
    error_criterion: float = 0.001
    nu_h: float = 0.05         # 5% heteroassociative noise
    sigma_input: float = 0.5   # Gaussian input noise std

    def __post_init__(self):
        if self.weights is None:
            self.weights = np.zeros((self.size, self.size))

    def reset(self):
        """Reset weights to zero."""
        self.weights = np.zeros((self.size, self.size))

    def sign(self, x: np.ndarray) -> np.ndarray:
        """Sign function: +1 if >= 0, -1 otherwise."""
        return np.where(x >= 0, 1.0, -1.0)

    def relax_async(self, state: np.ndarray, max_cycles: int = None) -> np.ndarray:
        """
        Asynchronous relaxation until convergence or max cycles.
        One cycle = N random unit updates.
        """
        if max_cycles is None:
            max_cycles = 4 * self.size

        state = state.copy()
        N = self.size

        for cycle in range(max_cycles):
            changed = False
            order = np.random.permutation(N)

            for i in order:
                # Local field excluding self-connection
                h_i = np.dot(self.weights[i], state) - self.weights[i, i] * state[i]
                new_val = 1.0 if h_i >= 0 else -1.0

                if new_val != state[i]:
                    state[i] = new_val
                    changed = True

            if not changed:
                break

        return state

    def is_stable(self, pattern: np.ndarray) -> bool:
        """
        Check if pattern is a stable state (fixed point).
        Relaxing from the pattern returns the same pattern.
        """
        relaxed = self.relax_async(pattern.copy(), max_cycles=10)
        return np.array_equal(relaxed, pattern) or np.array_equal(relaxed, -pattern)

    def probe_for_pseudoitems(self, n_probes: int = 300, max_items: int = 256) -> List[np.ndarray]:
        """
        Probe network to find unique stable states (pseudoitems).
        """
        pseudoitems = []
        seen = set()

        for _ in range(n_probes):
            if len(pseudoitems) >= max_items:
                break

            # Random probe
            state = np.random.choice([-1.0, 1.0], size=self.size)

            # Relax to stable state
            stable = self.relax_async(state)

            # Check uniqueness (and inverse)
            key = tuple(stable.astype(int))
            inv_key = tuple((-stable).astype(int))

            if key not in seen and inv_key not in seen:
                seen.add(key)
                pseudoitems.append(stable)

        return pseudoitems

    def apply_heteroassociative_noise(self, pattern: np.ndarray) -> np.ndarray:
        """Flip nu_h fraction of bits."""
        noisy = pattern.copy()
        n_flip = int(round(self.nu_h * self.size))
        flip_idx = np.random.choice(self.size, size=n_flip, replace=False)
        noisy[flip_idx] *= -1
        return noisy

    def train_delta(self, training_set: List[np.ndarray], new_pattern_idx: int):
        """
        Train with delta learning rule.
        Apply noise only to new pattern, not pseudoitems.
        """
        smoothed_error = 1.0

        for epoch in range(int(self.max_epochs)):
            order = np.random.permutation(len(training_set))
            epoch_errors = 0.0

            for idx in order:
                target = training_set[idx]
                is_new = (idx == new_pattern_idx)

                # Prepare input
                if is_new:
                    inp = self.apply_heteroassociative_noise(target)
                else:
                    inp = target.copy()

                # Update each unit
                for i in range(self.size):
                    # Local field
                    h_i = np.dot(self.weights[i], inp) - self.weights[i, i] * inp[i]

                    # Add input noise for new patterns
                    if is_new:
                        h_i += np.random.normal(0, self.sigma_input)

                    # Output
                    psi_i = 1.0 if h_i >= 0 else -1.0

                    # Error and update
                    error_i = target[i] - psi_i
                    if abs(error_i) > 0.5:  # error is ±2 or 0
                        self.weights[i] += self.eta * error_i * inp
                        self.weights[i, i] = 0  # No self-connection
                        epoch_errors += abs(error_i) / 2

            # Early stopping
            smoothed_error = smoothed_error * 0.9 + epoch_errors * 0.1
            if smoothed_error < self.error_criterion:
                return epoch + 1

        return int(self.max_epochs)

# %% [markdown]
# ## Pattern Generation

# %%
def generate_random_patterns(n_patterns: int, size: int) -> List[np.ndarray]:
    """Generate random bipolar patterns {-1, +1}."""
    patterns = []
    for _ in range(n_patterns):
        pat = np.random.choice([-1.0, 1.0], size=size)
        patterns.append(pat)
    return patterns

# %% [markdown]
# ## McCallum 1995 Protocol

# %%
def run_mccallum_1995_protocol(
    network_size: int = 100,
    max_patterns: int = 100,
    n_probes: int = 300,
    max_pseudoitems: int = 256,
    verbose: bool = True
) -> Tuple[List[int], List[int], List[int]]:
    """
    Run McCallum's 1995 pseudorehearsal protocol.

    Returns:
        patterns_learned: List of M values (1, 2, 3, ...)
        stable_counts: Number of stable patterns at each M
        pseudoitem_counts: Number of pseudoitems found at each M
    """
    # Initialize
    net = McCallumDHN(size=network_size)
    patterns = generate_random_patterns(max_patterns, network_size)

    patterns_learned = []
    stable_counts = []
    pseudoitem_counts = []

    if verbose:
        print(f"McCallum 1995 Protocol: N={network_size}, max_patterns={max_patterns}")
        print(f"Probes: {n_probes}, Max pseudoitems: {max_pseudoitems}")
        print("-" * 60)

    for M in range(1, max_patterns + 1):
        # Build training set
        if M == 1:
            # First pattern: train alone
            training_set = [patterns[0]]
            new_idx = 0
            n_pseudo = 0
        else:
            # Probe for pseudoitems
            pseudoitems = net.probe_for_pseudoitems(n_probes, max_pseudoitems)
            n_pseudo = len(pseudoitems)

            # Training set = pseudoitems + new pattern
            training_set = pseudoitems + [patterns[M - 1]]
            new_idx = len(training_set) - 1

        # Train with delta learning
        epochs = net.train_delta(training_set, new_idx)

        # Count stable patterns (weak criterion: fixed point check)
        n_stable = 0
        for mu in range(M):
            if net.is_stable(patterns[mu]):
                n_stable += 1

        # Record results
        patterns_learned.append(M)
        stable_counts.append(n_stable)
        pseudoitem_counts.append(n_pseudo)

        if verbose and (M <= 10 or M % 10 == 0 or M == max_patterns):
            print(f"M={M:3d}: stable={n_stable:3d}/{M}, pseudoitems={n_pseudo:3d}, epochs={epochs}")

    if verbose:
        print("-" * 60)
        print("Protocol complete!")

    return patterns_learned, stable_counts, pseudoitem_counts

# %% [markdown]
# ## Run the Experiment

# %%
# Run McCallum 1995 protocol
print("Running McCallum 1995 protocol...")
print("=" * 60)
start_time = time.time()

M_values, stable_values, pseudo_values = run_mccallum_1995_protocol(
    network_size=100,
    max_patterns=100,
    n_probes=300,
    max_pseudoitems=256,
    verbose=True
)

elapsed = time.time() - start_time
print(f"\nTotal time: {elapsed:.1f} seconds")

# %% [markdown]
# ## Plot Results

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Stable patterns vs patterns learned
ax1 = axes[0]
ax1.plot(M_values, stable_values, 'b-', linewidth=2, label='Stable patterns')
ax1.plot(M_values, M_values, 'k--', linewidth=1, alpha=0.5, label='Perfect recall (y=x)')
ax1.fill_between(M_values, stable_values, alpha=0.3)
ax1.set_xlabel('Patterns Learned (M)')
ax1.set_ylabel('Number of Stable Patterns')
ax1.set_title('McCallum 1995: Pseudorehearsal Capacity\n(N=100, stability criterion)')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 100)
ax1.set_ylim(0, 105)

# Plot 2: Pseudoitems found
ax2 = axes[1]
ax2.plot(M_values, pseudo_values, 'r-', linewidth=2)
ax2.axhline(y=256, color='k', linestyle='--', alpha=0.5, label='Max (256)')
ax2.set_xlabel('Patterns Learned (M)')
ax2.set_ylabel('Pseudoitems Found')
ax2.set_title('Pseudoitems Discovered per Incorporation')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 100)

plt.tight_layout()
plt.savefig('../plots/mccallum_1995_protocol.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nFigure saved to: scripts/plots/mccallum_1995_protocol.png")

# %% [markdown]
# ## Summary Statistics

# %%
print("\n" + "=" * 60)
print("SUMMARY STATISTICS")
print("=" * 60)

# Final capacity
final_stable = stable_values[-1]
print(f"Final stable patterns after M=100: {final_stable}")

# Capacity at different thresholds
for threshold in [90, 80, 70, 50]:
    # Find first M where stable < threshold% of M
    capacity = 100
    for i, (m, s) in enumerate(zip(M_values, stable_values)):
        if s < threshold / 100 * m:
            capacity = m - 1
            break
    print(f"M* at {threshold}% threshold: {capacity}")

# Average stability ratio
avg_ratio = np.mean([s/m for m, s in zip(M_values, stable_values)])
print(f"\nAverage stability ratio: {avg_ratio:.2%}")

# Pseudoitem statistics
print(f"\nPseudoitems: min={min(pseudo_values)}, max={max(pseudo_values)}, "
      f"mean={np.mean(pseudo_values):.1f}")

# %% [markdown]
# ## Multiple Runs (for averaging)

# %%
def run_multiple_trials(n_trials: int = 10, **kwargs) -> dict:
    """Run multiple trials and collect statistics."""
    all_stable = []
    all_pseudo = []

    print(f"Running {n_trials} trials...")
    for trial in range(n_trials):
        M, stable, pseudo = run_mccallum_1995_protocol(verbose=False, **kwargs)
        all_stable.append(stable)
        all_pseudo.append(pseudo)
        print(f"  Trial {trial+1}/{n_trials} complete")

    all_stable = np.array(all_stable)
    all_pseudo = np.array(all_pseudo)

    return {
        'M': M,
        'stable_mean': all_stable.mean(axis=0),
        'stable_std': all_stable.std(axis=0),
        'pseudo_mean': all_pseudo.mean(axis=0),
        'pseudo_std': all_pseudo.std(axis=0),
    }

# %%
# Run 10 trials for averaging
print("=" * 60)
print("Running multiple trials for averaging...")
print("=" * 60)

results = run_multiple_trials(
    n_trials=10,
    network_size=100,
    max_patterns=100,
    n_probes=300,
    max_pseudoitems=256
)

# %% [markdown]
# ## Plot Averaged Results

# %%
fig, ax = plt.subplots(figsize=(10, 6))

M = results['M']
mean = results['stable_mean']
std = results['stable_std']

# Plot mean with std band
ax.plot(M, mean, 'b-', linewidth=2, label='Mean stable patterns')
ax.fill_between(M, mean - std, mean + std, alpha=0.3, label='±1 std')
ax.plot(M, M, 'k--', linewidth=1, alpha=0.5, label='Perfect recall (y=x)')

ax.set_xlabel('Patterns Learned (M)', fontsize=14)
ax.set_ylabel('Number of Stable Patterns', fontsize=14)
ax.set_title('McCallum 1995 Pseudorehearsal (N=100, 10 trials)\nStability criterion: pattern is fixed point', fontsize=14)
ax.legend(loc='upper left', fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 100)
ax.set_ylim(0, 105)

plt.tight_layout()
plt.savefig('../plots/mccallum_1995_averaged.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nFigure saved to: scripts/plots/mccallum_1995_averaged.png")

# %%
# Print key results
print("\n" + "=" * 60)
print("AVERAGED RESULTS (10 trials)")
print("=" * 60)
print(f"Stable patterns at M=100: {mean[-1]:.1f} ± {std[-1]:.1f}")
print(f"Stable patterns at M=50:  {mean[49]:.1f} ± {std[49]:.1f}")
print(f"Stable patterns at M=20:  {mean[19]:.1f} ± {std[19]:.1f}")

# %% [markdown]
# ## Comparison with McCallum's Original Results
#
# From McCallum's thesis (Figure 4.23), for N=100 with Pr256 (256 pseudoitems):
# - Initial capacity ~10-15 patterns
# - Gradual increase to ~18 stable patterns by M=95
#
# The results above should show similar behavior:
# - Early dip as network gets loaded
# - Eventual plateau around 15-20 stable patterns
# - This demonstrates pseudorehearsal prevents complete catastrophic forgetting
