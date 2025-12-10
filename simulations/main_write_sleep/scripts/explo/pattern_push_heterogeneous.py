# %% [markdown]
# # Pattern Push Analysis with Heterogeneous Sparsities
#
# This script measures the **push** from the neutral state toward patterns
# with varying sparsities in a Continuous Hopfield Network (CHN).
#
# ## Network Dynamics
#
# The CHN dynamics follow (with $c=r=1$):
# $$\frac{du_i}{dt} = \sum_j W_{ij}v_j - u_i$$
# where $v_i = \sigma(u_i)$ is the firing rate and $\sigma(x) = \frac{1}{1+e^{-x}}$.
#
# ## Metric Definition
#
# At the neutral state $\mathbf{u}_N = \mathbf{0}$, the firing rates are
# $\mathbf{v}_N = \sigma(\mathbf{0}) = 0.5 \cdot \mathbf{1}$.
#
# For each stored pattern $\mathbf{x}^\mu$, we define the **push** $P_\mu$ as the projection
# of the derivative onto the direction toward the pattern:
# $$P_\mu = \left\langle \frac{d\mathbf{v}}{dt}\bigg|_{\mathbf{v}_N}, \hat{d}_\mu \right\rangle$$
#
# where $\hat{d}_\mu = \frac{\mathbf{x}^\mu - \mathbf{v}_N}{\|\mathbf{x}^\mu - \mathbf{v}_N\|}$
# is the unit vector pointing from neutral state to pattern $\mu$.

# %% Imports
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import (setup_write_experiment, setup_sleep_experiment,
                   run_cpp, read_parameters, DATA_DIR)

PLOTS_DIR = Path(__file__).parent.parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# %% Styling
sns.set_style("darkgrid")
sns.set_context("paper", font_scale=1.5)
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times']
plt.rcParams.update({
    'font.size': 20,
    'axes.labelsize': 20,
    'axes.titlesize': 20,
    'xtick.labelsize': 17,
    'ytick.labelsize': 17,
    'legend.fontsize': 20,
    'figure.titlesize': 20,
    'lines.linewidth': 2.5,
    'axes.linewidth': 1.5,
    'axes.grid': False,
    'font.weight': 'bold'
})

# %% Generate patterns with heterogeneous sparsities
sparsities = [0.40, 0.45, 0.50, 0.55, 0.60]
N = 200
M = len(sparsities)

# Generate each pattern with its specific sparsity (sparsity = fraction of active units)
np.random.seed(42)
X_list = []
for s in sparsities:
    nb_active = int(s * N)
    pattern = np.zeros(N, dtype=bool)
    pattern[np.random.choice(N, nb_active, replace=False)] = True
    X_list.append(pattern)
patterns = np.array(X_list)

# %% Train network
config = setup_write_experiment(
    name="pattern_push_heterogeneous",
    patterns=patterns,
    params={"leak": 1.0, "drive_target": 6.0, "learning_rate": 0.0001,
            "momentum": 0.9, "max_iterations": 5000, "convergence_threshold": 0.01}
)
run_cpp("write", config)

# %% Load network and compute push metrics
sim_dir = DATA_DIR / "trained_networks" / "pattern_push_heterogeneous" / "sim_nb_0"
W = np.loadtxt(sim_dir / "weights.data")
X = np.loadtxt(sim_dir / "patterns.data")
v_N = np.ones(N) * 0.5

# Create x-axis labels with sparsity values
sparsity_labels = [f'{s:.2f}' for s in sparsities]

# Network derivative at neutral state
sigma = lambda x: 1.0 / (1.0 + np.exp(-x))
f = -v_N + sigma(W @ v_N)

# Compute push P_mu for each pattern
P = np.array([
    np.dot(f, (X[mu] - v_N) / np.linalg.norm(X[mu] - v_N))
    for mu in range(M)
])

print("Push values at neutral state (heterogeneous sparsities):")
for mu in range(M):
    print(f"  Pattern {mu} (s={sparsities[mu]:.2f}): P_{mu} = {P[mu]:.4f}")

# %% Plot push vs sparsity
fig, ax = plt.subplots(figsize=(8, 4))

ax.bar(range(M), P, facecolor='white', edgecolor='black', linewidth=2.5, width=0.6)
ax.axhline(0, color='black', linewidth=1.5)

ax.set_xlabel(r'Sparsity $s$')
ax.set_ylabel(r'Push $P_\mu$')
ax.set_xticks(range(M))
ax.set_xticklabels(sparsity_labels)
ax.tick_params(axis='both', which='both', bottom=True, left=True, top=False, right=False)

for spine in ['bottom', 'left']:
    ax.spines[spine].set_visible(True)
    ax.spines[spine].set_color('black')
    ax.spines[spine].set_linewidth(1.5)
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'pattern_push_heterogeneous.png', dpi=300, bbox_inches='tight')
print(f"Saved to {PLOTS_DIR / 'pattern_push_heterogeneous.png'}")
plt.show()

# %% [markdown]
# ## Explained Variance Analysis
#
# We decompose the derivative $\mathbf{f} = \frac{d\mathbf{v}}{dt}|_{\mathbf{v}_N}$ into:
# - $\mathbf{f}_\parallel$: projection onto the subspace spanned by $\{\hat{d}_\mu\}_{\mu=1}^M$
# - $\mathbf{f}_\perp = \mathbf{f} - \mathbf{f}_\parallel$: residual orthogonal to all pattern directions
#
# The **explained variance**: $R^2 = \frac{\|\mathbf{f}_\parallel\|^2}{\|\mathbf{f}\|^2}$

# %% Compute explained variance
D = np.column_stack([(X[mu] - v_N) / np.linalg.norm(X[mu] - v_N) for mu in range(M)])
f_parallel = D @ np.linalg.solve(D.T @ D, D.T @ f)
R2 = np.linalg.norm(f_parallel)**2 / np.linalg.norm(f)**2

print(f"R² (explained) = {R2:.4f}")
print(f"1 - R² (unexplained) = {1 - R2:.4f}")

# %% Plot R² bars
fig, ax = plt.subplots(figsize=(5, 4))
ax.bar([0, 1], [R2, 1 - R2], facecolor='white', edgecolor='black', linewidth=2.5, width=0.6)
ax.set_xticks([0, 1])
ax.set_xticklabels([r'$R^2$', r'$1 - R^2$'])
ax.set_ylabel('Fraction')
ax.set_ylim(0, 1)
for spine in ['bottom', 'left']:
    ax.spines[spine].set_visible(True)
    ax.spines[spine].set_color('black')
    ax.spines[spine].set_linewidth(1.5)
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'pattern_push_heterogeneous_R2.png', dpi=300, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Push Evolution During Autonomous Retrieval
#
# During AR, self-inhibition $A_i$ is potentiated after each pattern retrieval:
# $$A_i \leftarrow A_i + \beta \cdot v_i(t_f)$$
#
# The dynamics with inhibition become:
# $$\frac{du_i}{dt} = \sum_j W_{ij}v_j - A_i v_i - u_i$$

# %% Run sleep simulation
sleep_config = setup_sleep_experiment(
    name="pattern_push_heterogeneous",
    trained_networks_dir=DATA_DIR / "trained_networks" / "pattern_push_heterogeneous",
    params={
        "beta": 0.1, "delta": 0.01, "max_queries": M,
        "stop_on_spurious": 0, "stop_on_all_found": 0,
        "save_inhibition_matrices": 1
    }
)
run_cpp("sleep", sleep_config)

# %% Load AR results and inhibition matrices
sleep_dir = DATA_DIR / "sleep_results" / "pattern_push_heterogeneous" / "sim_nb_0"
results = pd.read_csv(sleep_dir / "results.data")
recovered_indices = results["recovered_pattern_idx"].values

num_iters = len(results)
P_evolution = np.zeros((num_iters, M))

for t in range(num_iters):
    A = np.loadtxt(sleep_dir / f"inhib_matrix_iter_{t}.data")
    A_diag = np.diag(A)
    f_t = -v_N + sigma(W @ v_N) - A_diag * v_N
    P_evolution[t] = [np.dot(f_t, (X[mu] - v_N) / np.linalg.norm(X[mu] - v_N)) for mu in range(M)]

# %% Plot push evolution (single column, shared x-axis)
fig, axes = plt.subplots(num_iters, 1, figsize=(6, 2.5*num_iters), sharex=True, sharey=True)
axes = np.atleast_1d(axes)

explored_set = set()
for t in range(num_iters):
    ax = axes[t]
    colors = ['white'] * M

    for idx in explored_set:
        colors[idx] = '#FFD699'  # Light orange (past explored)

    if recovered_indices[t] >= 0:
        colors[recovered_indices[t]] = '#d62728'  # Red (just retrieved)
        explored_set.add(recovered_indices[t])

    ax.bar(range(M), P_evolution[t], color=colors, edgecolor='black', linewidth=2, width=0.6)
    ax.axhline(0, color='black', linewidth=1)
    ax.set_ylabel(f'Iter {t}', fontsize=14)

    for spine in ['bottom', 'left']:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color('black')
        ax.spines[spine].set_linewidth(1.5)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

axes[-1].set_xlabel(r'Sparsity $s$', fontsize=16)
axes[-1].set_xticks(range(M))
axes[-1].set_xticklabels(sparsity_labels)
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'pattern_push_heterogeneous_AR_evolution.png', dpi=300, bbox_inches='tight')
print(f"Saved to {PLOTS_DIR / 'pattern_push_heterogeneous_AR_evolution.png'}")
plt.show()

# %% [markdown]
# ## Why Inhibition Reduces Push More for Explored Patterns
#
# The push $P_\mu$ depends on the derivative at neutral state. With inhibition $A_i$:
# $$\mathbf{f}^{(t)} = -\mathbf{v}_N + \sigma(W\mathbf{v}_N) - \mathbf{A}^{(t)} \odot \mathbf{v}_N$$
#
# Since $A_i$ is potentiated by $\beta v_i(t_f)$ where $v_i(t_f) \approx x_i^\mu$ for the retrieved pattern $\mu$:
# - Neurons active in pattern $\mu$ (where $x_i^\mu = 1$) receive strong inhibition
# - Neurons inactive in pattern $\mu$ (where $x_i^\mu = 0$) receive weak inhibition
#
# The inhibition term $-\mathbf{A} \odot \mathbf{v}_N$ is largest precisely where $\mathbf{x}^\mu$ is active,
# which maximally opposes the direction $\hat{d}_\mu$. This creates a **targeted suppression** of the
# explored pattern while leaving other patterns relatively unaffected.
