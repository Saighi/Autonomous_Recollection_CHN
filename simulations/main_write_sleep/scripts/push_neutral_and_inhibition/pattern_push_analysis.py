# %% [markdown]
# # Pattern Push Analysis
#
# This script measures the **push** from the neutral state toward each stored pattern
# in a Continuous Hopfield Network (CHN).
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
#
# A positive $P_\mu$ indicates the network dynamics push toward pattern $\mu$
# from the neutral state.

# %% Imports
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import (generate_patterns, setup_write_experiment, setup_sleep_experiment,
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
    'axes.labelsize': 30,
    'axes.titlesize': 30,
    'xtick.labelsize': 25,
    'ytick.labelsize': 25,
    'legend.fontsize': 20,
    'figure.titlesize': 20,
    'lines.linewidth': 2.5,
    'axes.linewidth': 1.5,
    'axes.grid': False,
    'font.weight': 'bold'
})

# %% Train network
patterns = generate_patterns(k=4, n=200, sparsity=0.5, rho=0.5)

config = setup_write_experiment(
    name="pattern_push_test",
    patterns=patterns,
    params={"leak": 1.0, "drive_target": 6.0, "learning_rate": 0.0001,
            "momentum": 0.9, "max_iterations": 5000, "convergence_threshold": 0.01}
)
run_cpp("write", config)

# %% Load network and compute push metrics
sim_dir = DATA_DIR / "trained_networks" / "pattern_push_test" / "sim_nb_0"
W = np.loadtxt(sim_dir / "weights.data")  # Weight matrix W_ij
X = np.loadtxt(sim_dir / "patterns.data")  # Stored patterns x^mu
params = read_parameters(sim_dir / "parameters.data")

N = int(params["network_size"])  # Network size
M = int(params["num_patterns"])  # Number of stored patterns
v_N = np.ones(N) * 0.5  # Neutral state firing rate: v_N = sigma(0) = 0.5

# Network derivative at neutral state: dv/dt|_{v_N}
sigma = lambda x: 1.0 / (1.0 + np.exp(-x))
f = -v_N + sigma(W @ v_N)  # Simplified dynamics with r=1

# Compute push P_mu for each pattern x^mu
P = np.array([
    np.dot(f, (X[mu] - v_N) / np.linalg.norm(X[mu] - v_N))
    for mu in range(M)
])

for mu in range(M):
    print(f"P_{mu} = {P[mu]:.4f}")

# %% Plot
fig, ax = plt.subplots(figsize=(8, 4))

ax.bar(range(M), P, facecolor='white', edgecolor='black', linewidth=2.5, width=0.6)
ax.axhline(0, color='black', linewidth=1.5)

ax.set_xlabel(r'Pattern index $\mu$')
ax.set_ylabel(r'Push $P_\mu$')
ax.set_xticks(range(M))
ax.tick_params(axis='both', which='both', bottom=True, left=True, top=False, right=False)

# Explicitly show bottom and left spines with black color
for spine in ['bottom', 'left']:
    ax.spines[spine].set_visible(True)
    ax.spines[spine].set_color('black')
    ax.spines[spine].set_linewidth(1.5)
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'pattern_push.png', dpi=300, bbox_inches='tight')
print(f"Saved to {PLOTS_DIR / 'pattern_push.png'}")
plt.show()

# %% [markdown]
# ## Explained Variance Analysis
#
# We decompose the derivative $\mathbf{f} = \frac{d\mathbf{v}}{dt}|_{\mathbf{v}_N}$ into:
# - $\mathbf{f}_\parallel$: projection onto the subspace spanned by $\{\hat{d}_\mu\}_{\mu=1}^M$
# - $\mathbf{f}_\perp = \mathbf{f} - \mathbf{f}_\parallel$: residual orthogonal to all pattern directions
#
# Let $D = [\hat{d}_1 | \cdots | \hat{d}_M]$ be the $N \times M$ matrix of direction vectors.
# The projection onto the column space of $D$ is:
# $$\mathbf{f}_\parallel = D(D^T D)^{-1} D^T \mathbf{f}$$
#
# The **explained variance** quantifies how much of the dynamics is directed toward patterns:
# $$R^2 = \frac{\|\mathbf{f}_\parallel\|^2}{\|\mathbf{f}\|^2}$$

# %% Compute explained variance
D = np.column_stack([(X[mu] - v_N) / np.linalg.norm(X[mu] - v_N) for mu in range(M)])
f_parallel = D @ np.linalg.solve(D.T @ D, D.T @ f)
R2 = np.linalg.norm(f_parallel)**2 / np.linalg.norm(f)**2

print(f"R² (explained) = {R2:.4f}")
print(f"1 - R² (unexplained) = {1 - R2:.4f}")

# %% Plot B: Two vertical bars
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
plt.savefig(PLOTS_DIR / 'pattern_push_R2_bars.png', dpi=300, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Push Evolution During Autonomous Retrieval
#
# During Autonomous Retrieval (AR), self-inhibition $A_i$ is potentiated after each pattern retrieval:
# $$A_i \leftarrow A_i + \beta \cdot v_i(t_f)$$
#
# The dynamics with inhibition become:
# $$\frac{du_i}{dt} = \sum_j W_{ij}v_j - A_i v_i - u_i$$
#
# We track how the push $P_\mu$ toward each pattern changes as inhibition accumulates,
# showing the progressive suppression of already-retrieved patterns.

# %% Run sleep simulation with inhibition matrix saving
sleep_config = setup_sleep_experiment(
    name="pattern_push_test",
    trained_networks_dir=DATA_DIR / "trained_networks" / "pattern_push_test",
    params={
        "beta": 0.1, "delta": 0.01, "max_queries": M,
        "stop_on_spurious": 0, "stop_on_all_found": 0,
        "save_inhibition_matrices": 1
    }
)
run_cpp("sleep", sleep_config)

# %% Load AR results and inhibition matrices
sleep_dir = DATA_DIR / "sleep_results" / "pattern_push_test" / "sim_nb_0"
results = pd.read_csv(sleep_dir / "results.data")
recovered_indices = results["recovered_pattern_idx"].values

# Load inhibition matrices and compute push at each iteration
num_iters = len(results)
P_evolution = np.zeros((num_iters, M))

for t in range(num_iters):
    A = np.loadtxt(sleep_dir / f"inhib_matrix_iter_{t}.data")
    A_diag = np.diag(A)  # Extract diagonal for diagonal inhibition
    f_t = -v_N + sigma(W @ v_N) - A_diag * v_N
    P_evolution[t] = [np.dot(f_t, (X[mu] - v_N) / np.linalg.norm(X[mu] - v_N)) for mu in range(M)]

# %% Plot push evolution (single column, shared x-axis)
fig, axes = plt.subplots(num_iters, 1, figsize=(6, 2.5*num_iters), sharex=True, sharey=True)
axes = np.atleast_1d(axes)

explored_set = set()  # Track patterns explored so far
for t in range(num_iters):
    ax = axes[t]
    colors = ['white'] * M

    # Color previously explored patterns in light orange
    for idx in explored_set:
        colors[idx] = '#FFD699'  # Light orange

    # Color just-retrieved pattern in red and add to explored set
    if recovered_indices[t] >= 0:
        colors[recovered_indices[t]] = '#d62728'  # Red
        explored_set.add(recovered_indices[t])

    ax.bar(range(M), P_evolution[t], color=colors, edgecolor='black', linewidth=2, width=0.6)
    ax.axhline(0, color='black', linewidth=1)
    ax.set_ylabel(f'Iter {t}')

    for spine in ['bottom', 'left']:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color('black')
        ax.spines[spine].set_linewidth(1.5)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

axes[-1].set_xlabel(r'Pattern index $\mu$')
axes[-1].set_xticks(range(M))
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'pattern_push_AR_evolution.png', dpi=300, bbox_inches='tight')
print(f"Saved to {PLOTS_DIR / 'pattern_push_AR_evolution.png'}")
plt.show()

# %% [markdown]
# ## Why Inhibition Reduces Push More for Explored Patterns
#
# The push $P_\mu$ depends on the derivative at neutral state. With inhibition $A_i$:
# $$\mathbf{f}^{(t)} = -\mathbf{v}_N + \sigma(W\mathbf{v}_N) - \mathbf{A}^{(t)} \odot \mathbf{v}_N$$
#
# Since $A_i$ is potentiated by $\beta v_i(t_f)$ where $v_i(t_f) \approx x_i^\mu$ for the retrieved pattern $\mu$:
# - Neurons active in pattern $\mu$ (where $x_i^\mu = 1$) receive strong inhibition: $A_i$ increases significantly
# - Neurons inactive in pattern $\mu$ (where $x_i^\mu = 0$) receive weak inhibition: $A_i$ stays low
#
# The push toward pattern $\mu$ is the projection onto $\hat{d}_\mu \propto (\mathbf{x}^\mu - \mathbf{v}_N)$.
# The inhibition term $-\mathbf{A} \odot \mathbf{v}_N$ is largest precisely where $\mathbf{x}^\mu$ is active,
# which maximally opposes the direction $\hat{d}_\mu$. This creates a **targeted suppression** of the
# explored pattern while leaving other patterns relatively unaffected.

