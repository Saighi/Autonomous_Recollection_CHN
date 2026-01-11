# %% [markdown]
# # Weight Distribution Visualization for Different Leak Parameters

# %% Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import read_parameters, DATA_DIR

PLOTS_DIR = Path(__file__).parent.parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# %% Styling
plt.rcParams.update({
    'text.usetex': True, 'font.family': 'serif', 'font.serif': ['Times'],
    'font.size': 20, 'font.weight': 'bold',
    'axes.labelsize': 25, 'axes.titlesize': 25, 'axes.linewidth': 1.5, 'axes.grid': False,
    'xtick.labelsize': 20, 'ytick.labelsize': 20, 'lines.linewidth': 2.5,
})

def style_ax(ax):
    for s in ['bottom', 'left']: ax.spines[s].set_linewidth(1.5)
    for s in ['top', 'right']: ax.spines[s].set_visible(False)
    ax.tick_params(bottom=True, left=True, top=False, right=False)

# %% Configuration
LEAK_VALUES = [0.5, 1.0, 1.5, 2.0]
TARGET_NS, TARGET_NP = 96, 10
WRITE_PATH = DATA_DIR / "trained_networks" / "SR_leak_sweep"
SLEEP_PATH = DATA_DIR / "sleep_results" / "SR_leak_sleep"

# %% Find simulation directory
def find_simulation(data_dir, leak, ns, np_):
    for sim_dir in data_dir.glob("sim_nb_*"):
        p = read_parameters(sim_dir / "parameters.data")
        if int(p['network_size']) == ns and int(p['num_patterns']) == np_ and abs(float(p['leak']) - leak) < 0.01:
            return sim_dir
    return None

# %% Load data
print("Loading data...")
weights_by_leak, inhib_by_leak = {}, {}

for leak in LEAK_VALUES:
    # Excitatory weights
    sim = find_simulation(WRITE_PATH, leak, TARGET_NS, TARGET_NP)
    if sim:
        weights_by_leak[leak] = np.loadtxt(sim / "weights.data").flatten()
        print(f"  leak={leak}: {sim.name}, std={weights_by_leak[leak].std():.4f}")

    # Inhibitory weights (reconstructed)
    sim = find_simulation(SLEEP_PATH, leak, TARGET_NS, TARGET_NP)
    if sim:
        patterns = np.loadtxt(sim / "patterns.data")
        results = pd.read_csv(sim / "results.data")
        beta = float(read_parameters(sim / "parameters.data").get('beta', 0.1))
        valid_idx = results['recovered_pattern_idx'][results['recovered_pattern_idx'] >= 0]
        inhib_by_leak[leak] = beta * patterns[valid_idx].sum(axis=0)

# %% Plot histograms
def plot_histograms(data_dict, xlabel, filename):
    fig, axes = plt.subplots(1, len(data_dict), figsize=(4 * len(data_dict), 4), sharey=True)
    for ax, (leak, data) in zip(np.atleast_1d(axes), sorted(data_dict.items())):
        ax.hist(data, bins=20, facecolor='white', edgecolor='black', linewidth=2.5)
        ax.set_xlabel(xlabel)
        ax.set_title(rf'$1/r = {leak}$')
        style_ax(ax)
    axes[0].set_ylabel('Count') if len(data_dict) > 1 else None
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=300, bbox_inches='tight')
    print(f"Saved {filename}")
    plt.show()

plot_histograms(weights_by_leak, r'Weight $W_{ij}$', 'SR_leak_weights.png')
plot_histograms(inhib_by_leak, r'Inhibition $A_i$', 'SR_leak_inhib.png')

# %% Plot comparison
leaks = sorted(weights_by_leak.keys())
avg_excit = [np.mean(np.abs(weights_by_leak[l])) for l in leaks]
avg_inhib = [np.mean(np.abs(inhib_by_leak[l])) for l in leaks]

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(leaks, avg_excit, 'o-', color='black', linewidth=2, markersize=10,
        markerfacecolor='white', markeredgecolor='black', markeredgewidth=2,
        label=r'Excitatory $\langle |W_{ij}| \rangle$')
ax.plot(leaks, avg_inhib, 's-', color='black', linewidth=2, markersize=10,
        markerfacecolor='black', label=r'Inhibitory $\langle A_i \rangle$')
ax.set_xlabel(r'Leak $1/r$')
ax.set_ylabel(r'Average absolute weight')
ax.legend(frameon=False)
style_ax(ax)
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'SR_leak_weights_comparison.png', dpi=300, bbox_inches='tight')
print("Saved SR_leak_weights_comparison.png")
plt.show()

# %%
