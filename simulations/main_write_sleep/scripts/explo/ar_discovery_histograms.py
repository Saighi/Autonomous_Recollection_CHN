# %% [markdown]
# # AR Pattern Discovery - Histogram Analysis
#
# This script produces histogram-based visualizations of Autonomous Retrieval (AR)
# performance across different memory loads and inhibition rates.
#
# ## Metrics
#
# ### 1. Average Number of Revisits per Pattern
#
# For each simulation, we count how many times each pattern was retrieved.
# A **revisit** is defined as any retrieval beyond the first:
# $$\text{revisits}_\mu = \max(0, \text{count}_\mu - 1)$$
#
# The **average revisits** is the mean across all patterns that were retrieved:
# $$\bar{r} = \frac{1}{|\mathcal{F}|} \sum_{\mu \in \mathcal{F}} \text{revisits}_\mu$$
#
# where $\mathcal{F}$ is the set of patterns that were found at least once.
#
# **Interpretation**: Higher revisit counts indicate less efficient exploration,
# as the network is repeatedly visiting the same patterns instead of discovering new ones.
#
# ### 2. Number of Iterations Before Spurious
#
# For each simulation, we count the total number of AR iterations before
# encountering a spurious state (attractor not corresponding to any stored pattern).
#
# **Special case**: If all patterns were successfully recovered before encountering
# a spurious state, the value is set to $p$ (the number of stored patterns).
# This represents the best possible outcome where the network completely explored
# all stored memories.
#
# **Interpretation**:
# - Higher values indicate better capacity before failure
# - A peak at $p$ indicates simulations that successfully recovered all patterns
# - Values below $p$ indicate simulations that hit spurious before complete recovery
#
# ## Experimental Design
#
# - **Rows**: $p \in \{6, 8, 10, 12\}$ (number of stored patterns)
# - **Columns**: $\beta \in \{0.025, 0.05, 0.1\}$ (inhibition plasticity rate)
# - **Repetitions**: 20 per $(p, \beta)$ condition
# - **Network size**: $N = 200$
# - **Pattern correlation**: $\rho = 0.5$

# %% Imports
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import (
    generate_patterns, setup_write_experiment, setup_sleep_experiment,
    run_cpp, list_simulations, read_parameters, DATA_DIR, build
)

PLOTS_DIR = Path(__file__).parent.parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# %% Parameters
print("="*70)
print("EXPERIMENTAL PARAMETERS")
print("="*70)

# Parameter sweep
NUM_PATTERNS = [10,12,14,16]  # Rows in subplot grid
BETA_VALUES = [0.025, 0.05, 0.1]  # Columns in subplot grid
NB_REPETITIONS = 100  # Simulations per (p, beta) condition

# Fixed parameters
NETWORK_SIZE = 200
SPARSITY = 0.5
RHO = 0.5
LEAK = 1.0
DRIVE_TARGET = 6.0
LEARNING_RATE = 0.0001
MOMENTUM = 0.9
MAX_ITER = 100000
CONVERGENCE_THRESHOLD = 0.01
DELTA = 0.01
MAX_QUERIES = 200
NOISE_DYNAMICS = 1
STDDEV_DYNAMICS = 0.01

# Experiment name
EXPERIMENT_NAME = "ar_histogram_analysis"

print(f"Number of patterns: {NUM_PATTERNS}")
print(f"Beta values: {BETA_VALUES}")
print(f"Repetitions per condition: {NB_REPETITIONS}")
print(f"Network size: {NETWORK_SIZE}")
print(f"Pattern correlation (rho): {RHO}")
print(f"\nTotal training runs: {len(NUM_PATTERNS)} x {NB_REPETITIONS} = {len(NUM_PATTERNS) * NB_REPETITIONS}")
print(f"Total sleep runs: {len(NUM_PATTERNS) * NB_REPETITIONS} x {len(BETA_VALUES)} = {len(NUM_PATTERNS) * NB_REPETITIONS * len(BETA_VALUES)}")
print("="*70)

# %% Styling - EXACT match from pattern_push_analysis.py
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

print("Matplotlib styling configured (thesis-ready)")

# %% Build
print("\n" + "="*70)
print("BUILDING C++ EXECUTABLES")
print("="*70)
build()
print("Build complete!\n")

# %% Generate patterns
print("="*70)
print("GENERATING PATTERNS")
print("="*70)

patterns_dict = {}
for p in NUM_PATTERNS:
    patterns_dict[p] = generate_patterns(k=p, n=NETWORK_SIZE, sparsity=SPARSITY, rho=RHO)
    print(f"Generated {p} patterns (N={NETWORK_SIZE}, sparsity={SPARSITY}, rho={RHO})")

print(f"\nTotal pattern sets generated: {len(patterns_dict)}")

# %% Training phase
print("\n" + "="*70)
print("TRAINING PHASE")
print("="*70)
print(f"Total networks to train: {len(NUM_PATTERNS) * NB_REPETITIONS}")
print("="*70 + "\n")

for p in NUM_PATTERNS:
    print(f"\nTraining networks for p = {p}...")

    write_config = setup_write_experiment(
        name=EXPERIMENT_NAME,
        patterns=patterns_dict[p],
        params={
            "leak": LEAK,
            "drive_target": DRIVE_TARGET,
            "learning_rate": LEARNING_RATE,
            "momentum_coef": MOMENTUM,
            "max_iter": MAX_ITER,
            "convergence_threshold": CONVERGENCE_THRESHOLD,
        },
        varying_params={
            "nb_repetition": list(range(NB_REPETITIONS)),
        },
        run_name=f"p_{p}"
    )

    run_cpp("write", write_config)
    print(f"Completed training for p = {p}")

print("\n" + "="*70)
print("TRAINING COMPLETE")
print("="*70)

# %% Sleep phase
print("\n" + "="*70)
print("SLEEP PHASE")
print("="*70)

for beta in BETA_VALUES:
    print(f"\nRunning sleep with beta = {beta}...")

    for p in NUM_PATTERNS:
        trained_dir = DATA_DIR / "trained_networks" / EXPERIMENT_NAME / f"p_{p}"

        sleep_config = setup_sleep_experiment(
            name=EXPERIMENT_NAME,
            trained_networks_dir=trained_dir,
            params={
                "beta": beta,
                "delta": DELTA,
                "noise_dynamics": NOISE_DYNAMICS,
                "stddev_dynamics": STDDEV_DYNAMICS,
                "max_queries": MAX_QUERIES,
                "stop_on_spurious": 1,  # Stop when spurious encountered
                "stop_on_all_found": 1,  # Stop when all patterns recovered
                "save_trajectories": 0,
            },
            run_name=f"p_{p}_beta_{beta}"
        )

        run_cpp("sleep", sleep_config)
        print(f"  Completed p={p}, beta={beta}")

print("\n" + "="*70)
print("SLEEP PHASE COMPLETE")
print("="*70)

# %% Load results
print("\n" + "="*70)
print("LOADING RESULTS")
print("="*70)

# Store all results: results[p][beta] = list of DataFrames
all_results = {}

for p in NUM_PATTERNS:
    all_results[p] = {}

    for beta in BETA_VALUES:
        all_results[p][beta] = []

        results_dir = DATA_DIR / "sleep_results" / EXPERIMENT_NAME / f"p_{p}_beta_{beta}"
        sim_dirs = list_simulations(results_dir)

        for sim_dir in sim_dirs:
            results_file = sim_dir / "results.data"
            if results_file.exists():
                results_df = pd.read_csv(results_file)
                all_results[p][beta].append(results_df)

        print(f"p={p}, beta={beta}: loaded {len(all_results[p][beta])} simulations")

# %% Compute metrics
print("\n" + "="*70)
print("COMPUTING METRICS")
print("="*70)


def compute_metrics(results_df, num_patterns):
    """
    Compute metrics from a single simulation.

    Returns:
        avg_revisits: Average number of revisits per pattern
        iters_before_spurious: Number of iterations before spurious
                               (= num_patterns if all patterns recovered before spurious)
    """
    # Count retrievals per pattern
    pattern_counts = {}
    for _, row in results_df.iterrows():
        idx = int(row['recovered_pattern_idx'])
        if idx >= 0:  # Valid pattern (not spurious)
            pattern_counts[idx] = pattern_counts.get(idx, 0) + 1

    # Average revisits = sum of (count - 1) for each pattern / num patterns found
    total_revisits = sum(max(0, c - 1) for c in pattern_counts.values())
    num_found = len(pattern_counts)
    avg_revisits = total_revisits / num_found if num_found > 0 else 0

    # Iterations before spurious
    # Check if last row is spurious
    last_idx = int(results_df.iloc[-1]['recovered_pattern_idx'])
    if last_idx < 0:  # Ended with spurious
        iters_before_spurious = len(results_df) - 1
    else:
        # All patterns recovered before spurious - use num_patterns as value
        # This represents successful complete exploration
        iters_before_spurious = num_patterns

    return avg_revisits, iters_before_spurious


# Compute metrics for all simulations: metrics[p][beta] = list of (avg_revisits, iters_before_spurious)
metrics = {}

for p in NUM_PATTERNS:
    metrics[p] = {}

    for beta in BETA_VALUES:
        metrics[p][beta] = []

        for results_df in all_results[p][beta]:
            avg_revisits, iters_before_spurious = compute_metrics(results_df, p)
            metrics[p][beta].append((avg_revisits, iters_before_spurious))

        # Print summary statistics
        if metrics[p][beta]:
            revisits_arr = [m[0] for m in metrics[p][beta]]
            iters_arr = [m[1] for m in metrics[p][beta]]
            print(f"p={p:2d}, beta={beta:.3f}: "
                  f"revisits={np.mean(revisits_arr):.2f}+/-{np.std(revisits_arr):.2f}, "
                  f"iters={np.mean(iters_arr):.1f}+/-{np.std(iters_arr):.1f}")

print("\nMetrics computation complete!")

# %% [markdown]
# ## Figure 1: Average Number of Revisits per Pattern
#
# This figure shows the distribution of average revisits across simulations for each
# $(p, \beta)$ condition. Each histogram represents 20 simulations.
#
# **Rows**: Different numbers of stored patterns ($p = 6, 8, 10, 12$)
# **Columns**: Different inhibition rates ($\beta = 0.025, 0.05, 0.1$)
#
# **Interpretation**:
# - Lower values indicate more efficient exploration (less revisiting)
# - We expect revisits to increase with $p$ (harder to avoid revisits with more patterns)
# - Higher $\beta$ may reduce revisits by providing stronger suppression of explored patterns

# %% Figure 1: Revisit Histograms
print("\n" + "="*70)
print("FIGURE 1: Average Revisits Histograms")
print("="*70)

# Collect all revisit data to determine global bins
all_revisits = []
for p in NUM_PATTERNS:
    for beta in BETA_VALUES:
        all_revisits.extend([m[0] for m in metrics[p][beta]])

# Determine global bins for consistent X-axis
if all_revisits:
    x_min = min(all_revisits)
    x_max = max(all_revisits)
    bins_rev = np.linspace(x_min, x_max + 0.01, 15)
else:
    bins_rev = 10

fig, axes = plt.subplots(4, 3, figsize=(15, 16), sharex=True, sharey=True)

for i, p in enumerate(NUM_PATTERNS):
    for j, beta in enumerate(BETA_VALUES):
        ax = axes[i, j]

        # Extract average revisits for this condition
        revisits = [m[0] for m in metrics[p][beta]]

        if revisits:
            ax.hist(revisits, bins=bins_rev, facecolor='white', edgecolor='black', linewidth=2)

        # Title: only top row shows beta
        if i == 0:
            ax.set_title(rf'$\beta = {beta}$')

        # Y-label: only left column shows p
        if j == 0:
            ax.set_ylabel(rf'$p = {p}$', rotation=0, labelpad=40, va='center')

        # X-label: only bottom row
        if i == 3:
            ax.set_xlabel(r'Avg. revisits')

        # Spine configuration
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_color('black')
            ax.spines[spine].set_linewidth(1.5)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

plt.tight_layout()
output_path = PLOTS_DIR / 'ar_revisit_histograms.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Saved: {output_path}")
plt.show()

# %% [markdown]
# ## Figure 2: Number of Iterations Before Spurious
#
# This figure shows the distribution of iterations before encountering a spurious state
# for each $(p, \beta)$ condition. Each histogram represents 20 simulations.
#
# **Special case**: Simulations where all patterns were successfully recovered before
# encountering any spurious state appear at $x = p$ (the number of stored patterns).
# This represents the best possible outcome.
#
# **Rows**: Different numbers of stored patterns ($p = 6, 8, 10, 12$)
# **Columns**: Different inhibition rates ($\beta = 0.025, 0.05, 0.1$)
#
# **Interpretation**:
# - A peak at $p$ indicates many simulations successfully recovered all patterns
# - Lower values indicate earlier failure (spurious before complete recovery)
# - Higher $\beta$ may improve capacity by providing stronger pattern separation

# %% Figure 2: Iterations Before Spurious Histograms
print("\n" + "="*70)
print("FIGURE 2: Iterations Before Spurious Histograms")
print("="*70)

# Collect all iteration data to determine global bins
all_iters = []
for p in NUM_PATTERNS:
    for beta in BETA_VALUES:
        all_iters.extend([m[1] for m in metrics[p][beta]])

# Determine global bins for consistent X-axis
if all_iters:
    x_min = min(all_iters)
    x_max = max(all_iters)
    bins_iter = np.linspace(x_min, x_max + 0.01, 15)
else:
    bins_iter = 10

fig, axes = plt.subplots(4, 3, figsize=(15, 16), sharex=True, sharey=True)

for i, p in enumerate(NUM_PATTERNS):
    for j, beta in enumerate(BETA_VALUES):
        ax = axes[i, j]

        # Extract iterations before spurious for this condition
        iters = [m[1] for m in metrics[p][beta]]

        if iters:
            ax.hist(iters, bins=bins_iter, facecolor='white', edgecolor='black', linewidth=2)

        # Title: only top row shows beta
        if i == 0:
            ax.set_title(rf'$\beta = {beta}$')

        # Y-label: only left column shows p
        if j == 0:
            ax.set_ylabel(rf'$p = {p}$', rotation=0, labelpad=40, va='center')

        # X-label: only bottom row
        if i == 3:
            ax.set_xlabel(r'Iterations')

        # Spine configuration
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_color('black')
            ax.spines[spine].set_linewidth(1.5)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

plt.tight_layout()
output_path = PLOTS_DIR / 'ar_spurious_histograms.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Saved: {output_path}")
plt.show()

# %% [markdown]
# ## Summary
#
# This analysis provides two complementary views of AR performance:
#
# 1. **Revisit histograms** show how efficiently the network explores stored patterns.
#    Lower revisit counts indicate that the inhibition mechanism effectively suppresses
#    already-explored patterns.
#
# 2. **Iterations before spurious** histograms show the capacity of the AR mechanism.
#    Simulations that recover all $p$ patterns before any spurious represent successful
#    complete exploration.
#
# Together, these figures demonstrate the trade-offs in AR performance as memory load
# and inhibition rate vary.
