# %% [markdown]
# # Spontaneous Recovery Visualization - Leak Parameter Sweep
#
# This script loads and visualizes the results from SR_leak_sim.py
#
# Creates heatmap visualizations showing:
# - % simulations with successful recovery (all patterns before spurious)
# - First iteration where all patterns were recovered
# %% Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from pathlib import Path
import sys

# Add scripts directory to path (parent.parent = scripts/)
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import load_results, load_final_results, DATA_DIR

# %%=========================================================================
# CONFIGURATION SECTION
# ==========================================================================

# Experiment name (must match SR_leak_sim.py)
# SLEEP_NAME = "SR_leak_sleep"
SLEEP_NAME = "comparison_chn_cpp_sleep"
# SLEEP_NAME = "SR_leak_strong_sleep"
# SLEEP_NAME = "SR_sparsity_sleep"
#SLEEP_NAME = "SR_sparsity_sleep_small"
# SLEEP_NAME = "SR_heterogeneous_sparsity_sleep"
# SLEEP_NAME = "SR_heterogeneous_sparsity_sleep_small"
# Visualization parameters
# PARAM_NAME = "leak"  # Parameter that varies
PARAM_NAME = "rho"  # Parameter that varies
# PARAM_LATEX_SYMBOL = r"1/r"  # LaTeX symbol for the parameter
PARAM_LATEX_SYMBOL = r"\rho"  # LaTeX symbol for the parameter
# PARAM_NAME = "sparsity"  # Parameter that varies
# PARAM_LATEX_SYMBOL = "s"  # LaTeX symbol for the parameter
# PARAM_NAME = "sparsity_width"  # Parameter that varies
# PARAM_LATEX_SYMBOL = r"{\Delta}{s}"  # LaTeX symbol for the parameter
VALUES_TO_PLOT = None  # None = plot all values, or specify list like [0.25, 0.5, 1.0]

# Plot settings
SAVE_PLOTS = True
OUTPUT_DIR = Path(__file__).parent.parent / "plots"  # scripts/plots/
DPI = 300

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

# %% Helper functions
def get_spaced_indices(a, n, num_ticks=4):
    """Generate evenly spaced indices for tick marks."""
    return np.linspace(a, n, num_ticks, dtype=int)

# %% [markdown]
# ## Load and Preprocess Results

# %% Load data
print("="*70)
print("LOADING DATA")
print("="*70)

results_path = DATA_DIR / "sleep_results" / SLEEP_NAME

# Use load_final_results() which automatically:
# 1. Loads final_results.csv if available (one row per simulation)
# 2. Falls back to computing from all_simulation_data.csv via groupby
# 3. Also works with consolidated experiment.db files
df_last = load_final_results(results_path)

print(f"\nLoaded {len(df_last)} simulations from:")
print(f"  {results_path}")

# Verify parameter exists
if PARAM_NAME not in df_last.columns:
    available = [col for col in df_last.columns if col not in
                ['sim_ID', 'query_iter', 'nb_fnd_pat', 'nb_spurious',
                 'nb_iter_biased', 'nb_iter_free']]
    raise ValueError(f"Parameter '{PARAM_NAME}' not found. Available: {available}")

# %% Determine values to plot
all_param_values = np.sort(df_last[PARAM_NAME].unique())

if VALUES_TO_PLOT is None:
    values_to_plot = all_param_values
else:
    values_to_plot = [v for v in VALUES_TO_PLOT if v in all_param_values]
    if not values_to_plot:
        raise ValueError(f"None of {VALUES_TO_PLOT} found in data. Available: {list(all_param_values)}")

# %% [markdown]
# ## Summary Statistics

# %% Summary
print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)

print(f"\nParameter: {PARAM_NAME}")
print(f"Values to plot: {list(values_to_plot)}")
print(f"\nSuccess rates by {PARAM_NAME}:")

for param_val in values_to_plot:
    subset = df_last[df_last[PARAM_NAME] == param_val]
    success_rate = subset['all_recovered_before_spurious'].mean() * 100
    n_sims = len(subset)
    print(f"  {PARAM_NAME} = {param_val}: {success_rate:.1f}% ({n_sims} simulations)")

print(f"\nOverall success rate: {df_last['all_recovered_before_spurious'].mean()*100:.1f}%")

# %% [markdown]
# ## Main Heatmap Visualization
#
# Creates a 2-row × N-column figure where:
# - Row 1: % simulations with successful recovery (0 or 100% per cell)
# - Row 2: First iteration where all patterns were recovered
# - Each column represents a different parameter value

# %% Get unique values for axes
all_net_sizes = np.sort(df_last['network_size'].unique())
all_num_patterns = np.sort(df_last['num_patterns'].unique())

# %% Calculate global color scale limits
global_max_success = 100  # Binary: 0 or 100%
global_max_iter = 0

for param_val in values_to_plot:
    sub = df_last[df_last[PARAM_NAME] == param_val].copy()

    # For unsuccessful recoveries, set first_iter_all_found to NaN
    sub.loc[sub['all_recovered_before_spurious'] == 0, 'first_iter_all_found'] = np.nan

    pivot_iter = sub.pivot_table(
        values='first_iter_all_found',
        index='num_patterns',
        columns='network_size',
    )
    if not np.all(np.isnan(pivot_iter.values)):
        global_max_iter = max(global_max_iter, np.nanmax(pivot_iter.values))

print(f"\nColor scale ranges:")
print(f"  Success rate: 0 - {global_max_success}%")
print(f"  First iteration: 0 - {global_max_iter:.0f}")

# %% Create main heatmap figure
print("\n" + "="*70)
print("CREATING VISUALIZATIONS")
print("="*70)

n_cols = len(values_to_plot)

# Colormap with grey for no-data
default_cmap_name = plt.rcParams["image.cmap"]
cmap_iter = mpl.cm.get_cmap(default_cmap_name).copy()
cmap_iter.set_bad(color="lightgrey")

# Create figure - match original layout
r = 1.1
fig_width = max(9, 3 * n_cols) / r
fig, axes = plt.subplots(2, n_cols, figsize=(fig_width, 8 / r),
                         sharex=True, sharey=True, squeeze=False)

for i, param_val in enumerate(values_to_plot):
    sub = df_last[df_last[PARAM_NAME] == param_val].copy()

    # For unsuccessful recoveries, set first_iter_all_found to NaN (for 2nd row)
    sub.loc[sub['all_recovered_before_spurious'] == 0, 'first_iter_all_found'] = np.nan

    # Row 1: % successful recovery (will be 0 or 100% since one sim per cell)
    pivot_success = sub.pivot_table(
        values='all_recovered_before_spurious',
        index='num_patterns',
        columns='network_size',
    )
    im1 = axes[0, i].imshow(pivot_success * 100,
                            vmin=0, vmax=global_max_success)
    axes[0, i].set_title(rf"${PARAM_LATEX_SYMBOL}={param_val}$")
    axes[0, i].invert_yaxis()
    axes[0, i].grid(False)

    # Row 2: First iteration of full recovery (NaN -> grey)
    pivot_iter = (
        sub.pivot_table(
            values='first_iter_all_found',
            index='num_patterns',
            columns='network_size',
        )
        .reindex(index=all_num_patterns, columns=all_net_sizes)
    )
    masked_iter = np.ma.masked_invalid(pivot_iter.values)
    im2 = axes[1, i].imshow(masked_iter,
                            vmin=0, vmax=global_max_iter,
                            cmap=cmap_iter)
    axes[1, i].invert_yaxis()
    axes[1, i].grid(False)

# Set ticks
x_tick_indices = get_spaced_indices(1, len(all_net_sizes) - 1, 4)
y_tick_indices = get_spaced_indices(1, len(all_num_patterns) - 1, 7)

for row in axes:
    for ax in row:
        ax.tick_params(axis='both', which='both', bottom=True, left=True,
                      top=False, right=False)
        ax.set_xticks(x_tick_indices, all_net_sizes[x_tick_indices])
        ax.set_yticks(y_tick_indices, all_num_patterns[y_tick_indices])

# Add colorbars - positioned to match original layout
cbar1_ax = fig.add_axes([0.92, 0.56, 0.02, 0.3])
cbar1 = fig.colorbar(im1, cax=cbar1_ax)
cbar1.set_ticks(np.linspace(0, 100, 5))
cbar1.set_ticklabels([f'{int(val)}' for val in np.linspace(0, 100, 5)])

cbar2_ax = fig.add_axes([0.92, 0.14, 0.02, 0.3])
cbar2 = fig.colorbar(im2, cax=cbar2_ax)
cbar2.set_ticks(np.linspace(0, global_max_iter, 5))
cbar2.set_ticklabels([f'{int(val)}' for val in np.linspace(1, global_max_iter, 5)])

# Add axis labels
fig.text(0.51, 0.04, 'Network size', ha='center', va='center')
fig.text(0.04, 0.49, 'Nb stored pattern', ha='left', va='center', rotation=90)

# Save plot - NO tight_layout after adding colorbars
if SAVE_PLOTS:
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    output_path = OUTPUT_DIR / f"SR_{PARAM_NAME}_heatmap.png"
    plt.savefig(output_path, dpi=DPI)
    print(f"\nSaved main heatmap to: {output_path}")

plt.show()

# %% [markdown]
# ## Detailed Heatmap for Single Parameter Value
#
# Shows detailed success counts (number of successful simulations)
# for the first parameter value

# %% Detailed heatmap
detail_param_val = values_to_plot[0]
sub = df_last[df_last[PARAM_NAME] == detail_param_val].copy()

# Count successful recoveries per (network_size, num_patterns) combination
# Using aggfunc to count where all_recovered_before_spurious == 1
pivot_success = sub.pivot_table(
    values='all_recovered_before_spurious',
    index='num_patterns',
    columns='network_size',
    aggfunc=lambda x: (~(x == 0)).sum(),  # Count successes
    fill_value=0
)

fig, ax = plt.subplots(figsize=(12, 12))
sns.heatmap(
    pivot_success,
    annot=True, fmt="d",
    ax=ax, annot_kws={"fontsize": 18},
    cbar=False
)

ax.set_xlabel("Network size", fontsize=22)
ax.set_ylabel("Nb stored patterns", fontsize=22)
ax.set_title(rf"Successful Recoveries (${PARAM_LATEX_SYMBOL}={detail_param_val}$)",
            fontsize=24)
ax.tick_params(axis='both', which='both', bottom=True, left=True,
              top=False, right=False)

plt.xticks(rotation=45, ha="right", fontsize=18)
plt.yticks(rotation=0, fontsize=18)
ax.invert_yaxis()
plt.tight_layout()

if SAVE_PLOTS:
    detail_path = OUTPUT_DIR / f"SR_{PARAM_NAME}_detailed_{detail_param_val}.png"
    plt.savefig(detail_path, dpi=DPI)
    print(f"Saved detailed heatmap to: {detail_path}")

plt.show()

# %% Final summary
print("\n" + "="*70)
print("VISUALIZATION COMPLETE!")
print("="*70)
if SAVE_PLOTS:
    print(f"\nPlots saved to: {OUTPUT_DIR.absolute()}")
print(f"\nGenerated:")
print(f"  - Main heatmap with {len(values_to_plot)} parameter values")
print(f"  - Detailed heatmap for {PARAM_NAME}={detail_param_val}")
print("="*70 + "\n")

# %%
