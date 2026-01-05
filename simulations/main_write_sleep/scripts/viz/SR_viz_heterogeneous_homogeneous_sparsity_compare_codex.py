# %% [markdown]
# # Homogeneous vs Heterogeneous Sparsity Comparison
#
# This script compares memory recovery performance between:
# - **Homogeneous sparsity**: all patterns in a corpus have the same sparsity
# - **Heterogeneous sparsity**: patterns have varying sparsities within a range
#
# ## Methodology
#
# We compare heterogeneous corpora against the **average** of homogeneous corpora
# at the boundary sparsities of the heterogeneous range:
#
# | Comparison | Heterogeneous Range | Homogeneous Average |
# |------------|---------------------|---------------------|
# | Column 1   | width=0.4 -> [0.3, 0.7] | (sparsity=0.3 + sparsity=0.7) / 2 |
# | Column 2   | width=0.8 -> [0.1, 0.9] | (sparsity=0.1 + sparsity=0.9) / 2 |
#
# ## Interpretation
#
# The heatmaps show: **Homogeneous Average - Heterogeneous**
#
# - **Positive (red)**: Homogeneous boundary averages perform better
# - **Negative (blue)**: Heterogeneous mixing performs better
# - **Grey**: No successful runs to compare
#
# This reveals whether mixing sparsities within a single corpus helps or hurts
# memory consolidation compared to separate homogeneous corpora.

# %% Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from pathlib import Path
import sys

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import load_final_results, DATA_DIR

# %%=========================================================================
# CONFIGURATION
# ==========================================================================

# Data sources
HOMOGENEOUS_NAME = "SR_sparsity_sleep_small"
HETEROGENEOUS_NAME = "SR_heterogeneous_sparsity_sleep_small"

# Comparison definitions
# Each entry maps heterogeneous width to corresponding homogeneous boundary sparsities
# Note: heterogeneous uses mean_sparsity=0.5, so width=0.4 means range [0.3, 0.7]
COMPARISONS = [
    {"homo_sparsities": [0.3, 0.7], "hetero_width": 0.4, "label": "[0.3, 0.7]"},
    {"homo_sparsities": [0.1, 0.9], "hetero_width": 0.8, "label": "[0.1, 0.9]"},
]

# Plot settings
SAVE_PLOTS = True
OUTPUT_DIR = Path(__file__).parent.parent / "plots"
DPI = 300

# %% Styling (matching SR_viz.py)
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


def filter_by_float(df, column, value, atol=1e-6):
    """Filter DataFrame rows where column is close to value (handles float precision)."""
    return df[np.isclose(df[column], value, atol=atol)]


def pivot_mean(df, metric, all_num_patterns, all_net_sizes):
    """Pivot mean metric values onto the common grid."""
    pivot = df.pivot_table(
        values=metric,
        index='num_patterns',
        columns='network_size',
        aggfunc='mean'
    )
    return pivot.reindex(index=all_num_patterns, columns=all_net_sizes)


def safe_abs_max(values):
    """Return max(abs(values)) or 0 if all NaN."""
    if np.all(np.isnan(values)):
        return 0.0
    return float(np.nanmax(np.abs(values)))


def compute_difference(df_homo, df_hetero, homo_sparsities, hetero_width, metric,
                       all_net_sizes, all_num_patterns):
    """
    Compute difference: (average of homogeneous conditions) - heterogeneous condition.

    Parameters
    ----------
    df_homo : DataFrame
        Homogeneous sparsity results with 'sparsity' column
    df_hetero : DataFrame
        Heterogeneous sparsity results with 'sparsity_width' column
    homo_sparsities : list
        Sparsity values to average (e.g., [0.3, 0.7])
    hetero_width : float
        Heterogeneous width value (e.g., 0.4)
    metric : str
        Column name to compare ('all_recovered_before_spurious' or 'first_iter_all_found')
    all_net_sizes : array
        All network sizes for consistent grid
    all_num_patterns : array
        All pattern counts for consistent grid

    Returns
    -------
    DataFrame
        Difference matrix (num_patterns x network_size)
    """
    # Get homogeneous pivots for each boundary sparsity
    homo_pivots = []
    for s in homo_sparsities:
        sub = filter_by_float(df_homo, 'sparsity', s)
        homo_pivots.append(pivot_mean(sub, metric, all_num_patterns, all_net_sizes))

    with np.errstate(all='ignore'):
        homo_avg_values = np.nanmean(
            np.stack([pivot.values for pivot in homo_pivots]),
            axis=0
        )

    homo_avg = pd.DataFrame(
        homo_avg_values,
        index=all_num_patterns,
        columns=all_net_sizes
    )

    hetero_sub = filter_by_float(df_hetero, 'sparsity_width', hetero_width)
    hetero_pivot = pivot_mean(hetero_sub, metric, all_num_patterns, all_net_sizes)

    # Compute difference: homogeneous - heterogeneous
    return homo_avg - hetero_pivot


# %% [markdown]
# ## Load and Preprocess Data

# %% Load data
print("=" * 70)
print("LOADING DATA")
print("=" * 70)

homo_path = DATA_DIR / "sleep_results" / HOMOGENEOUS_NAME
hetero_path = DATA_DIR / "sleep_results" / HETEROGENEOUS_NAME

df_homo = load_final_results(homo_path)
df_hetero = load_final_results(hetero_path)

print(f"\nHomogeneous data: {len(df_homo)} simulations from {homo_path}")
print(f"  Sparsity values: {sorted(df_homo['sparsity'].unique())}")

print(f"\nHeterogeneous data: {len(df_hetero)} simulations from {hetero_path}")
print(f"  Sparsity widths: {sorted(df_hetero['sparsity_width'].unique())}")

# %% Preprocessing
# Set first_iter_all_found to NaN for failed recoveries
df_homo = df_homo.copy()
df_hetero = df_hetero.copy()

df_homo.loc[df_homo['all_recovered_before_spurious'] == 0, 'first_iter_all_found'] = np.nan
df_hetero.loc[df_hetero['all_recovered_before_spurious'] == 0, 'first_iter_all_found'] = np.nan

# Use the common grid between homogeneous and heterogeneous results
all_net_sizes = np.sort(np.intersect1d(
    df_homo['network_size'].unique(),
    df_hetero['network_size'].unique()
))
all_num_patterns = np.sort(np.intersect1d(
    df_homo['num_patterns'].unique(),
    df_hetero['num_patterns'].unique()
))

print(f"\nCommon grid: {len(all_net_sizes)} network sizes x {len(all_num_patterns)} pattern counts")

# %% [markdown]
# ## Compute Differences and Create Visualization

# %% Compute all differences and find global color limits
diff_matrices = {}
vmax_success = 0.0
vmax_iter = 0.0

for col_idx, comparison in enumerate(COMPARISONS):
    homo_sparsities = comparison["homo_sparsities"]
    hetero_width = comparison["hetero_width"]

    # Success rate difference
    diff_success = compute_difference(
        df_homo, df_hetero, homo_sparsities, hetero_width,
        'all_recovered_before_spurious', all_net_sizes, all_num_patterns
    ) * 100  # Convert to percentage

    # Iteration difference
    diff_iter = compute_difference(
        df_homo, df_hetero, homo_sparsities, hetero_width,
        'first_iter_all_found', all_net_sizes, all_num_patterns
    )

    diff_matrices[col_idx] = {'success': diff_success, 'iter': diff_iter}
    vmax_success = max(vmax_success, safe_abs_max(diff_success.values))
    vmax_iter = max(vmax_iter, safe_abs_max(diff_iter.values))

# Guard against zero limits
vmax_success = max(vmax_success, 1e-6)
vmax_iter = max(vmax_iter, 1e-6)

print(f"\nColor scale ranges:")
print(f"  Success rate diff: -{vmax_success:.1f}% to +{vmax_success:.1f}%")
print(f"  Iteration diff: -{vmax_iter:.0f} to +{vmax_iter:.0f}")

# %% Create figure
print("\n" + "=" * 70)
print("CREATING VISUALIZATION")
print("=" * 70)

n_cols = len(COMPARISONS)
n_rows = 2  # Row 0: success rate, Row 1: iteration count

# Diverging colormap: red = homo better, blue = hetero better
cmap_diff = mpl.cm.get_cmap('RdBu_r').copy()
cmap_diff.set_bad(color="lightgrey")

# Figure size - match SR_viz.py layout
r = 1.1
fig_width = max(9, 3 * n_cols) / r
fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, 8 / r),
                         sharex=True, sharey=True, squeeze=False)

# %% Plot heatmaps
for col_idx, comparison in enumerate(COMPARISONS):
    range_label = comparison["label"]
    diff_success = diff_matrices[col_idx]['success']
    diff_iter = diff_matrices[col_idx]['iter']

    # Row 0: Success rate difference
    ax = axes[0, col_idx]
    masked_success = np.ma.masked_invalid(diff_success.values)
    im_success = ax.imshow(
        masked_success,
        cmap=cmap_diff,
        vmin=-vmax_success, vmax=vmax_success
    )
    ax.set_title(rf"$s \in {range_label}$")
    ax.invert_yaxis()
    ax.grid(False)

    # Row 1: Iteration difference
    ax = axes[1, col_idx]
    masked_iter = np.ma.masked_invalid(diff_iter.values)
    im_iter = ax.imshow(
        masked_iter,
        cmap=cmap_diff,
        vmin=-vmax_iter, vmax=vmax_iter
    )
    ax.invert_yaxis()
    ax.grid(False)

# %% Set ticks
x_tick_indices = get_spaced_indices(1, len(all_net_sizes) - 1, 4)
y_tick_indices = get_spaced_indices(1, len(all_num_patterns) - 1, 7)

for row in axes:
    for ax in row:
        ax.tick_params(axis='both', which='both', bottom=True, left=True,
                       top=False, right=False)
        ax.set_xticks(x_tick_indices)
        ax.set_xticklabels(all_net_sizes[x_tick_indices])
        ax.set_yticks(y_tick_indices)
        ax.set_yticklabels(all_num_patterns[y_tick_indices])

# %% Add colorbars - positioned to match SR_viz.py layout
cbar1_ax = fig.add_axes([0.92, 0.56, 0.02, 0.3])
cbar1 = fig.colorbar(im_success, cax=cbar1_ax)
cbar1.set_ticks(np.linspace(-vmax_success, vmax_success, 5))
cbar1.set_ticklabels([f'{int(val)}' for val in np.linspace(-vmax_success, vmax_success, 5)])

cbar2_ax = fig.add_axes([0.92, 0.14, 0.02, 0.3])
cbar2 = fig.colorbar(im_iter, cax=cbar2_ax)
cbar2.set_ticks(np.linspace(-vmax_iter, vmax_iter, 5))
cbar2.set_ticklabels([f'{int(val)}' for val in np.linspace(-vmax_iter, vmax_iter, 5)])

# %% Add axis labels
fig.text(0.51, 0.04, 'Network size', ha='center', va='center')
fig.text(0.04, 0.49, 'Nb stored patterns', ha='left', va='center', rotation=90)

# Row labels
fig.text(0.89, 0.71, r'$\Delta$ Success', ha='right', va='center', fontsize=14)
fig.text(0.89, 0.29, r'$\Delta$ Iter', ha='right', va='center', fontsize=14)

# %% Save and display
if SAVE_PLOTS:
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    output_path = OUTPUT_DIR / "SR_homogeneous_vs_heterogeneous_comparison_codex.png"
    plt.savefig(output_path, dpi=DPI)
    print(f"\nSaved figure to: {output_path}")

plt.show()

# %% Summary statistics
print("\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)

for col_idx, comparison in enumerate(COMPARISONS):
    range_label = comparison["label"]
    diff_success = diff_matrices[col_idx]['success']
    diff_iter = diff_matrices[col_idx]['iter']

    print(f"\n{range_label} comparison:")

    # Success rate
    mean_diff_success = np.nanmean(diff_success.values)
    print(f"  Mean success rate diff: {mean_diff_success:+.2f}%")
    if mean_diff_success > 0:
        print("    -> Homogeneous boundaries average performs better overall")
    else:
        print("    -> Heterogeneous mixing performs better overall")

    # Iteration count
    mean_diff_iter = np.nanmean(diff_iter.values)
    print(f"  Mean iteration diff: {mean_diff_iter:+.1f}")
    if mean_diff_iter > 0:
        print("    -> Heterogeneous recovers faster (fewer iterations)")
    else:
        print("    -> Homogeneous boundaries recover faster")

print("\n" + "=" * 70)
print("VISUALIZATION COMPLETE!")
print("=" * 70)

# %%
