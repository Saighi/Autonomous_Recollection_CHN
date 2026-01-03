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
        sub = df_homo[df_homo['sparsity'] == s]
        pivot = sub.pivot_table(
            values=metric,
            index='num_patterns',
            columns='network_size',
            aggfunc='mean'
        ).reindex(index=all_num_patterns, columns=all_net_sizes)
        homo_pivots.append(pivot.values)

    # Average homogeneous conditions using nanmean
    # This handles cases where one condition succeeded but the other didn't
    homo_avg_values = np.nanmean(np.stack(homo_pivots), axis=0)
    homo_avg = pd.DataFrame(
        homo_avg_values,
        index=all_num_patterns,
        columns=all_net_sizes
    )

    # Get heterogeneous pivot
    hetero_sub = df_hetero[df_hetero['sparsity_width'] == hetero_width]
    hetero_pivot = hetero_sub.pivot_table(
        values=metric,
        index='num_patterns',
        columns='network_size',
        aggfunc='mean'
    ).reindex(index=all_num_patterns, columns=all_net_sizes)

    # Compute difference: homogeneous - heterogeneous
    diff = homo_avg - hetero_pivot

    return diff


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

# Get common axis values (intersection of both datasets)
all_net_sizes = np.sort(np.intersect1d(
    df_homo['network_size'].unique(),
    df_hetero['network_size'].unique()
))
all_num_patterns = np.sort(np.intersect1d(
    df_homo['num_patterns'].unique(),
    df_hetero['num_patterns'].unique()
))

print(f"\nCommon grid: {len(all_net_sizes)} network sizes x {len(all_num_patterns)} pattern counts")
print(f"  Network sizes: {all_net_sizes}")
print(f"  Pattern counts: {all_num_patterns}")

# %% Debug: Check data for first comparison
print("\n" + "=" * 70)
print("DEBUG: Data check for first comparison")
print("=" * 70)
test_homo_03 = df_homo[df_homo['sparsity'] == 0.3]
test_homo_07 = df_homo[df_homo['sparsity'] == 0.7]
test_hetero_04 = df_hetero[df_hetero['sparsity_width'] == 0.4]
print(f"Homogeneous sparsity=0.3: {len(test_homo_03)} rows")
print(f"Homogeneous sparsity=0.7: {len(test_homo_07)} rows")
print(f"Heterogeneous width=0.4: {len(test_hetero_04)} rows")

# Check a sample pivot
test_pivot = test_homo_03.pivot_table(
    values='all_recovered_before_spurious',
    index='num_patterns',
    columns='network_size',
    aggfunc='mean'
)
print(f"\nSample pivot shape (homo 0.3): {test_pivot.shape}")
print(f"Sample pivot non-NaN count: {test_pivot.notna().sum().sum()}")

# %% [markdown]
# ## Compute Differences and Create Visualization

# %% MINIMAL DEBUG TEST - plot one matrix directly
print("\n" + "=" * 70)
print("MINIMAL DEBUG TEST")
print("=" * 70)

# Get first comparison data
test_diff = compute_difference(
    df_homo, df_hetero, [0.3, 0.7], 0.4,
    'all_recovered_before_spurious', all_net_sizes, all_num_patterns
) * 100

print(f"Test matrix shape: {test_diff.shape}")
print(f"Test matrix dtype: {test_diff.values.dtype}")
print(f"Test matrix sample (first 3x3):\n{test_diff.values[:3, :3]}")
print(f"Test matrix has NaN: {np.any(np.isnan(test_diff.values))}")
print(f"Test matrix min: {np.nanmin(test_diff.values)}, max: {np.nanmax(test_diff.values)}")

# Simple single plot test
fig_test, ax_test = plt.subplots(figsize=(8, 6))
im_test = ax_test.imshow(test_diff.values, cmap='RdBu_r', vmin=-100, vmax=100)
ax_test.set_title('DEBUG: Single plot test')
fig_test.colorbar(im_test)
plt.savefig(OUTPUT_DIR / "DEBUG_single_plot.png", dpi=150)
print(f"\nSaved debug plot to: {OUTPUT_DIR / 'DEBUG_single_plot.png'}")
plt.show()

# %% Create figure
print("\n" + "=" * 70)
print("CREATING VISUALIZATION")
print("=" * 70)

n_cols = len(COMPARISONS)
n_rows = 2  # Row 0: success rate, Row 1: iteration count

# Diverging colormap: red = homo better, blue = hetero better
cmap_diff = 'RdBu_r'  # Use string name, simpler

# Figure size - don't share axes to avoid issues
fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 10))

# Store images for colorbars
im_success = None
im_iter = None
vmax_success = 0
vmax_iter = 0

# First pass: compute all differences to find global color limits
diff_matrices = {}
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

    # Update global limits
    if not np.all(np.isnan(diff_success.values)):
        vmax_success = max(vmax_success, np.nanmax(np.abs(diff_success.values)))
    if not np.all(np.isnan(diff_iter.values)):
        vmax_iter = max(vmax_iter, np.nanmax(np.abs(diff_iter.values)))

# Guard against zero limits
vmax_success = max(vmax_success, 1e-6)
vmax_iter = max(vmax_iter, 1e-6)

print(f"\nColor scale ranges:")
print(f"  Success rate diff: -{vmax_success:.1f}% to +{vmax_success:.1f}%")
print(f"  Iteration diff: -{vmax_iter:.0f} to +{vmax_iter:.0f}")

# Debug: print matrix info
for col_idx in diff_matrices:
    print(f"\nComparison {col_idx}:")
    print(f"  Success diff shape: {diff_matrices[col_idx]['success'].shape}")
    print(f"  Success diff non-NaN: {diff_matrices[col_idx]['success'].notna().sum().sum()}")
    print(f"  Iter diff shape: {diff_matrices[col_idx]['iter'].shape}")
    print(f"  Iter diff non-NaN: {diff_matrices[col_idx]['iter'].notna().sum().sum()}")

# %% Plot heatmaps
for col_idx, comparison in enumerate(COMPARISONS):
    range_label = comparison["label"]
    diff_success = diff_matrices[col_idx]['success']
    diff_iter = diff_matrices[col_idx]['iter']

    # Debug: print sample values
    print(f"\n[DEBUG] Plotting col {col_idx} success: min={np.nanmin(diff_success.values):.1f}, max={np.nanmax(diff_success.values):.1f}")

    # Row 0: Success rate difference
    ax = axes[0, col_idx]
    im_success = ax.imshow(
        diff_success.values,
        cmap=cmap_diff,
        vmin=-vmax_success, vmax=vmax_success,
        aspect='auto',
        origin='upper'
    )
    ax.set_title(f'Sparsity range: {range_label}')
    ax.grid(False)

    # Row 1: Iteration difference
    ax = axes[1, col_idx]
    im_iter = ax.imshow(
        diff_iter.values,
        cmap=cmap_diff,
        vmin=-vmax_iter, vmax=vmax_iter,
        aspect='auto',
        origin='upper'
    )
    ax.grid(False)

# %% Set ticks
x_tick_indices = get_spaced_indices(0, len(all_net_sizes) - 1, 4)
y_tick_indices = get_spaced_indices(0, len(all_num_patterns) - 1, 7)

for row in axes:
    for ax in row:
        ax.tick_params(axis='both', which='both', bottom=True, left=True,
                       top=False, right=False)
        ax.set_xticks(x_tick_indices)
        ax.set_xticklabels(all_net_sizes[x_tick_indices])
        ax.set_yticks(y_tick_indices)
        ax.set_yticklabels(all_num_patterns[y_tick_indices])

# %% Add colorbars - simple approach
fig.colorbar(im_success, ax=axes[0, :].tolist(), location='right', shrink=0.8, label=r'$\Delta$ Success rate (\%)')
fig.colorbar(im_iter, ax=axes[1, :].tolist(), location='right', shrink=0.8, label=r'$\Delta$ Iterations')

# %% Add axis labels
axes[1, 0].set_xlabel('Network size')
axes[1, 1].set_xlabel('Network size')
axes[0, 0].set_ylabel('Nb stored patterns')
axes[1, 0].set_ylabel('Nb stored patterns')

# %% Save and display
plt.tight_layout()

if SAVE_PLOTS:
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    output_path = OUTPUT_DIR / "SR_homogeneous_vs_heterogeneous_comparison.png"
    fig.savefig(output_path, dpi=DPI, bbox_inches='tight')
    print(f"\nSaved figure to: {output_path}")

fig  # Display in VS Code

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
        print(f"    -> Homogeneous boundaries average performs better overall")
    else:
        print(f"    -> Heterogeneous mixing performs better overall")

    # Iteration count
    mean_diff_iter = np.nanmean(diff_iter.values)
    print(f"  Mean iteration diff: {mean_diff_iter:+.1f}")
    if mean_diff_iter > 0:
        print(f"    -> Heterogeneous recovers faster (fewer iterations)")
    else:
        print(f"    -> Homogeneous boundaries recover faster")

print("\n" + "=" * 70)
print("VISUALIZATION COMPLETE!")
print("=" * 70)

# %%
