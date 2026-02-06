# %% [markdown]
# # Visualization: CHN (AR) vs DHN (Hebbian & Storkey) Comparison
#
# ## Overview
#
# This script generates figures comparing retrieval capacity across three methods:
# 1. **AR (CHN)** - Our Autonomous Retrieval method with sleep consolidation
# 2. **Hebbian (DHN)** - Discrete Hopfield with classic outer-product learning
# 3. **Storkey (DHN)** - Discrete Hopfield with local-field-corrected learning
#
# ## Figures Generated
#
# ### Figure 1-3: 5x4 Heatmap Grids (one per method)
#
# Each method gets a separate figure with:
# - **Rows**: 5 correlations (rho = 0.1, 0.25, 0.5, 0.75, 1.0)
# - **Columns**: 4 informed fractions (90%, 50%, 20%, 10%)
# - **Each cell**: Heatmap of success rate
#   - X-axis: Network size [100, 200, ..., 1000]
#   - Y-axis: Number of patterns [10, 15, ..., 100]
#   - Color: Success rate 0-100% (viridis colormap)
#
# **For AR method**: Since AR success at a given load corresponds to retrieval
# with ~10% informed units, we show sleep success (`all_recovered_before_spurious`)
# replicated across all 4 columns for consistent layout.
#
# **For Hebbian/Storkey**: Show actual partial cue query success rate for each
# informed fraction.
#
# ### Figure 4: 5x4 Line Plot Grid (90% Threshold Comparison)
#
# Compares storage capacity across methods:
# - **Rows**: 5 correlations
# - **Columns**: 4 informed fractions
# - **Each subplot**: 3 curves (AR, Hebbian, Storkey)
# - **X-axis**: Network size [100-1000]
# - **Y-axis**: Maximum patterns achieving >= 90% retrieval success
#
# This shows how capacity scales with network size for each method.
#
# ## Key Findings Expected
#
# Based on theoretical analysis and our prior experiments:
#
# 1. **Storkey > Hebbian**: Storkey should show higher capacity than Hebbian
#    due to crosstalk cancellation (~0.42N vs ~0.138N theoretical)
#
# 2. **AR competitive**: AR should compete with or exceed Storkey at lower
#    informed fractions due to iterative refinement during sleep
#
# 3. **Correlation effects**: Capacity advantage should be more pronounced
#    at lower correlations where pattern separation is harder
#
# 4. **Scaling**: All methods should show roughly linear capacity scaling
#    with network size, but with different slopes

# %% Imports
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_results, load_final_results, DATA_DIR

# %%=========================================================================
# CONFIGURATION SECTION
# ==========================================================================
# These parameters MUST match run_chn_sim.py and run_dhn_sim.py!

# Experiment names (must match the simulation scripts)
CHN_SLEEP_NAME = "comparison_chn_sleep"
DHN_HEBBIAN_QUERY_NAME = "comparison_dhn_hebbian_query"
DHN_STORKEY_QUERY_NAME = "comparison_dhn_storkey_query"

# Parameter grids (must match simulation scripts)
NETWORK_SIZES = list(range(100, 1001, 100))  # [100, 200, ..., 1000]
NUM_PATTERNS = list(range(10, 101, 5))       # [10, 15, ..., 100]
CORRELATIONS = [0.1, 0.25, 0.5, 0.75, 1.0]   # Pattern correlations (rows)
INFORMED_FRACTIONS = [0.9, 0.5, 0.2, 0.1]    # Partial cue fractions (columns)

# Visualization settings
SUCCESS_THRESHOLD = 0.9   # 90% threshold for capacity curves
HEATMAP_CMAP = 'viridis'  # Colormap for heatmaps
METHOD_COLORS = {
    'AR': '#1f77b4',       # Blue
    'Hebbian': '#ff7f0e',  # Orange
    'Storkey': '#2ca02c',  # Green
}

# Output directory for plots
PLOTS_DIR = DATA_DIR / "plots" / "comparison"

# %% Matplotlib settings
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 12,
    'figure.titlesize': 16,
    'lines.linewidth': 2,
    'axes.linewidth': 1.2,
    'axes.grid': False,
    'font.weight': 'normal',
})

# Create output directory
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
SAVE_PLOTS = True

# %% [markdown]
# ## Load Data

# %% Load CHN sleep results
print("Loading CHN sleep results...")
chn_sleep_dir = DATA_DIR / "sleep_results" / CHN_SLEEP_NAME

if chn_sleep_dir.exists():
    chn_df = load_final_results(chn_sleep_dir)
    print(f"  Loaded {len(chn_df)} CHN sleep simulations")
else:
    print(f"  WARNING: CHN sleep results not found at {chn_sleep_dir}")
    chn_df = None

# %% Load DHN Hebbian query results
print("Loading DHN Hebbian query results...")
hebbian_query_dir = DATA_DIR / "query_results" / DHN_HEBBIAN_QUERY_NAME

if hebbian_query_dir.exists():
    hebbian_df = load_results(hebbian_query_dir)
    print(f"  Loaded {len(hebbian_df)} Hebbian query simulations")
else:
    print(f"  WARNING: Hebbian query results not found at {hebbian_query_dir}")
    hebbian_df = None

# %% Load DHN Storkey query results
print("Loading DHN Storkey query results...")
storkey_query_dir = DATA_DIR / "query_results" / DHN_STORKEY_QUERY_NAME

if storkey_query_dir.exists():
    storkey_df = load_results(storkey_query_dir)
    print(f"  Loaded {len(storkey_df)} Storkey query simulations")
else:
    print(f"  WARNING: Storkey query results not found at {storkey_query_dir}")
    storkey_df = None

# %% [markdown]
# ## Helper Functions

# %% Helper functions
def filter_by_rho(df, rho, col='rho', atol=1e-6):
    """Filter DataFrame by correlation value (handles float precision)."""
    return df[np.isclose(df[col], rho, atol=atol)]

def filter_by_informed(df, informed, col='informed_fraction', atol=1e-6):
    """Filter DataFrame by informed fraction (handles float precision)."""
    return df[np.isclose(df[col], informed, atol=atol)]

def compute_success_pivot(df, success_col='query_success_rate'):
    """
    Compute success rate pivot table.

    Returns:
        pivot: DataFrame with network_size as columns, num_patterns as index
    """
    # Group by (network_size, num_patterns) and average success rate
    grouped = df.groupby(['network_size', 'num_patterns'])[success_col].mean().reset_index()

    # Pivot
    pivot = grouped.pivot_table(
        values=success_col,
        index='num_patterns',
        columns='network_size',
        aggfunc='mean'
    )

    return pivot

def find_max_capacity(df, threshold=SUCCESS_THRESHOLD, success_col='query_success_rate'):
    """
    For each network size, find max patterns achieving >= threshold success.

    Returns:
        Dict mapping network_size -> max_patterns
    """
    capacities = {}

    for net_size in NETWORK_SIZES:
        subset = df[df['network_size'] == net_size].copy()
        if len(subset) == 0:
            capacities[net_size] = 0
            continue

        # Group by num_patterns, average success
        by_patterns = subset.groupby('num_patterns')[success_col].mean()

        # Find max patterns achieving threshold
        successful = by_patterns[by_patterns >= threshold]
        if len(successful) == 0:
            capacities[net_size] = 0
        else:
            capacities[net_size] = successful.index.max()

    return capacities

def get_spaced_indices(a, n, num_ticks=5):
    """Generate evenly spaced indices for tick marks."""
    return np.linspace(a, n, num_ticks, dtype=int)

# %% [markdown]
# ## Figure 1: AR Method Heatmaps
#
# For AR, we show sleep success rate. Since AR success corresponds to ~10%
# informed cue capacity, we replicate the same heatmap across all columns.

# %% Figure 1: AR heatmaps
if chn_df is not None:
    print("\nGenerating AR heatmap figure...")

    fig_ar, axes_ar = plt.subplots(5, 4, figsize=(16, 20), sharex=True, sharey=True)
    fig_ar.suptitle('AR Method (CHN + Sleep) - Success Rate', fontsize=18, fontweight='bold')

    # Determine success column
    success_col = 'all_recovered_before_spurious'

    cmap = mpl.colormaps.get_cmap(HEATMAP_CMAP).copy()
    cmap.set_bad(color='lightgrey')

    for row_idx, rho in enumerate(CORRELATIONS):
        # Filter by correlation
        rho_df = filter_by_rho(chn_df, rho)

        if len(rho_df) == 0:
            for col_idx in range(4):
                axes_ar[row_idx, col_idx].text(0.5, 0.5, 'No data',
                    ha='center', va='center', transform=axes_ar[row_idx, col_idx].transAxes)
            continue

        # Compute success pivot (same for all columns since AR doesn't use informed_fraction)
        pivot = compute_success_pivot(rho_df, success_col=success_col)

        for col_idx, informed in enumerate(INFORMED_FRACTIONS):
            ax = axes_ar[row_idx, col_idx]

            # Plot heatmap
            data = pivot.values * 100  # Convert to percentage
            masked = np.ma.masked_invalid(data)

            im = ax.imshow(masked, vmin=0, vmax=100, cmap=cmap, aspect='auto')
            ax.invert_yaxis()

            # Column title (top row only)
            if row_idx == 0:
                ax.set_title(f'{int(informed*100)}% informed', fontweight='bold')

            # Row label (left column only)
            if col_idx == 0:
                ax.set_ylabel(rf'$\rho={rho}$' + '\nNum patterns')

            # X ticks (bottom row only)
            if row_idx == len(CORRELATIONS) - 1:
                x_ticks = get_spaced_indices(0, len(pivot.columns) - 1, 5)
                ax.set_xticks(x_ticks)
                ax.set_xticklabels([pivot.columns[i] for i in x_ticks])
                ax.set_xlabel('Network size')

            # Y ticks
            y_ticks = get_spaced_indices(0, len(pivot.index) - 1, 6)
            ax.set_yticks(y_ticks)
            ax.set_yticklabels([pivot.index[i] for i in y_ticks])

    # Colorbar
    cbar_ax = fig_ar.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig_ar.colorbar(im, cax=cbar_ax)
    cbar.set_label('Success rate (%)', fontsize=14)

    plt.tight_layout(rect=[0, 0, 0.9, 0.96])

    if SAVE_PLOTS:
        fig_ar.savefig(PLOTS_DIR / 'ar_heatmaps.png', dpi=300, bbox_inches='tight')
        print(f"  Saved to: {PLOTS_DIR / 'ar_heatmaps.png'}")

# %% [markdown]
# ## Figure 2: Hebbian Method Heatmaps

# %% Figure 2: Hebbian heatmaps
if hebbian_df is not None:
    print("\nGenerating Hebbian heatmap figure...")

    fig_heb, axes_heb = plt.subplots(5, 4, figsize=(16, 20), sharex=True, sharey=True)
    fig_heb.suptitle('Hebbian (DHN) - Partial Cue Success Rate', fontsize=18, fontweight='bold')

    cmap = mpl.colormaps.get_cmap(HEATMAP_CMAP).copy()
    cmap.set_bad(color='lightgrey')

    for row_idx, rho in enumerate(CORRELATIONS):
        for col_idx, informed in enumerate(INFORMED_FRACTIONS):
            ax = axes_heb[row_idx, col_idx]

            # Filter by (rho, informed_fraction)
            subset = filter_by_rho(hebbian_df, rho)
            subset = filter_by_informed(subset, informed)

            if len(subset) == 0:
                ax.text(0.5, 0.5, 'No data',
                    ha='center', va='center', transform=ax.transAxes)
                continue

            # Compute success pivot
            pivot = compute_success_pivot(subset, success_col='query_success_rate')

            # Plot heatmap
            data = pivot.values * 100
            masked = np.ma.masked_invalid(data)

            im = ax.imshow(masked, vmin=0, vmax=100, cmap=cmap, aspect='auto')
            ax.invert_yaxis()

            # Labels
            if row_idx == 0:
                ax.set_title(f'{int(informed*100)}% informed', fontweight='bold')
            if col_idx == 0:
                ax.set_ylabel(rf'$\rho={rho}$' + '\nNum patterns')
            if row_idx == len(CORRELATIONS) - 1:
                x_ticks = get_spaced_indices(0, len(pivot.columns) - 1, 5)
                ax.set_xticks(x_ticks)
                ax.set_xticklabels([pivot.columns[i] for i in x_ticks])
                ax.set_xlabel('Network size')

            y_ticks = get_spaced_indices(0, len(pivot.index) - 1, 6)
            ax.set_yticks(y_ticks)
            ax.set_yticklabels([pivot.index[i] for i in y_ticks])

    cbar_ax = fig_heb.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig_heb.colorbar(im, cax=cbar_ax)
    cbar.set_label('Success rate (%)', fontsize=14)

    plt.tight_layout(rect=[0, 0, 0.9, 0.96])

    if SAVE_PLOTS:
        fig_heb.savefig(PLOTS_DIR / 'hebbian_heatmaps.png', dpi=300, bbox_inches='tight')
        print(f"  Saved to: {PLOTS_DIR / 'hebbian_heatmaps.png'}")

# %% [markdown]
# ## Figure 3: Storkey Method Heatmaps

# %% Figure 3: Storkey heatmaps
if storkey_df is not None:
    print("\nGenerating Storkey heatmap figure...")

    fig_stk, axes_stk = plt.subplots(5, 4, figsize=(16, 20), sharex=True, sharey=True)
    fig_stk.suptitle('Storkey (DHN) - Partial Cue Success Rate', fontsize=18, fontweight='bold')

    cmap = mpl.colormaps.get_cmap(HEATMAP_CMAP).copy()
    cmap.set_bad(color='lightgrey')

    for row_idx, rho in enumerate(CORRELATIONS):
        for col_idx, informed in enumerate(INFORMED_FRACTIONS):
            ax = axes_stk[row_idx, col_idx]

            subset = filter_by_rho(storkey_df, rho)
            subset = filter_by_informed(subset, informed)

            if len(subset) == 0:
                ax.text(0.5, 0.5, 'No data',
                    ha='center', va='center', transform=ax.transAxes)
                continue

            pivot = compute_success_pivot(subset, success_col='query_success_rate')

            data = pivot.values * 100
            masked = np.ma.masked_invalid(data)

            im = ax.imshow(masked, vmin=0, vmax=100, cmap=cmap, aspect='auto')
            ax.invert_yaxis()

            if row_idx == 0:
                ax.set_title(f'{int(informed*100)}% informed', fontweight='bold')
            if col_idx == 0:
                ax.set_ylabel(rf'$\rho={rho}$' + '\nNum patterns')
            if row_idx == len(CORRELATIONS) - 1:
                x_ticks = get_spaced_indices(0, len(pivot.columns) - 1, 5)
                ax.set_xticks(x_ticks)
                ax.set_xticklabels([pivot.columns[i] for i in x_ticks])
                ax.set_xlabel('Network size')

            y_ticks = get_spaced_indices(0, len(pivot.index) - 1, 6)
            ax.set_yticks(y_ticks)
            ax.set_yticklabels([pivot.index[i] for i in y_ticks])

    cbar_ax = fig_stk.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig_stk.colorbar(im, cax=cbar_ax)
    cbar.set_label('Success rate (%)', fontsize=14)

    plt.tight_layout(rect=[0, 0, 0.9, 0.96])

    if SAVE_PLOTS:
        fig_stk.savefig(PLOTS_DIR / 'storkey_heatmaps.png', dpi=300, bbox_inches='tight')
        print(f"  Saved to: {PLOTS_DIR / 'storkey_heatmaps.png'}")

# %% [markdown]
# ## Figure 4: 90% Threshold Capacity Curves
#
# For each (rho, informed_fraction) cell, plot three curves showing
# how maximum capacity (patterns achieving >= 90% success) scales
# with network size.

# %% Figure 4: Capacity curves
print("\nGenerating capacity comparison figure...")

fig_cap, axes_cap = plt.subplots(5, 4, figsize=(18, 20), sharex=True, sharey=True)
fig_cap.suptitle(f'Storage Capacity at {int(SUCCESS_THRESHOLD*100)}% Success Threshold',
                  fontsize=18, fontweight='bold')

for row_idx, rho in enumerate(CORRELATIONS):
    for col_idx, informed in enumerate(INFORMED_FRACTIONS):
        ax = axes_cap[row_idx, col_idx]

        # AR method (use sleep success, same for all informed fractions)
        if chn_df is not None:
            ar_subset = filter_by_rho(chn_df, rho)
            if len(ar_subset) > 0:
                ar_caps = find_max_capacity(ar_subset, SUCCESS_THRESHOLD,
                                            success_col='all_recovered_before_spurious')
                x_ar = sorted(ar_caps.keys())
                y_ar = [ar_caps[x] for x in x_ar]
                ax.plot(x_ar, y_ar, 'o-', color=METHOD_COLORS['AR'],
                       label='AR (CHN)', linewidth=2.5, markersize=6)

        # Hebbian method
        if hebbian_df is not None:
            heb_subset = filter_by_rho(hebbian_df, rho)
            heb_subset = filter_by_informed(heb_subset, informed)
            if len(heb_subset) > 0:
                heb_caps = find_max_capacity(heb_subset, SUCCESS_THRESHOLD,
                                             success_col='query_success_rate')
                x_heb = sorted(heb_caps.keys())
                y_heb = [heb_caps[x] for x in x_heb]
                ax.plot(x_heb, y_heb, 's--', color=METHOD_COLORS['Hebbian'],
                       label='Hebbian', linewidth=2, markersize=5)

        # Storkey method
        if storkey_df is not None:
            stk_subset = filter_by_rho(storkey_df, rho)
            stk_subset = filter_by_informed(stk_subset, informed)
            if len(stk_subset) > 0:
                stk_caps = find_max_capacity(stk_subset, SUCCESS_THRESHOLD,
                                             success_col='query_success_rate')
                x_stk = sorted(stk_caps.keys())
                y_stk = [stk_caps[x] for x in x_stk]
                ax.plot(x_stk, y_stk, '^:', color=METHOD_COLORS['Storkey'],
                       label='Storkey', linewidth=2, markersize=5)

        # Formatting
        if row_idx == 0:
            ax.set_title(f'{int(informed*100)}% informed', fontweight='bold')
        if col_idx == 0:
            ax.set_ylabel(rf'$\rho={rho}$' + '\nMax patterns')
        if row_idx == len(CORRELATIONS) - 1:
            ax.set_xlabel('Network size')

        # Add legend to top-left cell only
        if row_idx == 0 and col_idx == 0:
            ax.legend(loc='upper left', frameon=True)

        # Grid
        ax.grid(True, alpha=0.3, linestyle='--')

        # Ensure y-axis starts at 0
        ax.set_ylim(bottom=0)

plt.tight_layout(rect=[0, 0, 1, 0.96])

if SAVE_PLOTS:
    fig_cap.savefig(PLOTS_DIR / 'capacity_comparison.png', dpi=300, bbox_inches='tight')
    print(f"  Saved to: {PLOTS_DIR / 'capacity_comparison.png'}")

# %% Done
print("\n" + "=" * 70)
print("VISUALIZATION COMPLETE")
print("=" * 70)
print(f"\nFigures saved to: {PLOTS_DIR}")
print("\nFiles created:")
print("  - ar_heatmaps.png")
print("  - hebbian_heatmaps.png")
print("  - storkey_heatmaps.png")
print("  - capacity_comparison.png")

plt.show()

# %%
