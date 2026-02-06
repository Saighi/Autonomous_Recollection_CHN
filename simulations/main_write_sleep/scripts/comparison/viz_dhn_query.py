# %% [markdown]
# # DHN Query Results Visualization
#
# This script visualizes Discrete Hopfield Network query results comparing
# Hebbian vs Storkey learning rules across different parameter combinations.
#
# Creates:
# 1. Heatmaps: Recovery rate for (network_size × num_patterns) - separated by rho
# 2. Capacity curves: Max patterns at 20%, 50%, 90% thresholds for Hebbian vs Storkey

# %% Imports
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for script execution
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from pathlib import Path
import sys

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import load_results, DATA_DIR

# %%=========================================================================
# CONFIGURATION
# ==========================================================================

# Data paths
HEBBIAN_DIR = DATA_DIR / "query_results" / "comparison_dhn_hebbian_query"
STORKEY_DIR = DATA_DIR / "query_results" / "comparison_dhn_storkey_query"

# Parameters
CORRELATIONS = [0.0, 0.2, 0.4, 0.6, 0.8]
INFORMED_FRACTIONS = [0.9, 0.75, 0.5, 0.25, 0.1]
THRESHOLDS = [0.2, 0.5, 0.9]

# Colors for learning rules
COLORS = {'Hebbian': '#ff7f0e', 'Storkey': '#2ca02c'}

# Output settings
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


def aggregate_to_simulation_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-pattern rows to simulation level.

    The CSV has one row per pattern per simulation. We take the first row
    per sim_ID since query_success_rate and query_avg_steps are already
    simulation-level aggregates.
    """
    # Group by sim_ID and take first row (all rows have same sim-level values)
    sim_cols = ['sim_ID', 'query_success_rate', 'query_avg_steps',
                'network_size', 'num_patterns', 'rho', 'informed_fraction']

    # Filter to columns that exist
    available_cols = [c for c in sim_cols if c in df.columns]

    # Get unique simulations
    df_sim = df[available_cols].drop_duplicates(subset=['sim_ID'])

    return df_sim


def compute_capacity(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Compute max patterns achieving >= threshold recovery rate per network_size.

    Returns DataFrame with columns: network_size, informed_fraction, max_patterns
    """
    results = []

    for (net_size, inf_frac), group in df.groupby(['network_size', 'informed_fraction']):
        # Find max num_patterns where recovery_rate >= threshold
        passing = group[group['query_success_rate'] >= threshold]
        if len(passing) > 0:
            max_patterns = passing['num_patterns'].max()
        else:
            max_patterns = 0

        results.append({
            'network_size': net_size,
            'informed_fraction': inf_frac,
            'max_patterns': max_patterns
        })

    return pd.DataFrame(results)


# %% [markdown]
# ## Load Data

# %% Load data
print("=" * 70)
print("LOADING DATA")
print("=" * 70)

hebbian_df = load_results(HEBBIAN_DIR)
storkey_df = load_results(STORKEY_DIR)

print(f"\nHebbian data: {len(hebbian_df)} rows")
print(f"Storkey data: {len(storkey_df)} rows")

# Aggregate to simulation level
hebbian_sim = aggregate_to_simulation_level(hebbian_df)
storkey_sim = aggregate_to_simulation_level(storkey_df)

print(f"\nHebbian simulations: {len(hebbian_sim)}")
print(f"Storkey simulations: {len(storkey_sim)}")

# Get unique values for axes
all_net_sizes = np.sort(hebbian_sim['network_size'].unique())
all_num_patterns = np.sort(hebbian_sim['num_patterns'].unique())
all_rhos = np.sort(hebbian_sim['rho'].unique())
all_inf_fracs = np.sort(hebbian_sim['informed_fraction'].unique())[::-1]  # Descending

print(f"\nNetwork sizes: {list(all_net_sizes)}")
print(f"Num patterns: {len(all_num_patterns)} values from {all_num_patterns.min()} to {all_num_patterns.max()}")
print(f"Rho values: {list(all_rhos)}")
print(f"Informed fractions: {list(all_inf_fracs)}")

# %% [markdown]
# ## Part 1: Heatmaps (separated by rho)
#
# For each rho value:
# - 2 rows: Recovery rate, Avg convergence steps
# - 5 columns: informed_fractions [0.9, 0.75, 0.5, 0.25, 0.1]

# %% Heatmap plotting function
def plot_heatmap_figure(df: pd.DataFrame, rho: float, learning_rule: str):
    """
    Plot heatmap figure for a single rho value.

    Creates 2×5 grid:
    - Row 1: Recovery rate heatmaps
    - Row 2: Avg convergence steps heatmaps
    - Columns: informed_fractions
    """
    sub = df[df['rho'] == rho]

    # Colormap with grey for missing data
    default_cmap_name = plt.rcParams["image.cmap"]
    cmap = mpl.cm.get_cmap(default_cmap_name).copy()
    cmap.set_bad(color="lightgrey")

    n_cols = len(INFORMED_FRACTIONS)

    # Calculate global max for convergence steps
    global_max_steps = sub['query_avg_steps'].max()

    # Create figure
    r = 1.1
    fig_width = max(9, 3 * n_cols) / r
    fig, axes = plt.subplots(2, n_cols, figsize=(fig_width, 8 / r),
                             sharex=True, sharey=True, squeeze=False)

    for i, inf_frac in enumerate(INFORMED_FRACTIONS):
        sub_inf = sub[sub['informed_fraction'] == inf_frac]

        # Row 1: Recovery rate (0-100%)
        pivot_recovery = sub_inf.pivot_table(
            values='query_success_rate',
            index='num_patterns',
            columns='network_size',
        )
        # Reindex to ensure all values present
        pivot_recovery = pivot_recovery.reindex(
            index=all_num_patterns, columns=all_net_sizes
        )

        masked_recovery = np.ma.masked_invalid(pivot_recovery.values * 100)
        im1 = axes[0, i].imshow(masked_recovery, vmin=0, vmax=100, cmap=cmap)
        axes[0, i].set_title(rf"$f={inf_frac}$")
        axes[0, i].invert_yaxis()
        axes[0, i].grid(False)

        # Row 2: Avg convergence steps
        pivot_steps = sub_inf.pivot_table(
            values='query_avg_steps',
            index='num_patterns',
            columns='network_size',
        )
        pivot_steps = pivot_steps.reindex(
            index=all_num_patterns, columns=all_net_sizes
        )

        masked_steps = np.ma.masked_invalid(pivot_steps.values)
        im2 = axes[1, i].imshow(masked_steps, vmin=0, vmax=global_max_steps, cmap=cmap)
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

    # Add colorbars
    cbar1_ax = fig.add_axes([0.92, 0.56, 0.02, 0.3])
    cbar1 = fig.colorbar(im1, cax=cbar1_ax)
    cbar1.set_ticks(np.linspace(0, 100, 5))
    cbar1.set_ticklabels([f'{int(val)}' for val in np.linspace(0, 100, 5)])
    cbar1.set_label(r'Recovery \%')

    cbar2_ax = fig.add_axes([0.92, 0.14, 0.02, 0.3])
    cbar2 = fig.colorbar(im2, cax=cbar2_ax)
    cbar2.set_ticks(np.linspace(0, global_max_steps, 5))
    cbar2.set_ticklabels([f'{val:.1f}' for val in np.linspace(0, global_max_steps, 5)])
    cbar2.set_label('Avg steps')

    # Add axis labels
    fig.text(0.51, 0.04, 'Network size', ha='center', va='center')
    fig.text(0.04, 0.49, 'Nb stored patterns', ha='left', va='center', rotation=90)

    # Add title
    fig.suptitle(rf'{learning_rule} Learning ($\rho={rho}$)', y=0.98)

    # Save
    if SAVE_PLOTS:
        OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
        filename = f"dhn_{learning_rule.lower()}_rho_{rho}.png"
        output_path = OUTPUT_DIR / filename
        plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
        print(f"Saved: {output_path}")

    plt.close(fig)


# %% Generate Hebbian heatmaps
print("\n" + "=" * 70)
print("GENERATING HEBBIAN HEATMAPS")
print("=" * 70)

for rho in CORRELATIONS:
    plot_heatmap_figure(hebbian_sim, rho, 'Hebbian')

# %% Generate Storkey heatmaps
print("\n" + "=" * 70)
print("GENERATING STORKEY HEATMAPS")
print("=" * 70)

for rho in CORRELATIONS:
    plot_heatmap_figure(storkey_sim, rho, 'Storkey')

# %% [markdown]
# ## Part 2: Capacity Curves (separated by rho)
#
# For each rho value, create a figure with:
# - 3 rows (thresholds: 20%, 50%, 90%)
# - 5 columns (informed_fractions)
# - Each subplot: 2 curves (Hebbian orange, Storkey green)

# %% Capacity curve plotting function
def plot_capacity_figure(hebbian_df: pd.DataFrame, storkey_df: pd.DataFrame, rho: float):
    """
    Plot capacity curves for a single rho value.

    Creates 3×5 grid comparing Hebbian vs Storkey at different thresholds.
    """
    heb_sub = hebbian_df[hebbian_df['rho'] == rho]
    stk_sub = storkey_df[storkey_df['rho'] == rho]

    n_rows = len(THRESHOLDS)
    n_cols = len(INFORMED_FRACTIONS)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 10),
                             sharex=True, sharey=True, squeeze=False)

    # Compute global max for y-axis
    global_max_patterns = 0

    for threshold in THRESHOLDS:
        for inf_frac in INFORMED_FRACTIONS:
            # Hebbian
            heb_inf = heb_sub[heb_sub['informed_fraction'] == inf_frac]
            for net_size, group in heb_inf.groupby('network_size'):
                passing = group[group['query_success_rate'] >= threshold]
                if len(passing) > 0:
                    global_max_patterns = max(global_max_patterns, passing['num_patterns'].max())

            # Storkey
            stk_inf = stk_sub[stk_sub['informed_fraction'] == inf_frac]
            for net_size, group in stk_inf.groupby('network_size'):
                passing = group[group['query_success_rate'] >= threshold]
                if len(passing) > 0:
                    global_max_patterns = max(global_max_patterns, passing['num_patterns'].max())

    # Plot each subplot
    for row_idx, threshold in enumerate(THRESHOLDS):
        for col_idx, inf_frac in enumerate(INFORMED_FRACTIONS):
            ax = axes[row_idx, col_idx]

            # Compute capacity for Hebbian
            heb_inf = heb_sub[heb_sub['informed_fraction'] == inf_frac]
            heb_capacity = []
            for net_size in all_net_sizes:
                group = heb_inf[heb_inf['network_size'] == net_size]
                passing = group[group['query_success_rate'] >= threshold]
                max_p = passing['num_patterns'].max() if len(passing) > 0 else 0
                heb_capacity.append(max_p)

            # Compute capacity for Storkey
            stk_inf = stk_sub[stk_sub['informed_fraction'] == inf_frac]
            stk_capacity = []
            for net_size in all_net_sizes:
                group = stk_inf[stk_inf['network_size'] == net_size]
                passing = group[group['query_success_rate'] >= threshold]
                max_p = passing['num_patterns'].max() if len(passing) > 0 else 0
                stk_capacity.append(max_p)

            # Plot
            ax.plot(all_net_sizes, heb_capacity, 'o-', color=COLORS['Hebbian'],
                   label='Hebbian', markersize=4)
            ax.plot(all_net_sizes, stk_capacity, 's-', color=COLORS['Storkey'],
                   label='Storkey', markersize=4)

            ax.grid(True, alpha=0.3)

            # Add title for top row
            if row_idx == 0:
                ax.set_title(rf'$f={inf_frac}$')

            # Add threshold label for first column
            if col_idx == 0:
                ax.set_ylabel(rf'{int(threshold*100)}\% threshold')

            # Add legend only to first subplot
            if row_idx == 0 and col_idx == 0:
                ax.legend(loc='upper left', fontsize=12)

    # Set shared limits
    for ax in axes.flat:
        ax.set_ylim(0, global_max_patterns * 1.05)
        ax.set_xlim(all_net_sizes.min() * 0.9, all_net_sizes.max() * 1.05)

    # Add axis labels
    fig.text(0.5, 0.02, 'Network size', ha='center', va='center', fontsize=18)
    fig.text(0.02, 0.5, 'Max patterns', ha='center', va='center',
             rotation=90, fontsize=18)

    # Add title
    fig.suptitle(rf'Capacity Comparison: Hebbian vs Storkey ($\rho={rho}$)',
                 y=0.98, fontsize=20)

    plt.tight_layout(rect=[0.04, 0.04, 1, 0.96])

    # Save
    if SAVE_PLOTS:
        OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
        filename = f"dhn_capacity_rho_{rho}.png"
        output_path = OUTPUT_DIR / filename
        plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
        print(f"Saved: {output_path}")

    plt.close(fig)


# %% Generate capacity curves
print("\n" + "=" * 70)
print("GENERATING CAPACITY CURVES")
print("=" * 70)

for rho in CORRELATIONS:
    plot_capacity_figure(hebbian_sim, storkey_sim, rho)

# %% [markdown]
# ## Summary

# %% Summary statistics
print("\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)

print("\nOverall recovery rates:")
print(f"  Hebbian: {hebbian_sim['query_success_rate'].mean()*100:.1f}%")
print(f"  Storkey: {storkey_sim['query_success_rate'].mean()*100:.1f}%")

print("\nRecovery rates by rho:")
for rho in CORRELATIONS:
    heb_rate = hebbian_sim[hebbian_sim['rho'] == rho]['query_success_rate'].mean() * 100
    stk_rate = storkey_sim[storkey_sim['rho'] == rho]['query_success_rate'].mean() * 100
    print(f"  rho={rho}: Hebbian={heb_rate:.1f}%, Storkey={stk_rate:.1f}%")

# %% Final summary
print("\n" + "=" * 70)
print("VISUALIZATION COMPLETE!")
print("=" * 70)

if SAVE_PLOTS:
    print(f"\nPlots saved to: {OUTPUT_DIR.absolute()}")
    print(f"\nGenerated files:")
    for rho in CORRELATIONS:
        print(f"  - dhn_hebbian_rho_{rho}.png")
        print(f"  - dhn_storkey_rho_{rho}.png")
        print(f"  - dhn_capacity_rho_{rho}.png")
    print(f"\nTotal: {len(CORRELATIONS) * 3} files")

print("=" * 70 + "\n")

# %%
