# %% [markdown]
# # McCallum Comparison Visualization
#
# This notebook creates publication-ready comparison figures for four memory
# capacity methods:
#
# 1. **McCallum Pseudorehearsal** (DHN) - Delta learning with probing
# 2. **AR / Continuous Incorporation** (CHN) - Sleep consolidation
# 3. **Hebbian** (DHN) - One-shot outer product learning
# 4. **Storkey** (DHN) - Incremental local field correction
#
# ## Output
# - `scripts/plots/mccallum_comparison.png` - M* vs N for each rho (column layout)
# - `scripts/plots/mccallum_comparison_byN.png` - M* vs rho for each N

# %% [markdown]
# ## Figure Interpretation Guide
#
# ### What the plots show
# - **Lines**: Memory capacity $M^*$ (number of patterns that can be reliably stored
#   and retrieved) as a function of network size $N$
# - **Shaded regions**: Standard deviation ($\pm 1\sigma$) across simulation seeds,
#   indicating variability in capacity measurements
#
# ### Key metrics
# - **$M^*$ (theta=0.9)**: Maximum number of patterns $M$ such that at least 90% of
#   simulations achieved perfect retrieval of all $M$ patterns using 50% partial cues
# - **$\rho$ (correlation)**: Pattern similarity parameter. $\rho=0$ means uncorrelated
#   patterns; higher $\rho$ means more similar patterns (harder to distinguish)

# %% [markdown]
# ## Imports and Configuration

# %%
import sys
from pathlib import Path

# Get absolute path to parent scripts directory
_THIS_DIR = Path(__file__).resolve().parent if '__file__' in dir() else Path.cwd()
_SCRIPTS_DIR = _THIS_DIR.parent
sys.path.insert(0, str(_SCRIPTS_DIR))

from utils import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np

# Publication-quality plot settings
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'font.family': 'sans-serif',
    'axes.linewidth': 1.2,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
})

# Method colors - distinct and colorblind-friendly
METHOD_COLORS = {
    'McCallum':  '#E67E22',  # Orange
    'AR':        '#2C3E50',  # Dark blue-gray
    'Hebbian':   '#922B21',  # Dark burgundy
    'Storkey':   '#1E8449',  # Dark forest green
}

METHOD_MARKERS = {
    'McCallum':  'o',
    'AR':        's',
    'Hebbian':   '^',
    'Storkey':   'D',
}

METHOD_LABELS = {
    'McCallum':  'McCallum (Pseudorehearsal)',
    'AR':        'AR (Sleep Consolidation)',
    'Hebbian':   'Hebbian',
    'Storkey':   'Storkey',
}

# Parameters from experimental grid
NETWORK_SIZES = [50, 100, 150, 200, 250]
RHO_VALUES = [0.0, 0.2, 0.4, 0.5, 0.6]
THETA = 0.9

# Data directories
DATA_BASE = DATA_DIR / "mccallum_results"

# %% [markdown]
# ## Load Results

# %%
def load_M_star_summary(method_dir: Path, method_name: str) -> pd.DataFrame:
    """Load M* summary CSV for a method."""
    summary_path = method_dir / "M_star_summary.csv"
    if summary_path.exists():
        df = pd.read_csv(summary_path)
        df['method'] = method_name
        return df
    else:
        # Try to compute from all_simulation_data.csv
        all_data_path = method_dir / "all_simulation_data.csv"
        if all_data_path.exists():
            print(f"Computing M* summary for {method_name}...")
            all_df = pd.read_csv(all_data_path)
            return compute_M_star_summary(all_df, method_name)
        else:
            print(f"Warning: No data found for {method_name} at {method_dir}")
            return pd.DataFrame()


def compute_M_star_summary(df: pd.DataFrame, method_name: str) -> pd.DataFrame:
    """Compute M* summary from raw simulation data.

    Handles different data formats:
    - McCallum/AR: has 'M_star' column directly
    - DHN (Hebbian/Storkey): has per-pattern 'recovered' column that needs aggregation
    """
    results = []

    # Check if this is DHN-style per-pattern data
    if 'recovered' in df.columns and 'pattern_idx' in df.columns:
        # Aggregate per-pattern results to per-(N, rho, seed, num_patterns) level
        agg = df.groupby(['network_size', 'rho', 'seed', 'num_patterns']).agg({
            'recovered': ['sum', 'count']
        }).reset_index()
        agg.columns = ['network_size', 'rho', 'seed', 'num_patterns', 'recovered_sum', 'total']
        agg['all_recovered'] = (agg['recovered_sum'] == agg['total'])

        for N in agg['network_size'].unique():
            for rho in agg['rho'].unique():
                M_star_list = []

                for seed in agg['seed'].unique():
                    subset = agg[
                        (agg['network_size'] == N) &
                        (agg['rho'] == rho) &
                        (agg['seed'] == seed)
                    ].sort_values('num_patterns')

                    # Find max num_patterns where ALL patterns were recovered
                    M_star_s = 0
                    for _, row in subset.iterrows():
                        if row['all_recovered']:
                            M_star_s = int(row['num_patterns'])

                    M_star_list.append(M_star_s)

                if len(M_star_list) > 0:
                    M_star_list = np.array(M_star_list)
                    # Compute M* using theta threshold
                    max_M = max(M_star_list)
                    M_star = 0
                    for M in range(int(max_M), -1, -1):
                        fraction = sum(1 for m in M_star_list if m >= M) / len(M_star_list)
                        if fraction >= THETA:
                            M_star = M
                            break

                    results.append({
                        'N': N,
                        'rho': rho,
                        'M_star': M_star,
                        'mean_M_star': M_star_list.mean(),
                        'std_M_star': M_star_list.std(),
                        'num_sims': len(M_star_list),
                        'method': method_name
                    })

    else:
        # McCallum/AR style: already has M_star or all_queries_passed per simulation
        for N in df['network_size'].unique():
            for rho in df['rho'].unique():
                subset = df[(df['network_size'] == N) & (df['rho'] == rho)]

                if 'M_star' in subset.columns:
                    M_star_list = subset['M_star'].values
                elif 'seed' in subset.columns:
                    # Need to aggregate by seed
                    M_star_list = []
                    for seed in subset['seed'].unique():
                        seed_subset = subset[subset['seed'] == seed]
                        max_success = 0
                        for _, row in seed_subset.sort_values('num_patterns').iterrows():
                            if 'all_queries_passed' in row and row['all_queries_passed'] > 0.5:
                                max_success = int(row['num_patterns'])
                            elif 'sleep_success' in row and row['sleep_success'] > 0.5:
                                max_success = int(row['num_patterns'])
                        M_star_list.append(max_success)
                    M_star_list = np.array(M_star_list)
                else:
                    continue

                if len(M_star_list) == 0:
                    continue

                # Compute M* using theta threshold
                max_M = max(M_star_list)
                M_star = 0
                for M in range(int(max_M), -1, -1):
                    fraction = sum(1 for m in M_star_list if m >= M) / len(M_star_list)
                    if fraction >= THETA:
                        M_star = M
                        break

                results.append({
                    'N': N,
                    'rho': rho,
                    'M_star': M_star,
                    'mean_M_star': M_star_list.mean(),
                    'std_M_star': M_star_list.std(),
                    'num_sims': len(M_star_list),
                    'method': method_name
                })

    return pd.DataFrame(results)

# %%
# Load all method results
mccallum_df = load_M_star_summary(DATA_BASE / "mccallum", "McCallum")
ar_df = load_M_star_summary(DATA_BASE / "ar", "AR")
hebbian_df = load_M_star_summary(DATA_BASE / "hebbian", "Hebbian")
storkey_df = load_M_star_summary(DATA_BASE / "storkey", "Storkey")

# Combine
all_results = pd.concat([mccallum_df, ar_df, hebbian_df, storkey_df], ignore_index=True)

if len(all_results) > 0:
    print(f"Loaded {len(all_results)} results")
    print("\nMethods found:", list(all_results['method'].unique()))
    print("\nRho values found:", sorted(all_results['rho'].unique()))
    print("\nSample data:")
    print(all_results.head(10))
else:
    print("No data loaded! Run the simulation scripts first.")

# %% [markdown]
# ## Main Comparison Figure (Column Layout)
#
# ### Visual Guide
# - **Solid lines with markers**: $M^*$ values (90th percentile capacity)
# - **Shaded bands**: $\pm 1$ standard deviation across seeds, showing variability
#   in capacity measurements. Wider bands indicate less consistent performance.

# %%
def plot_comparison_column(all_results: pd.DataFrame, save_path: Path = None):
    """
    Create column layout: one subplot per rho value showing M* vs N for all methods.
    Publication-quality formatting with shared axes and rho labels on right.
    """
    if len(all_results) == 0:
        print("No data to plot!")
        return

    # Get actual rho values from data
    rho_values = sorted(all_results['rho'].unique())
    n_rho = len(rho_values)

    fig, axes = plt.subplots(n_rho, 1, figsize=(8, 3 * n_rho), sharex=True, sharey=True)
    if n_rho == 1:
        axes = [axes]

    for ax, rho in zip(axes, rho_values):
        subset = all_results[all_results['rho'] == rho]

        for method in ['Storkey', 'AR', 'McCallum', 'Hebbian']:
            method_data = subset[subset['method'] == method]
            if len(method_data) == 0:
                continue

            method_data = method_data.sort_values('N')
            N_vals = method_data['N'].values
            M_star_vals = method_data['M_star'].values

            ax.plot(N_vals, M_star_vals,
                    marker=METHOD_MARKERS[method],
                    color=METHOD_COLORS[method],
                    linewidth=2.5,
                    markersize=9,
                    label=METHOD_LABELS[method])

            # Add shaded region for std deviation
            if 'std_M_star' in method_data.columns:
                std_vals = method_data['std_M_star'].values
                ax.fill_between(N_vals,
                               M_star_vals - std_vals,
                               M_star_vals + std_vals,
                               color=METHOD_COLORS[method],
                               alpha=0.15)

        # Add rho label on right side
        ax.text(1.02, 0.5, f'$\\rho = {rho}$',
                transform=ax.transAxes,
                fontsize=15,
                fontweight='bold',
                va='center',
                ha='left')

        ax.set_xticks(NETWORK_SIZES)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(NETWORK_SIZES[0] - 10, NETWORK_SIZES[-1] + 10)

    # Shared labels (only on edges)
    axes[-1].set_xlabel('Network Size $N$', fontsize=16)
    fig.text(0.02, 0.5, '$M^*$', va='center', rotation='vertical', fontsize=18, fontweight='bold')

    # Single legend at top
    handles = [Line2D([0], [0], color=METHOD_COLORS[m], marker=METHOD_MARKERS[m],
                      linewidth=2.5, markersize=9, label=METHOD_LABELS[m])
               for m in ['Storkey', 'AR', 'McCallum', 'Hebbian']]
    fig.legend(handles=handles, loc='upper center', ncol=2,
               bbox_to_anchor=(0.5, 1.02), fontsize=12, frameon=True,
               fancybox=True, shadow=False)

    plt.tight_layout()
    plt.subplots_adjust(left=0.12, right=0.88, hspace=0.08)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved figure to: {save_path}")

    plt.show()

# %%
# Create and save main figure
output_path = SCRIPT_DIR / "plots" / "mccallum_comparison.png"
output_path.parent.mkdir(exist_ok=True)

plot_comparison_column(all_results, output_path)

# %% [markdown]
# ## Alternative: M* vs Rho (Column Layout by N)
#
# This view shows how capacity degrades with increasing pattern correlation
# for each network size.

# %%
def plot_comparison_by_N_column(all_results: pd.DataFrame, save_path: Path = None):
    """
    Create column layout: one subplot per N value showing M* vs rho for all methods.
    """
    if len(all_results) == 0:
        print("No data to plot!")
        return

    # Get actual N values from data
    N_values = sorted(all_results['N'].unique())
    n_N = len(N_values)

    fig, axes = plt.subplots(n_N, 1, figsize=(8, 2.8 * n_N), sharex=True, sharey=True)
    if n_N == 1:
        axes = [axes]

    for ax, N in zip(axes, N_values):
        subset = all_results[all_results['N'] == N]

        for method in ['Storkey', 'AR', 'McCallum', 'Hebbian']:
            method_data = subset[subset['method'] == method]
            if len(method_data) == 0:
                continue

            method_data = method_data.sort_values('rho')
            rho_vals = method_data['rho'].values
            M_star_vals = method_data['M_star'].values

            ax.plot(rho_vals, M_star_vals,
                    marker=METHOD_MARKERS[method],
                    color=METHOD_COLORS[method],
                    linewidth=2.5,
                    markersize=9,
                    label=METHOD_LABELS[method])

            # Add shaded region for std deviation
            if 'std_M_star' in method_data.columns:
                std_vals = method_data['std_M_star'].values
                ax.fill_between(rho_vals,
                               M_star_vals - std_vals,
                               M_star_vals + std_vals,
                               color=METHOD_COLORS[method],
                               alpha=0.15)

        # Add N label on right side
        ax.text(1.02, 0.5, f'$N = {N}$',
                transform=ax.transAxes,
                fontsize=15,
                fontweight='bold',
                va='center',
                ha='left')

        ax.grid(True, alpha=0.3, linestyle='--')

    # Shared labels
    axes[-1].set_xlabel('Pattern Correlation $\\rho$', fontsize=16)
    fig.text(0.02, 0.5, '$M^*$', va='center', rotation='vertical', fontsize=18, fontweight='bold')

    # Single legend at top
    handles = [Line2D([0], [0], color=METHOD_COLORS[m], marker=METHOD_MARKERS[m],
                      linewidth=2.5, markersize=9, label=METHOD_LABELS[m])
               for m in ['Storkey', 'AR', 'McCallum', 'Hebbian']]
    fig.legend(handles=handles, loc='upper center', ncol=2,
               bbox_to_anchor=(0.5, 1.02), fontsize=12, frameon=True,
               fancybox=True, shadow=False)

    plt.tight_layout()
    plt.subplots_adjust(left=0.12, right=0.88, hspace=0.08)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved figure to: {save_path}")

    plt.show()

# %%
output_path_byN = SCRIPT_DIR / "plots" / "mccallum_comparison_byN.png"
plot_comparison_by_N_column(all_results, output_path_byN)

# %% [markdown]
# ## Summary Statistics Table
#
# The table below shows the computed $M^*$ values for each method and configuration.
# $M^*$ is defined as the maximum $M$ such that at least $\theta = 90\%$ of simulations
# achieved perfect retrieval of all $M$ patterns.

# %%
def print_summary_table(all_results: pd.DataFrame):
    """Print formatted summary table."""
    if len(all_results) == 0:
        print("No data!")
        return

    print("\n" + "=" * 80)
    print("CAPACITY COMPARISON SUMMARY")
    print(f"M* defined with theta = {THETA} (90th percentile)")
    print("=" * 80)

    # Pivot table: rows = (N, rho), columns = methods
    pivot = all_results.pivot_table(
        values='M_star',
        index=['N', 'rho'],
        columns='method',
        aggfunc='first'
    )

    # Reorder columns
    cols = [c for c in ['Storkey', 'AR', 'McCallum', 'Hebbian'] if c in pivot.columns]
    pivot = pivot[cols]

    print(pivot.to_string())

    print("\n" + "-" * 80)
    print("CAPACITY SCALING (M*/N ratio at rho=0)")
    print("-" * 80)

    rho0_data = all_results[all_results['rho'] == 0.0]
    if len(rho0_data) > 0:
        for method in cols:
            method_data = rho0_data[rho0_data['method'] == method]
            if len(method_data) > 0:
                # Linear fit to get scaling coefficient
                ratios = method_data['M_star'] / method_data['N']
                avg_ratio = ratios.mean()
                print(f"  {method:25s}: {avg_ratio:.3f}")

    print("-" * 80)
    print("Theoretical (uncorrelated): Hebbian ~0.138, Storkey ~0.42")

# %%
print_summary_table(all_results)

# %% [markdown]
# ## Heatmap Visualization
#
# Each heatmap shows $M^*$ as a function of network size $N$ (columns) and
# pattern correlation $\rho$ (rows). Darker colors indicate higher capacity.
# The numerical values are annotated in each cell.

# %%
def plot_method_heatmaps(all_results: pd.DataFrame, save_path: Path = None):
    """
    Create 2x2 heatmap grid showing M* for each method.
    """
    if len(all_results) == 0:
        print("No data to plot!")
        return

    # Get actual values from data
    N_values = sorted(all_results['N'].unique())
    rho_values = sorted(all_results['rho'].unique())

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    methods = ['Storkey', 'AR', 'McCallum', 'Hebbian']

    for ax, method in zip(axes.flat, methods):
        method_data = all_results[all_results['method'] == method]
        if len(method_data) == 0:
            ax.set_title(f'{method} (no data)')
            ax.axis('off')
            continue

        # Create pivot table
        pivot = method_data.pivot_table(
            values='M_star',
            index='rho',
            columns='N',
            aggfunc='first'
        )

        # Plot heatmap
        im = ax.imshow(pivot.values, cmap='YlOrRd', aspect='auto',
                       vmin=0, vmax=all_results['M_star'].max())

        # Set ticks
        ax.set_xticks(range(len(N_values)))
        ax.set_xticklabels(N_values, fontsize=12)
        ax.set_yticks(range(len(rho_values)))
        ax.set_yticklabels([f'{r:.1f}' for r in rho_values], fontsize=12)

        ax.set_xlabel('$N$', fontsize=14)
        ax.set_ylabel('$\\rho$', fontsize=14)
        ax.set_title(METHOD_LABELS[method], color=METHOD_COLORS[method],
                     fontweight='bold', fontsize=14)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('$M^*$', fontsize=12)

        # Annotate with values
        for i in range(len(rho_values)):
            for j in range(len(N_values)):
                if i < pivot.shape[0] and j < pivot.shape[1]:
                    val = pivot.iloc[i, j]
                    if not np.isnan(val):
                        ax.text(j, i, f'{int(val)}', ha='center', va='center',
                               color='white' if val > pivot.values.max() * 0.5 else 'black',
                               fontsize=11, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved figure to: {save_path}")

    plt.show()

# %%
output_path_heatmap = SCRIPT_DIR / "plots" / "mccallum_comparison_heatmap.png"
plot_method_heatmaps(all_results, output_path_heatmap)

# %% [markdown]
# ## Key Findings
#
# ### Interpretation of Results
#
# 1. **Storkey outperforms all methods** at low correlation ($\rho \leq 0.2$),
#    with capacity scaling approximately as $M^* \approx 0.12N$
#
# 2. **High correlation degrades all methods**: At $\rho \geq 0.5$, capacity
#    no longer scales with $N$ - all methods plateau at $M^* \approx 2-5$
#
# 3. **AR (Sleep Consolidation)** shows intermediate performance, benefiting
#    from continuous activations but limited by the strict spurious-as-failure criterion
#
# 4. **McCallum Pseudorehearsal** shows similar scaling to Hebbian, as expected
#    from the discrete network foundation
#
# ### Shaded Regions
#
# The shaded bands represent $\pm 1$ standard deviation across simulation seeds.
# Wider bands indicate:
# - More variability in capacity measurements
# - Less reliable/consistent performance
# - Sensitivity to initial conditions or pattern sets
