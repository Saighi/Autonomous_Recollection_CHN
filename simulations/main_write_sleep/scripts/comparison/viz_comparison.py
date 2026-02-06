# %% [markdown]
# # CHN vs DHN Comparison Visualization
#
# ## Overview
#
# This script generates publication-ready figures comparing Autonomous Recollection (AR)
# in Continuous Hopfield Networks (CHN) with Hebbian and Storkey learning rules in
# Discrete Hopfield Networks (DHN).
#
# ## Methods Compared
#
# 1. **AR (CHN)** - Autonomous Retrieval with sleep-based memory consolidation
#    - No external cues required
#    - Success measured by `all_recovered_before_spurious`
#
# 2. **Hebbian (DHN)** - Classic outer-product learning rule
#    - Partial cue retrieval with `informed_fraction`
#    - Success measured by `query_success_rate`
#
# 3. **Storkey (DHN)** - Local-field corrected learning rule
#    - Better capacity than Hebbian (~0.42N vs ~0.138N theoretical)
#    - Same query mechanism as Hebbian
#
# ## Figures Generated
#
# - **Figure Set A**: CHN heatmaps with capacity boundary stars (one per rho)
# - **Figure Set B**: Capacity comparison grids (5x5 for each threshold: 90%, 80%, 50%)
# - **Figure Set C**: Theoretical comparison at rho=0

# %% Imports
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_final_results, DATA_DIR

# %% =========================================================================
# CONFIGURATION
# =============================================================================

# Output settings
OUTPUT_DIR = Path(__file__).parent.parent / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SAVE_PLOTS = True
DPI = 300

# Method colors
METHOD_COLORS = {
    'AR': '#1f77b4',       # Blue
    'Hebbian': '#ff7f0e',  # Orange
    'Storkey': '#2ca02c',  # Green
}

# Rho values for comparison
RHO_VALUES = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8]

# Informed fractions for DHN
INFORMED_FRACTIONS = [0.1, 0.25, 0.5, 0.75, 0.9]

# Success thresholds for capacity curves
SUCCESS_THRESHOLDS = [0.9, 0.8, 0.5]

# DHN saturation limit (max patterns tested was 99)
DHN_MAX_TESTED_PATTERNS = 99
DHN_SATURATION_THRESHOLD = 97

# %% Styling (Publication-ready, from SR_viz.py)
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
    'legend.fontsize': 16,
    'figure.titlesize': 22,
    'lines.linewidth': 2.5,
    'axes.linewidth': 1.5,
    'axes.grid': False,
    'font.weight': 'bold'
})

# %% =========================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_chn_data():
    """
    Load and merge CHN data from multiple sources.

    Sources (rho = 0.0, 0.2, 0.4, 0.6, 0.8):
    - capacity_scaling_larger_small_sleep: network_size [300-500], num_patterns [5-39]
    - capacity_scaling_larger_large_sleep: network_size [1000], num_patterns [10-49]
    - SR_correlation_compatible_large_sleep: network_size [100, 200], num_patterns [1-24]

    Sources (rho = 0.5):
    - capacity_scaling_larger_0.5_small_sleep: network_size [300-500], num_patterns [5-39]
    - capacity_scaling_larger_0.5_large_sleep: network_size [1000], num_patterns [10-49]
    - SR_correlation_0.5_compatible_large_sleep: network_size [100, 200], num_patterns [1-24]

    Returns:
        pd.DataFrame: Merged CHN data with all network sizes and pattern counts
    """
    sources = [
        # Original sources (rho = 0.0, 0.2, 0.4, 0.6, 0.8)
        "capacity_scaling_larger_small_sleep",
        "capacity_scaling_larger_large_sleep",
        "SR_correlation_compatible_large_sleep",
        # New sources for rho = 0.5
        "capacity_scaling_larger_0.5_small_sleep",
        "capacity_scaling_larger_0.5_large_sleep",
        "SR_correlation_0.5_compatible_large_sleep",  # network_size [100, 200] for rho=0.5
    ]

    dfs = []
    for source in sources:
        path = DATA_DIR / "sleep_results" / source / "final_results.csv"
        if path.exists():
            df = pd.read_csv(path)
            df['source'] = source
            dfs.append(df)
            print(f"  Loaded {len(df)} rows from {source}")
            print(f"    network_sizes: {sorted(df['network_size'].unique())}")
            print(f"    num_patterns: {df['num_patterns'].min()}-{df['num_patterns'].max()}")
        else:
            print(f"  WARNING: {source} not found at {path}")

    if not dfs:
        return None

    merged = pd.concat(dfs, ignore_index=True)
    print(f"\n  Total CHN rows: {len(merged)}")
    print(f"  Network sizes: {sorted(merged['network_size'].unique())}")
    return merged


def load_dhn_data():
    """
    Load DHN data for Hebbian and Storkey learning rules.

    Sources (rho = 0.0, 0.2, 0.4, 0.6, 0.8):
    - comparison_dhn_hebbian_query
    - comparison_dhn_storkey_query

    Sources (rho = 0.5):
    - comparison_dhn_hebbian_query_0.5
    - comparison_dhn_storkey_query_0.5

    Returns:
        tuple: (hebbian_df, storkey_df)
    """
    # Hebbian sources
    hebbian_sources = [
        "comparison_dhn_hebbian_query",      # rho = 0.0, 0.2, 0.4, 0.6, 0.8
        "comparison_dhn_hebbian_query_0.5",  # rho = 0.5
    ]

    # Storkey sources
    storkey_sources = [
        "comparison_dhn_storkey_query",      # rho = 0.0, 0.2, 0.4, 0.6, 0.8
        "comparison_dhn_storkey_query_0.5",  # rho = 0.5
    ]

    # Load and merge Hebbian data
    hebbian_dfs = []
    for source in hebbian_sources:
        path = DATA_DIR / "query_results" / source / "final_results.csv"
        if path.exists():
            df = pd.read_csv(path)
            hebbian_dfs.append(df)
            print(f"  Loaded {len(df)} Hebbian rows from {source}")
        else:
            print(f"  WARNING: Hebbian data not found at {path}")

    hebbian_df = pd.concat(hebbian_dfs, ignore_index=True) if hebbian_dfs else None
    if hebbian_df is not None:
        print(f"  Total Hebbian rows: {len(hebbian_df)}")

    # Load and merge Storkey data
    storkey_dfs = []
    for source in storkey_sources:
        path = DATA_DIR / "query_results" / source / "final_results.csv"
        if path.exists():
            df = pd.read_csv(path)
            storkey_dfs.append(df)
            print(f"  Loaded {len(df)} Storkey rows from {source}")
        else:
            print(f"  WARNING: Storkey data not found at {path}")

    storkey_df = pd.concat(storkey_dfs, ignore_index=True) if storkey_dfs else None
    if storkey_df is not None:
        print(f"  Total Storkey rows: {len(storkey_df)}")

    return hebbian_df, storkey_df


# %% =========================================================================
# HELPER FUNCTIONS
# =============================================================================

def filter_by_rho(df, rho, col='rho', atol=1e-6):
    """Filter DataFrame by rho value (handles float precision)."""
    return df[np.isclose(df[col], rho, atol=atol)]


def filter_by_informed(df, informed, col='informed_fraction', atol=1e-6):
    """Filter DataFrame by informed fraction (handles float precision)."""
    return df[np.isclose(df[col], informed, atol=atol)]


def compute_success_pivot(df, success_col):
    """
    Compute success rate pivot table.

    Args:
        df: DataFrame with network_size, num_patterns, and success column
        success_col: Name of success rate column

    Returns:
        DataFrame: Pivot table with network_size as columns, num_patterns as index
    """
    grouped = df.groupby(['network_size', 'num_patterns'])[success_col].mean().reset_index()
    pivot = grouped.pivot_table(
        values=success_col,
        index='num_patterns',
        columns='network_size',
        aggfunc='mean'
    )
    return pivot


def compute_capacity_monotonic(success_pivot, threshold):
    """
    Compute capacity using monotonic criterion.

    For each network size, find max num_patterns where ALL smaller
    num_patterns also achieve >= threshold.

    Args:
        success_pivot: DataFrame with num_patterns as index, network_size as columns
        threshold: Success rate threshold (0-1)

    Returns:
        dict: {network_size: max_capacity}
    """
    capacities = {}

    for net_size in success_pivot.columns:
        col = success_pivot[net_size].dropna().sort_index()

        if len(col) == 0:
            capacities[net_size] = 0
            continue

        max_capacity = 0
        all_above = True

        for num_pat in sorted(col.index):
            if col[num_pat] >= threshold and all_above:
                max_capacity = num_pat
            else:
                all_above = False

        capacities[net_size] = max_capacity

    return capacities


def detect_saturation(capacities, max_tested=DHN_MAX_TESTED_PATTERNS,
                      threshold=DHN_SATURATION_THRESHOLD):
    """
    Detect if capacity values are saturated (hit the max tested patterns).

    Args:
        capacities: dict of {network_size: capacity}
        max_tested: Maximum patterns tested
        threshold: Capacity value to consider as saturated

    Returns:
        dict: {network_size: bool} indicating saturation
    """
    return {k: v >= threshold for k, v in capacities.items()}


def get_spaced_indices(a, n, num_ticks=5):
    """Generate evenly spaced indices for tick marks."""
    return np.linspace(a, n, num_ticks, dtype=int)


# %% =========================================================================
# THEORETICAL CAPACITY FUNCTIONS
# =============================================================================

def theoretical_hebbian_capacity(n):
    """
    Theoretical Hebbian capacity (N / (2 * ln(N))).
    Valid for large N.
    """
    if n <= 1:
        return 0
    return n / (2 * np.log(n))


def theoretical_storkey_capacity(n):
    """
    Theoretical Storkey capacity (N / (4 * sqrt(2 * ln(N)))).
    Valid for large N.
    """
    if n <= 1:
        return 0
    return n / (4 * np.sqrt(2 * np.log(n)))


# %% =========================================================================
# LOAD ALL DATA
# =============================================================================

print("=" * 70)
print("LOADING DATA")
print("=" * 70)

print("\nLoading CHN data...")
chn_df = load_chn_data()

print("\nLoading DHN data...")
hebbian_df, storkey_df = load_dhn_data()

# %% =========================================================================
# FIGURE SET A: CHN HEATMAPS WITH CAPACITY STARS
# =============================================================================

print("\n" + "=" * 70)
print("GENERATING FIGURE SET A: CHN Heatmaps with Capacity Stars")
print("=" * 70)

if chn_df is not None:
    # Star markers for different thresholds
    THRESHOLD_MARKERS = {
        0.9: ('*', 'red', '90\\%'),
        0.8: ('s', 'yellow', '80\\%'),
        0.5: ('D', 'cyan', '50\\%'),
    }

    for rho in RHO_VALUES:
        print(f"\n  Creating heatmap for rho={rho}...")

        # Filter data by rho
        rho_df = filter_by_rho(chn_df, rho)

        if len(rho_df) == 0:
            print(f"    No data for rho={rho}")
            continue

        # Compute success rate pivot
        success_pivot = compute_success_pivot(rho_df, 'all_recovered_before_spurious')

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot heatmap
        cmap = mpl.colormaps.get_cmap('viridis').copy()
        cmap.set_bad(color='lightgrey')

        data = success_pivot.values * 100  # Convert to percentage
        masked_data = np.ma.masked_invalid(data)

        im = ax.imshow(masked_data, vmin=0, vmax=100, cmap=cmap, aspect='auto')
        ax.invert_yaxis()

        # Set tick labels
        x_positions = range(len(success_pivot.columns))
        y_positions = range(len(success_pivot.index))

        # Show every other tick for readability
        x_tick_step = max(1, len(success_pivot.columns) // 6)
        y_tick_step = max(1, len(success_pivot.index) // 8)

        ax.set_xticks(x_positions[::x_tick_step])
        ax.set_xticklabels([str(int(x)) for x in success_pivot.columns[::x_tick_step]])
        ax.set_yticks(y_positions[::y_tick_step])
        ax.set_yticklabels([str(int(y)) for y in success_pivot.index[::y_tick_step]])

        ax.set_xlabel('Network size')
        ax.set_ylabel('Number of patterns')
        ax.set_title(rf'AR Success Rate ($\rho={rho}$)')

        # Add capacity boundary stars for each threshold
        for thresh, (marker, color, label) in THRESHOLD_MARKERS.items():
            capacities = compute_capacity_monotonic(success_pivot, thresh)

            # Plot stars at capacity boundary
            star_x = []
            star_y = []

            for net_size, capacity in capacities.items():
                if capacity > 0 and net_size in success_pivot.columns:
                    x_idx = list(success_pivot.columns).index(net_size)
                    if capacity in success_pivot.index:
                        y_idx = list(success_pivot.index).index(capacity)
                        star_x.append(x_idx)
                        star_y.append(y_idx)

            if star_x:
                ax.scatter(star_x, star_y, marker=marker, c=color, s=150,
                          edgecolors='black', linewidth=1.5, label=label, zorder=10)

        ax.legend(title='Capacity at:', loc='upper right', framealpha=0.9)

        # Colorbar
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Success rate (\\%)')

        plt.tight_layout()

        if SAVE_PLOTS:
            output_path = OUTPUT_DIR / f"chn_heatmap_rho_{rho}.png"
            plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
            print(f"    Saved to: {output_path}")

        plt.show()

# %% =========================================================================
# FIGURE SET B: CAPACITY GRID PLOTS
# =============================================================================

print("\n" + "=" * 70)
print("GENERATING FIGURE SET B: Capacity Grid Plots")
print("=" * 70)

for threshold in SUCCESS_THRESHOLDS:
    print(f"\n  Creating capacity grid for {int(threshold*100)}% threshold...")

    n_rows = len(RHO_VALUES)  # 6 rows for 6 rho values
    fig, axes = plt.subplots(n_rows, 5, figsize=(22, 24), sharex=True, sharey=True)
    fig.suptitle(rf'Storage Capacity at {int(threshold*100)}\% Success Threshold',
                 fontsize=24, fontweight='bold', y=0.98)

    for row_idx, rho in enumerate(RHO_VALUES):
        for col_idx, informed in enumerate(INFORMED_FRACTIONS):
            ax = axes[row_idx, col_idx]

            # --- AR (CHN) ---
            if chn_df is not None:
                ar_subset = filter_by_rho(chn_df, rho)
                if len(ar_subset) > 0:
                    ar_pivot = compute_success_pivot(ar_subset, 'all_recovered_before_spurious')
                    ar_caps = compute_capacity_monotonic(ar_pivot, threshold)

                    x_ar = sorted(ar_caps.keys())
                    y_ar = [ar_caps[x] for x in x_ar]

                    ax.plot(x_ar, y_ar, 'o-', color=METHOD_COLORS['AR'],
                           label='AR (CHN)', linewidth=2.5, markersize=7)

            # --- Hebbian (DHN) ---
            if hebbian_df is not None:
                heb_subset = filter_by_rho(hebbian_df, rho)
                heb_subset = filter_by_informed(heb_subset, informed)

                if len(heb_subset) > 0:
                    heb_pivot = compute_success_pivot(heb_subset, 'query_success_rate')
                    heb_caps = compute_capacity_monotonic(heb_pivot, threshold)
                    heb_saturated = detect_saturation(heb_caps)

                    # Separate saturated and non-saturated points
                    x_heb_normal = [x for x in sorted(heb_caps.keys()) if not heb_saturated[x]]
                    y_heb_normal = [heb_caps[x] for x in x_heb_normal]

                    x_heb_sat = [x for x in sorted(heb_caps.keys()) if heb_saturated[x]]
                    y_heb_sat = [heb_caps[x] for x in x_heb_sat]

                    # Plot normal points
                    if x_heb_normal:
                        ax.plot(x_heb_normal, y_heb_normal, 's--', color=METHOD_COLORS['Hebbian'],
                               label='Hebbian', linewidth=2, markersize=6)

                    # Plot saturated points with triangles
                    if x_heb_sat:
                        ax.scatter(x_heb_sat, y_heb_sat, marker='^', c=METHOD_COLORS['Hebbian'],
                                  s=100, edgecolors='black', linewidth=1.5, zorder=10)
                        # Draw dashed line through all points
                        x_all = sorted(heb_caps.keys())
                        y_all = [heb_caps[x] for x in x_all]
                        ax.plot(x_all, y_all, '--', color=METHOD_COLORS['Hebbian'],
                               linewidth=1.5, alpha=0.5)

            # --- Storkey (DHN) ---
            if storkey_df is not None:
                stk_subset = filter_by_rho(storkey_df, rho)
                stk_subset = filter_by_informed(stk_subset, informed)

                if len(stk_subset) > 0:
                    stk_pivot = compute_success_pivot(stk_subset, 'query_success_rate')
                    stk_caps = compute_capacity_monotonic(stk_pivot, threshold)
                    stk_saturated = detect_saturation(stk_caps)

                    # Separate saturated and non-saturated points
                    x_stk_normal = [x for x in sorted(stk_caps.keys()) if not stk_saturated[x]]
                    y_stk_normal = [stk_caps[x] for x in x_stk_normal]

                    x_stk_sat = [x for x in sorted(stk_caps.keys()) if stk_saturated[x]]
                    y_stk_sat = [stk_caps[x] for x in x_stk_sat]

                    # Plot normal points
                    if x_stk_normal:
                        ax.plot(x_stk_normal, y_stk_normal, 'D:', color=METHOD_COLORS['Storkey'],
                               label='Storkey', linewidth=2, markersize=5)

                    # Plot saturated points with triangles
                    if x_stk_sat:
                        ax.scatter(x_stk_sat, y_stk_sat, marker='^', c=METHOD_COLORS['Storkey'],
                                  s=100, edgecolors='black', linewidth=1.5, zorder=10)
                        # Draw dotted line through all points
                        x_all = sorted(stk_caps.keys())
                        y_all = [stk_caps[x] for x in x_all]
                        ax.plot(x_all, y_all, ':', color=METHOD_COLORS['Storkey'],
                               linewidth=1.5, alpha=0.5)

            # Labels and formatting
            if row_idx == 0:
                ax.set_title(rf'{int(informed*100)}\% informed', fontweight='bold', fontsize=16)

            if col_idx == 0:
                ax.set_ylabel(rf'$\rho={rho}$' + '\nMax patterns', fontsize=14)

            if row_idx == len(RHO_VALUES) - 1:
                ax.set_xlabel('Network size', fontsize=14)

            # Legend in top-left subplot only
            if row_idx == 0 and col_idx == 0:
                ax.legend(loc='upper left', frameon=True, fontsize=12)

            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_ylim(0, 105)  # Fixed range to show full DHN capacity (max tested = 99)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if SAVE_PLOTS:
        output_path = OUTPUT_DIR / f"capacity_grid_{int(threshold*100)}pct.png"
        plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
        print(f"    Saved to: {output_path}")

    plt.show()

# %% =========================================================================
# FIGURE SET C: THEORETICAL COMPARISON AT RHO=0
# =============================================================================

print("\n" + "=" * 70)
print("GENERATING FIGURE SET C: Theoretical Comparison at rho=0")
print("=" * 70)

fig, ax = plt.subplots(figsize=(12, 8))

# Network size range for plotting
n_range = np.linspace(50, 1100, 200)

# Theoretical curves
theory_hebbian = [theoretical_hebbian_capacity(n) for n in n_range]
theory_storkey = [theoretical_storkey_capacity(n) for n in n_range]

ax.plot(n_range, theory_hebbian, '--', color=METHOD_COLORS['Hebbian'],
       linewidth=2, alpha=0.7, label='Hebbian (theory)')
ax.plot(n_range, theory_storkey, '--', color=METHOD_COLORS['Storkey'],
       linewidth=2, alpha=0.7, label='Storkey (theory)')

# Empirical data at rho=0
rho_zero = 0.0
threshold = 0.9  # Use 90% threshold for comparison

# AR empirical
if chn_df is not None:
    ar_subset = filter_by_rho(chn_df, rho_zero)
    if len(ar_subset) > 0:
        ar_pivot = compute_success_pivot(ar_subset, 'all_recovered_before_spurious')
        ar_caps = compute_capacity_monotonic(ar_pivot, threshold)

        x_ar = sorted(ar_caps.keys())
        y_ar = [ar_caps[x] for x in x_ar]

        ax.plot(x_ar, y_ar, 'o-', color=METHOD_COLORS['AR'],
               linewidth=2.5, markersize=8, label='AR (empirical)')

# Hebbian empirical - use highest informed_fraction for best comparison to theory
if hebbian_df is not None:
    heb_subset = filter_by_rho(hebbian_df, rho_zero)
    heb_subset = filter_by_informed(heb_subset, 0.9)  # 90% informed

    if len(heb_subset) > 0:
        heb_pivot = compute_success_pivot(heb_subset, 'query_success_rate')
        heb_caps = compute_capacity_monotonic(heb_pivot, threshold)
        heb_saturated = detect_saturation(heb_caps)

        x_heb = sorted(heb_caps.keys())
        y_heb = [heb_caps[x] for x in x_heb]

        # Mark saturated points
        for i, x in enumerate(x_heb):
            if heb_saturated[x]:
                ax.scatter([x], [y_heb[i]], marker='^', c=METHOD_COLORS['Hebbian'],
                          s=150, edgecolors='black', linewidth=2, zorder=10)

        ax.plot(x_heb, y_heb, 's-', color=METHOD_COLORS['Hebbian'],
               linewidth=2, markersize=6, label='Hebbian (empirical, 90\\% inf.)')

# Storkey empirical
if storkey_df is not None:
    stk_subset = filter_by_rho(storkey_df, rho_zero)
    stk_subset = filter_by_informed(stk_subset, 0.9)  # 90% informed

    if len(stk_subset) > 0:
        stk_pivot = compute_success_pivot(stk_subset, 'query_success_rate')
        stk_caps = compute_capacity_monotonic(stk_pivot, threshold)
        stk_saturated = detect_saturation(stk_caps)

        x_stk = sorted(stk_caps.keys())
        y_stk = [stk_caps[x] for x in x_stk]

        # Mark saturated points
        for i, x in enumerate(x_stk):
            if stk_saturated[x]:
                ax.scatter([x], [y_stk[i]], marker='^', c=METHOD_COLORS['Storkey'],
                          s=150, edgecolors='black', linewidth=2, zorder=10)

        ax.plot(x_stk, y_stk, 'D-', color=METHOD_COLORS['Storkey'],
               linewidth=2, markersize=6, label='Storkey (empirical, 90\\% inf.)')

ax.set_xlabel('Network size $N$')
ax.set_ylabel('Storage capacity (max patterns)')
ax.set_title(rf'Theoretical vs Empirical Capacity at $\rho=0$ (90\% success threshold)')
ax.legend(loc='upper left', frameon=True)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(0, 1100)
ax.set_ylim(bottom=0)

# Add note about saturation (use plain text triangle marker to avoid LaTeX issues)
ax.text(0.98, 0.02, r'$\triangle$ = saturated (capacity $\geq$ max tested)',
        transform=ax.transAxes, fontsize=12, ha='right', va='bottom',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()

if SAVE_PLOTS:
    output_path = OUTPUT_DIR / "theoretical_comparison_rho_0.png"
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    print(f"  Saved to: {output_path}")

plt.show()

# %% =========================================================================
# SUMMARY STATISTICS
# =============================================================================

print("\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)

# CHN overall success rate by rho
if chn_df is not None:
    print("\nCHN (AR) Success Rates by rho:")
    for rho in RHO_VALUES:
        rho_df = filter_by_rho(chn_df, rho)
        if len(rho_df) > 0:
            success_rate = rho_df['all_recovered_before_spurious'].mean() * 100
            print(f"  rho={rho}: {success_rate:.1f}% ({len(rho_df)} simulations)")

# DHN overall success rates
if hebbian_df is not None:
    print("\nDHN Hebbian Overall Success Rate:")
    success_rate = hebbian_df['query_success_rate'].mean() * 100
    print(f"  {success_rate:.1f}% ({len(hebbian_df)} simulations)")

if storkey_df is not None:
    print("\nDHN Storkey Overall Success Rate:")
    success_rate = storkey_df['query_success_rate'].mean() * 100
    print(f"  {success_rate:.1f}% ({len(storkey_df)} simulations)")

# %% =========================================================================
# DONE
# =============================================================================

print("\n" + "=" * 70)
print("VISUALIZATION COMPLETE!")
print("=" * 70)

if SAVE_PLOTS:
    print(f"\nPlots saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("  - chn_heatmap_rho_*.png (5 heatmaps)")
    print("  - capacity_grid_*pct.png (3 grid plots)")
    print("  - theoretical_comparison_rho_0.png")

print("\n" + "=" * 70)

# %%
