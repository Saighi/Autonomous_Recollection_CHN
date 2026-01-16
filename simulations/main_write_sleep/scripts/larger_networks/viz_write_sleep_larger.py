#%%
"""
Visualization: CHN Write+Sleep Results for Larger Networks

Visualizes autonomous retrieval (AR) success from the write_sleep_chn_larger experiment.
- Small networks (N=300-500): Shown together in multi-panel figures
- Large networks (N=1000): Shown separately

For each network size and rho, shows:
- AR success rate vs number of stored patterns
- Recovery fraction over queries
"""

#%% Imports
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_final_results, DATA_DIR

#%% Configuration
SMALL_SLEEP_NAME = "capacity_scaling_larger_small_sleep"
LARGE_SLEEP_NAME = "capacity_scaling_larger_large_sleep"

# Plot output directory
PLOTS_DIR = DATA_DIR / "plots" / "larger_networks"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
SAVE_PLOTS = True

# Color palette for different rho values
RHO_COLORS = {
    0.0: '#1f77b4',   # Blue - uncorrelated
    0.25: '#2ca02c',  # Green - low correlation
    0.5: '#9467bd',   # Purple - medium correlation
    0.75: '#ff7f0e',  # Orange - high correlation
    0.9: '#d62728',   # Red - very high correlation
}

RHO_LABELS = {
    0.0: r'$\rho = 0$ (uncorrelated)',
    0.25: r'$\rho = 0.25$',
    0.5: r'$\rho = 0.5$',
    0.75: r'$\rho = 0.75$',
    0.9: r'$\rho = 0.9$ (correlated)',
}

#%% Matplotlib Settings
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'lines.linewidth': 2,
    'lines.markersize': 6,
    'axes.linewidth': 1.2,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

#%% Load Data
print("=" * 70)
print("LOADING RESULTS")
print("=" * 70)

small_dir = DATA_DIR / "sleep_results" / SMALL_SLEEP_NAME
large_dir = DATA_DIR / "sleep_results" / LARGE_SLEEP_NAME

print(f"\nSmall networks: {small_dir}")
if small_dir.exists():
    small_df = load_final_results(small_dir)
    print(f"  Loaded {len(small_df)} simulations")
    print(f"  Network sizes: {sorted(small_df['network_size'].unique())}")
    print(f"  Rho values: {sorted(small_df['rho'].unique())}")
else:
    print("  WARNING: Not found!")
    small_df = None

print(f"\nLarge networks: {large_dir}")
if large_dir.exists():
    large_df = load_final_results(large_dir)
    print(f"  Loaded {len(large_df)} simulations")
    print(f"  Network sizes: {sorted(large_df['network_size'].unique())}")
    print(f"  Rho values: {sorted(large_df['rho'].unique())}")
else:
    print("  WARNING: Not found!")
    large_df = None

#%% Helper Functions
def compute_success_rate(df, success_col='all_recovered_before_spurious'):
    """
    Compute AR success rate for each (network_size, num_patterns, rho) combination.

    Returns DataFrame with columns:
    - network_size, num_patterns, rho
    - success_rate: Mean success rate across seeds
    - n_sims: Number of simulations
    """
    grouped = df.groupby(['network_size', 'num_patterns', 'rho']).agg(
        success_rate=(success_col, 'mean'),
        n_sims=(success_col, 'count')
    ).reset_index()
    return grouped


def compute_recovery_fraction(df):
    """
    Compute mean recovery fraction (nb_fnd_pat / num_patterns) for each combination.
    """
    df = df.copy()
    df['recovery_fraction'] = df['nb_fnd_pat'] / df['num_patterns']
    grouped = df.groupby(['network_size', 'num_patterns', 'rho']).agg(
        recovery_fraction=('recovery_fraction', 'mean'),
        recovery_std=('recovery_fraction', 'std'),
        n_sims=('recovery_fraction', 'count')
    ).reset_index()
    return grouped


#%% ============================================================================
# FIGURE 1: Small Networks (300-500) - AR Success Rate
# ============================================================================
print("\n" + "=" * 70)
print("FIGURE 1: Small Networks - AR Success Rate")
print("=" * 70)

if small_df is not None:
    success_df = compute_success_rate(small_df)
    network_sizes = sorted(small_df['network_size'].unique())
    rho_values = sorted(small_df['rho'].unique())

    # Create figure with subplot for each network size
    fig1, axes = plt.subplots(1, len(network_sizes), figsize=(4 * len(network_sizes), 4), sharey=True)

    for idx, net_size in enumerate(network_sizes):
        ax = axes[idx]
        subset = success_df[success_df['network_size'] == net_size]

        for rho in rho_values:
            rho_subset = subset[np.isclose(subset['rho'], rho, atol=1e-6)]
            if len(rho_subset) > 0:
                rho_subset = rho_subset.sort_values('num_patterns')
                ax.plot(rho_subset['num_patterns'], rho_subset['success_rate'] * 100,
                       'o-', color=RHO_COLORS.get(rho, 'gray'),
                       label=RHO_LABELS.get(rho, f'rho={rho}'))

        ax.set_xlabel('Number of Patterns')
        ax.set_title(f'N = {net_size}')
        ax.set_ylim(-5, 105)
        ax.axhline(90, color='gray', linestyle=':', alpha=0.5, label='90% threshold')

        if idx == 0:
            ax.set_ylabel('AR Success Rate (%)')

    # Add legend to the last subplot
    axes[-1].legend(loc='lower left', fontsize=9)

    fig1.suptitle('Autonomous Retrieval Success: Small Networks (N=300-500)', fontsize=14, y=1.02)
    plt.tight_layout()

    if SAVE_PLOTS:
        filepath = PLOTS_DIR / 'small_networks_success_rate.png'
        fig1.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")

#%% ============================================================================
# FIGURE 2: Small Networks (300-500) - Recovery Fraction
# ============================================================================
print("\n" + "=" * 70)
print("FIGURE 2: Small Networks - Recovery Fraction")
print("=" * 70)

if small_df is not None:
    recovery_df = compute_recovery_fraction(small_df)
    network_sizes = sorted(small_df['network_size'].unique())
    rho_values = sorted(small_df['rho'].unique())

    fig2, axes = plt.subplots(1, len(network_sizes), figsize=(4 * len(network_sizes), 4), sharey=True)

    for idx, net_size in enumerate(network_sizes):
        ax = axes[idx]
        subset = recovery_df[recovery_df['network_size'] == net_size]

        for rho in rho_values:
            rho_subset = subset[np.isclose(subset['rho'], rho, atol=1e-6)]
            if len(rho_subset) > 0:
                rho_subset = rho_subset.sort_values('num_patterns')
                ax.errorbar(rho_subset['num_patterns'], rho_subset['recovery_fraction'] * 100,
                           yerr=rho_subset['recovery_std'] * 100,
                           fmt='o-', color=RHO_COLORS.get(rho, 'gray'),
                           capsize=3, label=RHO_LABELS.get(rho, f'rho={rho}'))

        ax.set_xlabel('Number of Patterns')
        ax.set_title(f'N = {net_size}')
        ax.set_ylim(-5, 105)

        if idx == 0:
            ax.set_ylabel('Recovery Fraction (%)')

    axes[-1].legend(loc='lower left', fontsize=9)

    fig2.suptitle('Pattern Recovery Fraction: Small Networks (N=300-500)', fontsize=14, y=1.02)
    plt.tight_layout()

    if SAVE_PLOTS:
        filepath = PLOTS_DIR / 'small_networks_recovery_fraction.png'
        fig2.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")

#%% ============================================================================
# FIGURE 3: Small Networks - Capacity Heatmap
# ============================================================================
print("\n" + "=" * 70)
print("FIGURE 3: Small Networks - Capacity Heatmap")
print("=" * 70)

if small_df is not None:
    success_df = compute_success_rate(small_df)
    network_sizes = sorted(small_df['network_size'].unique())
    rho_values = sorted(small_df['rho'].unique())

    # Dynamic grid layout based on number of rho values
    n_rho = len(rho_values)
    n_cols = min(3, n_rho)
    n_rows = (n_rho + n_cols - 1) // n_cols
    fig3, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for idx, rho in enumerate(rho_values):
        ax = axes[idx]
        rho_df = success_df[np.isclose(success_df['rho'], rho, atol=1e-6)]

        # Create pivot table
        pivot = rho_df.pivot(index='num_patterns', columns='network_size', values='success_rate')
        pivot = pivot.sort_index(ascending=False)  # High patterns at top

        # Plot heatmap
        im = ax.imshow(pivot.values * 100, aspect='auto', cmap='RdYlGn', vmin=0, vmax=100)

        # Labels
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)

        ax.set_xlabel('Network Size (N)')
        ax.set_ylabel('Number of Patterns (K)')
        ax.set_title(RHO_LABELS.get(rho, f'rho={rho}'))

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('AR Success (%)')

    # Hide unused subplots
    for idx in range(n_rho, len(axes)):
        axes[idx].set_visible(False)

    fig3.suptitle('AR Success Heatmaps: Small Networks (N=300-500)', fontsize=14, y=1.02)
    plt.tight_layout()

    if SAVE_PLOTS:
        filepath = PLOTS_DIR / 'small_networks_heatmap.png'
        fig3.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")

#%% ============================================================================
# FIGURE 4: Large Networks (N=1000) - AR Success Rate
# ============================================================================
print("\n" + "=" * 70)
print("FIGURE 4: Large Networks (N=1000) - AR Success Rate")
print("=" * 70)

if large_df is not None:
    success_df = compute_success_rate(large_df)
    rho_values = sorted(large_df['rho'].unique())

    fig4, ax = plt.subplots(figsize=(10, 6))

    for rho in rho_values:
        rho_subset = success_df[np.isclose(success_df['rho'], rho, atol=1e-6)]
        if len(rho_subset) > 0:
            rho_subset = rho_subset.sort_values('num_patterns')
            ax.plot(rho_subset['num_patterns'], rho_subset['success_rate'] * 100,
                   'o-', color=RHO_COLORS.get(rho, 'gray'), markersize=8,
                   label=RHO_LABELS.get(rho, f'rho={rho}'))

    ax.set_xlabel('Number of Patterns (K)')
    ax.set_ylabel('AR Success Rate (%)')
    ax.set_title('Autonomous Retrieval Success: Large Network (N=1000)')
    ax.set_ylim(-5, 105)
    ax.axhline(90, color='gray', linestyle=':', alpha=0.5, label='90% threshold')
    ax.legend(loc='lower left')

    plt.tight_layout()

    if SAVE_PLOTS:
        filepath = PLOTS_DIR / 'large_network_1000_success_rate.png'
        fig4.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")

#%% ============================================================================
# FIGURE 5: Large Networks (N=1000) - Recovery Fraction
# ============================================================================
print("\n" + "=" * 70)
print("FIGURE 5: Large Networks (N=1000) - Recovery Fraction")
print("=" * 70)

if large_df is not None:
    recovery_df = compute_recovery_fraction(large_df)
    rho_values = sorted(large_df['rho'].unique())

    fig5, ax = plt.subplots(figsize=(10, 6))

    for rho in rho_values:
        rho_subset = recovery_df[np.isclose(recovery_df['rho'], rho, atol=1e-6)]
        if len(rho_subset) > 0:
            rho_subset = rho_subset.sort_values('num_patterns')
            ax.errorbar(rho_subset['num_patterns'], rho_subset['recovery_fraction'] * 100,
                       yerr=rho_subset['recovery_std'] * 100,
                       fmt='o-', color=RHO_COLORS.get(rho, 'gray'), markersize=8,
                       capsize=3, label=RHO_LABELS.get(rho, f'rho={rho}'))

    ax.set_xlabel('Number of Patterns (K)')
    ax.set_ylabel('Recovery Fraction (%)')
    ax.set_title('Pattern Recovery Fraction: Large Network (N=1000)')
    ax.set_ylim(-5, 105)
    ax.legend(loc='lower left')

    plt.tight_layout()

    if SAVE_PLOTS:
        filepath = PLOTS_DIR / 'large_network_1000_recovery_fraction.png'
        fig5.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")

#%% ============================================================================
# FIGURE 6: Large Networks (N=1000) - Capacity Heatmap
# ============================================================================
print("\n" + "=" * 70)
print("FIGURE 6: Large Networks (N=1000) - Capacity Heatmap")
print("=" * 70)

if large_df is not None:
    success_df = compute_success_rate(large_df)
    rho_values = sorted(large_df['rho'].unique())
    num_patterns_values = sorted(large_df['num_patterns'].unique())

    # Create pivot table: rows = num_patterns, columns = rho
    pivot = success_df.pivot(index='num_patterns', columns='rho', values='success_rate')
    pivot = pivot.sort_index(ascending=False)  # High patterns at top

    fig6, ax = plt.subplots(figsize=(10, 8))

    # Plot heatmap
    im = ax.imshow(pivot.values * 100, aspect='auto', cmap='RdYlGn', vmin=0, vmax=100)

    # Labels
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f'{rho}' for rho in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    ax.set_xlabel(r'Pattern Correlation ($\rho$)')
    ax.set_ylabel('Number of Patterns (K)')
    ax.set_title('AR Success Rate Heatmap: Large Network (N=1000)')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('AR Success (%)')

    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j] * 100
            text_color = 'white' if val < 50 else 'black'
            ax.text(j, i, f'{val:.0f}', ha='center', va='center',
                   color=text_color, fontsize=9)

    plt.tight_layout()

    if SAVE_PLOTS:
        filepath = PLOTS_DIR / 'large_network_1000_heatmap.png'
        fig6.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")

#%% ============================================================================
# FIGURE 7: Large Networks (N=1000) - Detailed by Rho
# ============================================================================
print("\n" + "=" * 70)
print("FIGURE 7: Large Networks (N=1000) - Detailed by Rho")
print("=" * 70)

if large_df is not None:
    success_df = compute_success_rate(large_df)
    recovery_df = compute_recovery_fraction(large_df)
    rho_values = sorted(large_df['rho'].unique())

    # Dynamic grid layout based on number of rho values
    n_rho = len(rho_values)
    n_cols = min(3, n_rho)
    n_rows = (n_rho + n_cols - 1) // n_cols
    fig7, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for idx, rho in enumerate(rho_values):
        ax = axes[idx]
        color = RHO_COLORS.get(rho, 'gray')

        # Success rate
        success_subset = success_df[np.isclose(success_df['rho'], rho, atol=1e-6)]
        success_subset = success_subset.sort_values('num_patterns')

        # Recovery fraction
        recovery_subset = recovery_df[np.isclose(recovery_df['rho'], rho, atol=1e-6)]
        recovery_subset = recovery_subset.sort_values('num_patterns')

        # Plot both metrics
        ax.plot(success_subset['num_patterns'], success_subset['success_rate'] * 100,
               'o-', color=color, label='AR Success Rate')
        ax.errorbar(recovery_subset['num_patterns'], recovery_subset['recovery_fraction'] * 100,
                   yerr=recovery_subset['recovery_std'] * 100,
                   fmt='s--', color=color, alpha=0.6, capsize=3, label='Recovery Fraction')

        ax.set_xlabel('Number of Patterns (K)')
        ax.set_ylabel('Percentage (%)')
        ax.set_title(f'N=1000, {RHO_LABELS.get(rho, f"rho={rho}")}')
        ax.set_ylim(-5, 105)
        ax.axhline(90, color='gray', linestyle=':', alpha=0.5)
        ax.legend(loc='lower left', fontsize=9)

    # Hide unused subplots
    for idx in range(n_rho, len(axes)):
        axes[idx].set_visible(False)

    fig7.suptitle('Large Network (N=1000): AR Success vs Recovery by Correlation', fontsize=14, y=1.02)
    plt.tight_layout()

    if SAVE_PLOTS:
        filepath = PLOTS_DIR / 'large_network_1000_detailed.png'
        fig7.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")

#%% ============================================================================
# FIGURE 8: Capacity Comparison (All Sizes)
# ============================================================================
print("\n" + "=" * 70)
print("FIGURE 8: Capacity Comparison (All Sizes)")
print("=" * 70)

def find_capacity_threshold(df, threshold=0.9):
    """Find max patterns achieving >= threshold success rate for each (network_size, rho)."""
    success_df = compute_success_rate(df)
    capacities = []

    for net_size in success_df['network_size'].unique():
        for rho in success_df['rho'].unique():
            subset = success_df[(success_df['network_size'] == net_size) &
                               np.isclose(success_df['rho'], rho, atol=1e-6)]
            successful = subset[subset['success_rate'] >= threshold]
            if len(successful) > 0:
                max_patterns = successful['num_patterns'].max()
            else:
                max_patterns = 0
            capacities.append({
                'network_size': net_size,
                'rho': rho,
                'capacity': max_patterns
            })

    return pd.DataFrame(capacities)

if small_df is not None and large_df is not None:
    # Combine data
    combined_df = pd.concat([small_df, large_df], ignore_index=True)
    capacity_df = find_capacity_threshold(combined_df, threshold=0.9)
    rho_values = sorted(capacity_df['rho'].unique())

    fig8, ax = plt.subplots(figsize=(10, 6))

    for rho in rho_values:
        rho_subset = capacity_df[np.isclose(capacity_df['rho'], rho, atol=1e-6)]
        rho_subset = rho_subset.sort_values('network_size')
        ax.plot(rho_subset['network_size'], rho_subset['capacity'],
               'o-', color=RHO_COLORS.get(rho, 'gray'), markersize=10,
               label=RHO_LABELS.get(rho, f'rho={rho}'))

    ax.set_xlabel('Network Size (N)')
    ax.set_ylabel('Capacity (Max Patterns at 90% AR Success)')
    ax.set_title('CHN Storage Capacity Scaling by Pattern Correlation')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    # Add linear reference line (capacity ~ N)
    x_range = np.array([300, 1000])
    ax.plot(x_range, x_range * 0.03, '--', color='gray', alpha=0.5, label='~3% of N')

    plt.tight_layout()

    if SAVE_PLOTS:
        filepath = PLOTS_DIR / 'capacity_scaling_all_rho.png'
        fig8.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")

#%% Summary
print("\n" + "=" * 70)
print("VISUALIZATION COMPLETE")
print("=" * 70)

if small_df is not None:
    print(f"\nSmall networks (N=300-500):")
    print(f"  Simulations: {len(small_df)}")
    print(f"  Network sizes: {sorted(small_df['network_size'].unique())}")

if large_df is not None:
    print(f"\nLarge networks (N=1000):")
    print(f"  Simulations: {len(large_df)}")
    print(f"  Pattern range: {large_df['num_patterns'].min()} - {large_df['num_patterns'].max()}")

print(f"\nPlots saved to: {PLOTS_DIR}")

plt.show()
