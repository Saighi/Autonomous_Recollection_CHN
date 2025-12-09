# %% [markdown]
# # Heterogeneous Sparsity Query Count - Visualization Script
#
# This script analyzes and visualizes data from heterogeneous_nb_query_sim.py
#
# Produces two figures:
# 1. First query number vs pattern sparsity (when each pattern was first recovered)
# 2. Recovery count vs pattern sparsity (how many times each pattern was recovered)
#
# Run simulation first with: scripts/recovery_cinematic/heterogeneous_nb_query_sim.py

# %% Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import sys

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import (
    read_pattern_metadata,
    read_parameters,
    list_simulations,
    DATA_DIR
)

# %%=========================================================================
# CONFIGURATION SECTION
# ==========================================================================

# Must match simulation parameters
NETWORK_SIZES = [200, 250, 300]
NUM_PATTERNS_LIST = [5, 8, 11]

# Experiment names (must match simulation script)
EXPERIMENT_NAME = "heterogeneous_nb_query"
SLEEP_NAME = "heterogeneous_nb_query_sleep"

# %% [markdown]
# ## Phase 1: Load and Process Results

# %% Load results
print("="*70)
print("LOADING RESULTS")
print("="*70)

results_dir = DATA_DIR / "sleep_results" / SLEEP_NAME
sim_dirs = list_simulations(results_dir)
print(f"Found {len(sim_dirs)} simulation directories")

# Track first recovery query for each pattern
first_recovery_queries = []
# Track recovery counts for each pattern
recovery_counts = []

for sim_dir in sim_dirs:
    # Load simulation results
    results_file = sim_dir / "results.data"
    if not results_file.exists():
        continue

    sim_results = pd.read_csv(results_file)

    # Load pattern metadata
    metadata_file = sim_dir / "pattern_metadata.json"
    if not metadata_file.exists():
        continue
    metadata = read_pattern_metadata(metadata_file)

    # Load parameters
    params_file = sim_dir / "parameters.data"
    params = read_parameters(params_file)

    # For each pattern, collect statistics
    for pattern in metadata["patterns"]:
        idx = pattern["index"]
        sparsity = pattern["sparsity"]

        # Find first row where recovered_pattern_idx == idx
        recovery_rows = sim_results[sim_results["recovered_pattern_idx"] == idx]
        if len(recovery_rows) > 0:
            first_query = recovery_rows["query_iter"].min()
        else:
            first_query = np.nan  # Pattern never recovered

        # Count all recoveries
        count = (sim_results["recovered_pattern_idx"] == idx).sum()

        first_recovery_queries.append({
            "network_size": int(params["network_size"]),
            "num_patterns": int(params["num_patterns"]),
            "repetition": int(params.get("nb_repetition", 0)),
            "pattern_idx": idx,
            "sparsity": sparsity,
            "first_query": first_query
        })

        recovery_counts.append({
            "network_size": int(params["network_size"]),
            "num_patterns": int(params["num_patterns"]),
            "repetition": int(params.get("nb_repetition", 0)),
            "pattern_idx": idx,
            "sparsity": sparsity,
            "recovery_count": count
        })

df = pd.DataFrame(first_recovery_queries)
df_counts = pd.DataFrame(recovery_counts)
print(f"\nCollected {len(df)} pattern recovery records")
print(f"Patterns never recovered: {df['first_query'].isna().sum()}")

# %% Summary statistics
print("\nRecovery statistics by configuration:")
for n_size in NETWORK_SIZES:
    for n_pat in NUM_PATTERNS_LIST:
        subset = df[(df.network_size == n_size) & (df.num_patterns == n_pat)]
        recovered = subset['first_query'].notna().sum()
        total = len(subset)
        pct = 100 * recovered / total if total > 0 else 0
        mean_q = subset['first_query'].mean()
        print(f"  N={n_size}, K={n_pat}: {recovered}/{total} recovered ({pct:.1f}%), mean query={mean_q:.1f}")

# %% [markdown]
# ## Phase 2: Visualization - First Query vs Sparsity

# %% Create first query visualization
print("\n" + "="*70)
print("VISUALIZATION: First Query vs Sparsity")
print("="*70)

# Set up publication-quality style
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 12,
})

fig, axes = plt.subplots(3, 3, figsize=(14, 12))
fig.suptitle('Query Number vs Pattern Sparsity (Heterogeneous Patterns)', fontsize=14, fontweight='bold')

# Color for the main line and fill
color = '#1f77b4'  # matplotlib default blue

for i, n_size in enumerate(NETWORK_SIZES):
    for j, n_pat in enumerate(NUM_PATTERNS_LIST):
        ax = axes[i, j]

        # Filter data for this configuration
        subset = df[(df.network_size == n_size) & (df.num_patterns == n_pat)].copy()

        # Drop NaN values (unrecovered patterns)
        subset = subset.dropna(subset=['first_query'])

        if len(subset) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'N={n_size}, K={n_pat}')
            continue

        # Create sparsity bins for aggregation
        n_bins = 15
        subset['sparsity_bin'] = pd.cut(subset['sparsity'], bins=n_bins)

        # Compute mean and std per bin
        stats = subset.groupby('sparsity_bin', observed=True).agg({
            'first_query': ['mean', 'std', 'count'],
            'sparsity': 'mean'  # Use actual mean sparsity in bin
        }).reset_index()

        # Flatten column names
        stats.columns = ['bin', 'mean_query', 'std_query', 'count', 'mean_sparsity']
        stats = stats.dropna()

        if len(stats) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'N={n_size}, K={n_pat}')
            continue

        # Sort by sparsity
        stats = stats.sort_values('mean_sparsity')

        x = stats['mean_sparsity']
        y = stats['mean_query']
        yerr = stats['std_query'].fillna(0)

        # Plot shaded variance region
        ax.fill_between(x, y - yerr, y + yerr, alpha=0.25, color=color)

        # Plot mean line with markers
        ax.plot(x, y, 'o-', color=color, linewidth=2, markersize=5, markerfacecolor='white', markeredgewidth=1.5)

        # Styling
        ax.set_title(f'N={n_size}, K={n_pat}', fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(0.25, 0.75)

        # Only show y-label on leftmost column
        if j == 0:
            ax.set_ylabel('Query Number\n(first recovery)')

        # Only show x-label on bottom row
        if i == 2:
            ax.set_xlabel('Pattern Sparsity (P(0))')

plt.tight_layout()

# Save figure
output_path = Path(__file__).parent.parent / "plots" / "heterogeneous_nb_query.png"
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"\nFigure saved to: {output_path}")

plt.show()

# %% [markdown]
# ## Phase 3: Visualization - Recovery Count vs Sparsity

# %% Create recovery count visualization
print("\n" + "="*70)
print("VISUALIZATION: Recovery Count vs Sparsity")
print("="*70)

fig2, axes2 = plt.subplots(3, 3, figsize=(14, 12))
fig2.suptitle('Recovery Count vs Pattern Sparsity (Heterogeneous Patterns)', fontsize=14, fontweight='bold')

# Color for the main line and fill
color2 = '#2ca02c'  # matplotlib green

for i, n_size in enumerate(NETWORK_SIZES):
    for j, n_pat in enumerate(NUM_PATTERNS_LIST):
        ax = axes2[i, j]

        # Filter data for this configuration
        subset = df_counts[(df_counts.network_size == n_size) & (df_counts.num_patterns == n_pat)].copy()

        if len(subset) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'N={n_size}, K={n_pat}')
            continue

        # Create sparsity bins for aggregation
        n_bins = 15
        subset['sparsity_bin'] = pd.cut(subset['sparsity'], bins=n_bins)

        # Compute mean and std per bin
        stats = subset.groupby('sparsity_bin', observed=True).agg({
            'recovery_count': ['mean', 'std', 'count'],
            'sparsity': 'mean'  # Use actual mean sparsity in bin
        }).reset_index()

        # Flatten column names
        stats.columns = ['bin', 'mean_count', 'std_count', 'n_samples', 'mean_sparsity']
        stats = stats.dropna()

        if len(stats) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'N={n_size}, K={n_pat}')
            continue

        # Sort by sparsity
        stats = stats.sort_values('mean_sparsity')

        x = stats['mean_sparsity']
        y = stats['mean_count']
        yerr = stats['std_count'].fillna(0)

        # Plot shaded variance region
        ax.fill_between(x, y - yerr, y + yerr, alpha=0.25, color=color2)

        # Plot mean line with markers
        ax.plot(x, y, 'o-', color=color2, linewidth=2, markersize=5, markerfacecolor='white', markeredgewidth=1.5)

        # Styling
        ax.set_title(f'N={n_size}, K={n_pat}', fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(0.25, 0.75)

        # Only show y-label on leftmost column
        if j == 0:
            ax.set_ylabel('Recovery Count')

        # Only show x-label on bottom row
        if i == 2:
            ax.set_xlabel('Pattern Sparsity (P(0))')

plt.tight_layout()

# Save figure
output_path2 = Path(__file__).parent.parent / "plots" / "heterogeneous_recovery_count.png"
plt.savefig(output_path2, dpi=150, bbox_inches='tight', facecolor='white')
print(f"\nFigure saved to: {output_path2}")

plt.show()

# %% Summary
print("\n" + "="*70)
print("VISUALIZATION COMPLETE!")
print("="*70)
print(f"\nVisualizations:")
print(f"  - First query vs sparsity: {output_path}")
print(f"  - Recovery count vs sparsity: {output_path2}")
print("="*70 + "\n")

# %%
