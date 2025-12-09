# %% [markdown]
# # Heterogeneous Sparsity Query Count Analysis
#
# This script systematically tests how pattern sparsity affects the number of
# queries needed to recover each pattern during sleep.
#
# Key features:
# - Uses C++ native heterogeneous pattern generation
# - Sweeps across network sizes (200, 250, 300) and pattern counts (5, 8, 11)
# - 30 repetitions per configuration for statistical robustness
# - Plots average query number vs pattern sparsity with variance bands

# %% Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import sys

# Add scripts directory to path (parent.parent = scripts/)
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import (
    read_pattern_metadata,
    read_parameters,
    setup_write_experiment,
    setup_sleep_experiment,
    run_cpp,
    build,
    list_simulations,
    DATA_DIR
)

# %%=========================================================================
# CONFIGURATION SECTION
# ==========================================================================

# Network and pattern parameters - systematic sweep
NETWORK_SIZES = [200, 250, 300]
NUM_PATTERNS_LIST = [5, 8, 11]
NB_REPETITION = 200
REPETITIONS = list(range(1, NB_REPETITION + 1))

# Pattern generation parameters
MEAN_SPARSITY = 0.5     # Center of sparsity distribution (P(0) convention)
SPARSITY_WIDTH = 0.4    # Full width: sparsities in [0.3, 0.7]
RHO = 0.3               # Pattern correlation

# Training parameters
LEAK = 1.0
DRIVE_TARGET = 6.0
LEARNING_RATE = 0.0001
MAX_ITER = 100000
MOMENTUM_COEF = 0.9

# Sleep parameters
BETA = 0.1               # Inhibitory plasticity rate
DELTA = 0.01             # Integration timestep
MAX_QUERIES = 200        # Number of retrieval attempts
NOISE_DYNAMICS = 1       # Enable stochastic noise
STDDEV_DYNAMICS = 0.01   # Noise standard deviation
STOP_ON_SPURIOUS = 1     # Stop when spurious pattern encountered
STOP_ON_ALL_FOUND = 1    # Stop when all patterns recovered

# Experiment names
EXPERIMENT_NAME = "heterogeneous_nb_query"
SLEEP_NAME = "heterogeneous_nb_query_sleep"

# %% [markdown]
# ## Phase 1: Build C++ Executables

# %% Build
print("="*70)
print("BUILDING C++ EXECUTABLES")
print("="*70)
build()
print("Build complete!\n")

# %% [markdown]
# ## Phase 2: Training Phase (Write) with C++ Native Pattern Generation

# %% Setup and run training
total_networks = len(NETWORK_SIZES) * len(NUM_PATTERNS_LIST) * NB_REPETITION
print("="*70)
print("TRAINING PHASE (C++ NATIVE HETEROGENEOUS GENERATION)")
print("="*70)
print(f"Network sizes: {NETWORK_SIZES}")
print(f"Pattern counts: {NUM_PATTERNS_LIST}")
print(f"Repetitions per configuration: {NB_REPETITION}")
print(f"Total networks to train: {total_networks}")
print(f"\nPattern generation params:")
print(f"  Mean sparsity (P(0)): {MEAN_SPARSITY}")
print(f"  Sparsity width: {SPARSITY_WIDTH}")
print(f"  Expected sparsity range: [{MEAN_SPARSITY - SPARSITY_WIDTH/2:.2f}, {MEAN_SPARSITY + SPARSITY_WIDTH/2:.2f}]")
print(f"  Pattern correlation (rho): {RHO}")
print("="*70 + "\n")

# Use native pattern generation (C++ generates patterns with metadata)
write_config = setup_write_experiment(
    name=EXPERIMENT_NAME,
    patterns=None,  # No patterns from Python - C++ will generate them
    pattern_metadata=None,  # No metadata from Python - C++ will generate it
    params={
        # Training parameters
        "leak": LEAK,
        "drive_target": DRIVE_TARGET,
        "learning_rate": LEARNING_RATE,
        "max_iter": MAX_ITER,
        "momentum_coef": MOMENTUM_COEF,
        # Native pattern generation parameters
        "use_heterogeneous_sparsity": 1,  # Enable heterogeneous mode
        "mean_sparsity": MEAN_SPARSITY,
        "sparsity_width": SPARSITY_WIDTH,
    },
    varying_params={
        "network_size": NETWORK_SIZES,
        "num_patterns": NUM_PATTERNS_LIST,
        "rho": [RHO],
        "nb_repetition": REPETITIONS,
    },
    native_pattern_generation=True,  # Enable C++ native generation
)

print(f"Configuration saved to: {write_config}\n")
print("Starting training with C++ native pattern generation...")
print("All simulations will run in parallel...")
run_cpp("write", write_config)
print("\nTraining complete!")

# %% [markdown]
# ## Phase 3: Sleep Phase

# %% Setup and run sleep
print("\n" + "="*70)
print("SLEEP PHASE")
print("="*70)
print(f"Running sleep simulations on {total_networks} trained networks")
print(f"Beta (inhibitory plasticity): {BETA}")
print(f"Delta (timestep): {DELTA}")
print(f"Max queries: {MAX_QUERIES}")
print(f"Stop on spurious: {STOP_ON_SPURIOUS}")
print(f"Stop on all found: {STOP_ON_ALL_FOUND}")
print("="*70 + "\n")

sleep_config = setup_sleep_experiment(
    name=SLEEP_NAME,
    trained_networks_dir=DATA_DIR / "trained_networks" / EXPERIMENT_NAME,
    params={
        "beta": BETA,
        "delta": DELTA,
        "noise_dynamics": NOISE_DYNAMICS,
        "stddev_dynamics": STDDEV_DYNAMICS,
        "max_queries": MAX_QUERIES,
        "stop_on_spurious": STOP_ON_SPURIOUS,
        "stop_on_all_found": STOP_ON_ALL_FOUND,
        "save_trajectories": 0,
    }
)

print(f"Configuration saved to: {sleep_config}\n")
print("Starting sleep simulations...")
run_cpp("sleep", sleep_config)
print("\nSleep simulations complete!")

# %% [markdown]
# ## Phase 4: Analysis - Track First Recovery Query per Pattern

# %% Load and process results
print("\n" + "="*70)
print("ANALYSIS")
print("="*70)

results_dir = DATA_DIR / "sleep_results" / SLEEP_NAME
sim_dirs = list_simulations(results_dir)
print(f"Found {len(sim_dirs)} simulation directories")

# Track first recovery query for each pattern
first_recovery_queries = []

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

    # For each pattern, find first query where it was recovered
    for pattern in metadata["patterns"]:
        idx = pattern["index"]
        sparsity = pattern["sparsity"]

        # Find first row where recovered_pattern_idx == idx
        recovery_rows = sim_results[sim_results["recovered_pattern_idx"] == idx]
        if len(recovery_rows) > 0:
            first_query = recovery_rows["query_iter"].min()
        else:
            first_query = np.nan  # Pattern never recovered

        first_recovery_queries.append({
            "network_size": int(params["network_size"]),
            "num_patterns": int(params["num_patterns"]),
            "repetition": int(params.get("nb_repetition", 0)),
            "pattern_idx": idx,
            "sparsity": sparsity,
            "first_query": first_query
        })

df = pd.DataFrame(first_recovery_queries)
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
# ## Phase 5: Visualization - 3x3 Grid

# %% Create visualization
print("\n" + "="*70)
print("VISUALIZATION")
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
# ## Phase 6: Recovery Count vs Sparsity

# %% Compute recovery counts per pattern
print("\n" + "="*70)
print("RECOVERY COUNT ANALYSIS")
print("="*70)

# Re-process to count total recoveries per pattern (not just first)
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

    # For each pattern, count how many times it was recovered
    for pattern in metadata["patterns"]:
        idx = pattern["index"]
        sparsity = pattern["sparsity"]

        # Count all rows where recovered_pattern_idx == idx
        count = (sim_results["recovered_pattern_idx"] == idx).sum()

        recovery_counts.append({
            "network_size": int(params["network_size"]),
            "num_patterns": int(params["num_patterns"]),
            "repetition": int(params.get("nb_repetition", 0)),
            "pattern_idx": idx,
            "sparsity": sparsity,
            "recovery_count": count
        })

df_counts = pd.DataFrame(recovery_counts)
print(f"Collected {len(df_counts)} pattern recovery count records")

# %% Visualize recovery count vs sparsity
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
print("EXPERIMENT COMPLETE!")
print("="*70)
print(f"\nTrained networks: {DATA_DIR / 'trained_networks' / EXPERIMENT_NAME}")
print(f"Sleep results: {results_dir}")
print(f"\nVisualizations:")
print(f"  - First query vs sparsity: {output_path}")
print(f"  - Recovery count vs sparsity: {output_path2}")
print("\nNote: Patterns were generated natively in C++ with heterogeneous sparsities!")
print("="*70 + "\n")

# %%
