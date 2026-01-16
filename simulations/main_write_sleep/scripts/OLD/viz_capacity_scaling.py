#%%
"""
Visualization: CHN Capacity Scaling (Small + Larger Networks)

Combines results from small networks (comparison_chn_cpp_sleep) and
larger networks (comparison_chn_larger_sleep) to show how storage
capacity scales with network size for rho = 0.5.

Shows: Network size vs Max patterns achieving 90% AR success.
"""

#%% Imports
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils import load_final_results, DATA_DIR

#%% Configuration
SMALL_SLEEP_NAME = "comparison_chn_cpp_sleep"
LARGER_SLEEP_NAME = "comparison_chn_larger_sleep"

RHO_FILTER = 0.5  # Only show rho = 0.5
SUCCESS_THRESHOLD = 0.9  # 90% threshold for capacity

PLOTS_DIR = DATA_DIR / "plots" / "capacity_scaling"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
SAVE_PLOTS = True

#%% Matplotlib Settings
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'lines.linewidth': 2.5,
    'axes.linewidth': 1.2,
    'axes.grid': True,
})

#%% Load Small Networks Results
print("=" * 70)
print("LOADING RESULTS")
print("=" * 70)

small_dir = DATA_DIR / "sleep_results" / SMALL_SLEEP_NAME
print(f"\nSmall networks: {small_dir}")

if small_dir.exists():
    small_df = load_final_results(small_dir)
    print(f"  Loaded {len(small_df)} simulations")
else:
    print(f"  WARNING: Not found!")
    small_df = None

#%% Load Larger Networks Results
larger_dir = DATA_DIR / "sleep_results" / LARGER_SLEEP_NAME
print(f"\nLarger networks: {larger_dir}")

if larger_dir.exists():
    larger_df = load_final_results(larger_dir)
    print(f"  Loaded {len(larger_df)} simulations")
else:
    print(f"  WARNING: Not found!")
    larger_df = None

#%% Filter by rho = 0.5 and Combine
print(f"\nFiltering for rho = {RHO_FILTER}...")

dfs_to_combine = []

if small_df is not None:
    small_filtered = small_df[np.isclose(small_df['rho'], RHO_FILTER, atol=1e-6)]
    print(f"  Small networks (rho={RHO_FILTER}): {len(small_filtered)} simulations")
    dfs_to_combine.append(small_filtered)

if larger_df is not None:
    larger_filtered = larger_df[np.isclose(larger_df['rho'], RHO_FILTER, atol=1e-6)]
    print(f"  Larger networks (rho={RHO_FILTER}): {len(larger_filtered)} simulations")
    dfs_to_combine.append(larger_filtered)

if not dfs_to_combine:
    raise ValueError("No data found! Run write and sleep scripts first.")

combined_df = pd.concat(dfs_to_combine, ignore_index=True)
print(f"\nCombined: {len(combined_df)} total simulations")

#%% Compute Capacity for Each Network Size
def find_max_capacity(df, threshold=SUCCESS_THRESHOLD, success_col='all_recovered_before_spurious'):
    """
    For each network size, find max patterns achieving >= threshold success.
    """
    capacities = {}
    network_sizes = sorted(df['network_size'].unique())

    for net_size in network_sizes:
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

print(f"\nComputing capacity at {int(SUCCESS_THRESHOLD*100)}% threshold...")
capacities = find_max_capacity(combined_df)

print("\nCapacity by network size:")
for net_size in sorted(capacities.keys()):
    print(f"  N={net_size}: {capacities[net_size]} patterns")

#%% Plot Capacity Curve
print("\n" + "=" * 70)
print("GENERATING PLOT")
print("=" * 70)

fig, ax = plt.subplots(figsize=(10, 7))

x = sorted(capacities.keys())
y = [capacities[size] for size in x]

ax.plot(x, y, 'o-', color='#1f77b4', markersize=8, linewidth=2.5, label='AR (CHN)')

ax.set_xlabel('Network Size (N)')
ax.set_ylabel(f'Max Patterns at {int(SUCCESS_THRESHOLD*100)}% AR Success')
ax.set_title(f'CHN Storage Capacity Scaling ($\\rho$ = {RHO_FILTER})')

ax.set_xlim(0, max(x) * 1.05)
ax.set_ylim(0, max(y) * 1.15 if max(y) > 0 else 10)

ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(loc='upper left')

# Add linear fit reference
if len(x) > 2 and max(y) > 0:
    # Fit line through origin area
    x_arr = np.array(x)
    y_arr = np.array(y)
    # Simple linear regression
    slope = np.sum(x_arr * y_arr) / np.sum(x_arr ** 2)
    x_fit = np.linspace(0, max(x), 100)
    y_fit = slope * x_fit
    ax.plot(x_fit, y_fit, '--', color='gray', alpha=0.5, linewidth=1.5,
            label=f'Linear fit (slope={slope:.3f})')
    ax.legend(loc='upper left')

plt.tight_layout()

#%% Save Plot
if SAVE_PLOTS:
    filepath = PLOTS_DIR / 'capacity_scaling_rho05.png'
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"\nSaved to: {filepath}")

#%% Summary
print("\n" + "=" * 70)
print("VISUALIZATION COMPLETE")
print("=" * 70)
print(f"\nResults for rho = {RHO_FILTER}:")
print(f"  Network sizes: {min(x)} to {max(x)}")
print(f"  Capacity range: {min(y)} to {max(y)} patterns")
if len(x) > 2 and max(y) > 0:
    print(f"  Approx. capacity ratio: {slope:.3f} patterns per neuron")
print(f"\nPlot saved to: {PLOTS_DIR}")

plt.show()
