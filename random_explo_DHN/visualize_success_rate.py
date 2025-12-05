#!/usr/bin/env python3
"""
DHN Success Rate Visualization
Adapted from CHN visualization script
Simplified for direct 0/1 success values
"""
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Update the styling - EXACT SAME AS CHN
sns.set_style("darkgrid")  # or "whitegrid", "dark", "white", "ticks"
sns.set_context("paper", font_scale=1.5)  # or "paper", "talk", "poster"
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
    'font.weight' : 'bold'
})

def get_spaced_indices(a, n, num_ticks=4):
    """Get evenly spaced indices for tick positions"""
    return np.linspace(a, n, num_ticks, dtype=int)

#%%
# Read the DHN CSV file
data = pd.read_csv('correlation_random_query_perceptron_success/all_simulation_data.csv')

# Optional: Filter by learning rule if needed
# data = data[data['learning_rule'] == 1]  # 1=Perceptron, 0=Hebbian

# Rename columns to match CHN naming convention for code consistency
data = data.rename(columns={
    'net_size': 'network_size',
    'nb_pat': 'num_patterns'
})

#%%
# SIMPLIFIED SUCCESS CALCULATION - Direct from data!
# No need for complex iteration analysis - success is already 0/1
df = data.copy()
df["is_not_error_before_all_fnd"] = df["success"] == 1  # 1=success, 0=failure
df["is_error_before_all_fnd"] = df["success"] == 0

#%%
# Get unique parameter values
all_num_patterns = np.sort(df['num_patterns'].unique())
all_net_sizes = np.sort(df['network_size'].unique())
all_correlation = np.sort(df['noise_level'].unique())

#%%
# Tick indices for axes
x_tick_indices = get_spaced_indices(1, len(all_net_sizes)-1, 4)
y_tick_indices = get_spaced_indices(1, len(all_num_patterns)-1, 7)

#%%
# ── Heat-map of success rates ──────────────────────────────────────────────

# Correlation values to plot (same as CHN)
correlation_to_plot = [0.0, 0.25, 0.5, 0.75, 1.0]
n_cols = len(correlation_to_plot)

# Global colour-scale limit for success rate
global_max_error = 0

for noise_level in correlation_to_plot:
    sub = df.loc[df["noise_level"] == noise_level].copy()

    # Success-rate pivot table (average across repetitions if any)
    pt_err = sub.pivot_table(
        values="is_not_error_before_all_fnd",
        index="num_patterns",
        columns="network_size",
        aggfunc='mean'  # Average across repetitions
    )
    global_max_error = max(global_max_error, pt_err.values.max() * 100)

#%%
# ── Create the heatmap figure (SINGLE ROW - no iteration count) ──────────

# Figure dimensions (adjusted for single row)
r = 1.1
fig, axes = plt.subplots(1, 5, figsize=(15/r, 4/r), sharey=True, sharex=True)

for i, correlation in enumerate(correlation_to_plot):
    sub = df.loc[df["noise_level"] == correlation].copy()

    # Success rate heatmap (% of simulations successful)
    pt_err = sub.pivot_table(
        values="is_not_error_before_all_fnd",
        index="num_patterns",
        columns="network_size",
        aggfunc='mean'  # Average across repetitions
    )

    im = axes[i].imshow(pt_err * 100,
                        vmin=0, vmax=global_max_error)
    axes[i].set_title(rf"$\rho={correlation}$")
    axes[i].invert_yaxis()
    axes[i].grid(False)

# Tidy ticks
all_net_sizes = np.sort(df["network_size"].unique())
all_num_patterns = np.sort(df["num_patterns"].unique())
x_tick_indices = get_spaced_indices(1, len(all_net_sizes) - 1, 4)
y_tick_indices = get_spaced_indices(1, len(all_num_patterns) - 1, 7)

for ax in axes:
    ax.tick_params(axis='both', which='both', bottom=True, left=True, top=False, right=False)
    ax.set_xticks(x_tick_indices)
    ax.set_xticklabels(all_net_sizes[x_tick_indices])
    ax.set_yticks(y_tick_indices)
    ax.set_yticklabels(all_num_patterns[y_tick_indices])

# Add single colorbar (centered vertically for single row)
cbar_ax = fig.add_axes([0.92, 0.25, 0.01, 0.5])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_ticks(np.linspace(0, 100, 5))
cbar.set_ticklabels([f'{int(val)}' for val in np.linspace(0, 100, 5)])

# Axis labels
fig.text(0.51, 0.04, 'Network size', ha='center', va='center')
fig.text(0.07, 0.5, 'Nb stored pattern', ha='left', va='center', rotation=90)

# Save figure
plt.savefig("./plots/DHN_correlation_success_heatmap.png", dpi=300, bbox_inches='tight')
print("Visualization saved to: ./plots/DHN_correlation_success_heatmap.png")

#%%
