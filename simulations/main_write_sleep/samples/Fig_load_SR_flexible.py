#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import matplotlib as mpl

# =============================================================================
# CONFIGURATION SECTION - Modify these parameters as needed
# =============================================================================

# Parameter that varies in the simulation (e.g., "beta", "leak", "delta", etc.)
VARYING_PARAM = "leak"

# LaTeX symbol for the varying parameter (e.g., r"\lambda", r"\beta", etc.)
# Set to None to use the parameter name as-is
PARAM_LATEX_SYMBOL = r"\lambda"

# Values to plot - set to None to use all available values, or specify a list
# Examples:
#   VALUES_TO_PLOT = None           # Use all values
#   VALUES_TO_PLOT = [0.05, 0.1, 1.0]  # Use specific values
VALUES_TO_PLOT = None

# Data directory
DATA_DIR = "../../data/sleep_simulations/Fig_load_SR_leak"

# Output plot filename (will be auto-generated based on varying parameter)
OUTPUT_FILENAME = None  # Set to None for auto-generated name

# Save plots to files
SAVE_PLOTS = True  # Set to False to only display plots without saving

# =============================================================================
# END OF CONFIGURATION
# =============================================================================

# Update the styling
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

def equally_spaced_from_array(arr, n, ratio_taken):
    return arr[np.linspace(0, (len(arr)*ratio_taken)-1, n, dtype=int)]

def relative_iter(row, eta):
    return row['query_iter'] == int(eta*row['num_patterns'])

def get_spaced_indices(a, n, num_ticks=4):
    return np.linspace(a, n, num_ticks, dtype=int)

def get_param_label(param_name, latex_symbol=None):
    """Return a LaTeX-formatted label for the parameter.

    Args:
        param_name: Name of the parameter
        latex_symbol: Optional LaTeX symbol override. If None, uses param_name as-is.
    """
    if latex_symbol is not None:
        return latex_symbol
    return param_name

#%%
# Read the CSV file
data = pd.read_csv(DATA_DIR + '/all_simulation_data.csv')

# Verify that the varying parameter exists in the data
if VARYING_PARAM not in data.columns:
    available_params = [col for col in data.columns if col not in
                       ['sim_ID', 'query_iter', 'nb_fnd_pat', 'success_ratio',
                        'error_ratio', 'is_error_before_all_fnd']]
    raise ValueError(f"Parameter '{VARYING_PARAM}' not found in data. "
                    f"Available columns: {available_params}")

#%%
# Calculate the ratio of successfully queried patterns
data['success_ratio'] = data['nb_fnd_pat'] / (data['num_patterns'])
data['num_patterns'] = data['num_patterns'].astype(int)
data['error_ratio'] = 1
data['is_error_before_all_fnd'] = False

#%%
all_num_patterns = np.sort(data['num_patterns'].unique())
all_net_sizes = np.sort(data['network_size'].unique())
all_param_values = np.sort(data[VARYING_PARAM].unique())

# Determine which values to plot
if VALUES_TO_PLOT is None:
    values_to_plot = all_param_values
else:
    # Filter to only include values that exist in the data
    values_to_plot = [v for v in VALUES_TO_PLOT if v in all_param_values]
    if not values_to_plot:
        raise ValueError(f"None of the specified values {VALUES_TO_PLOT} "
                        f"found in data. Available values: {list(all_param_values)}")

print(f"Varying parameter: {VARYING_PARAM}")
print(f"Values to plot: {values_to_plot}")

#%%
x_tick_indices = get_spaced_indices(1, len(all_net_sizes)-1, 4)
y_tick_indices = get_spaced_indices(1, len(all_num_patterns)-1, 7)

#%%
# ── Vectorised analysis (replaces the slow loops) ───────────────────────────
df = data.copy()
df["all_found"] = df["nb_fnd_pat"] == df["num_patterns"]

# 1️⃣ First iteration where every pattern was found (per simulation)
first_all_found = (
    df.loc[df["all_found"]]
      .groupby("sim_ID")["query_iter"]
      .min()
      .rename("first_iter_all_fnd")
)
df = df.merge(first_all_found, on="sim_ID", how="left")
df["is_error_before_all_fnd"] = df["first_iter_all_fnd"].isna()
df["first_iter_all_fnd"] = (df["first_iter_all_fnd"].fillna(0).astype(int) + 1)

# 2️⃣ Keep only the last iteration of each simulation
idx_last = df.groupby("sim_ID")["query_iter"].idxmax()
df_last = df.loc[idx_last].copy()

# 3️⃣ Error statistics per (network_size, num_patterns)
err_stats = (
    df_last.groupby(["network_size", "num_patterns"])["is_error_before_all_fnd"]
           .agg(any_error="any", n_errors="sum")
           .reset_index()
)
df_last = df_last.merge(err_stats, on=["network_size", "num_patterns"])

#%%
# ── Heat-maps: % sims w/o error & first-iteration (grey = no convergence) ─

n_cols = len(values_to_plot)
param_label = get_param_label(VARYING_PARAM, PARAM_LATEX_SYMBOL)

# 1️⃣ global colour-scale limits (so all panels share one scale)
global_max_error = 0
global_max_iter = 0

for param_val in values_to_plot:
    sub = df_last.loc[df_last[VARYING_PARAM] == param_val].copy()
    sub["is_not_error_before_all_fnd"] = ~sub["is_error_before_all_fnd"]
    sub.loc[sub["is_error_before_all_fnd"], "first_iter_all_fnd"] = np.nan

    pt_err = sub.pivot_table(
        values="is_not_error_before_all_fnd",
        index="num_patterns",
        columns="network_size",
    )
    global_max_error = max(global_max_error, pt_err.values.max() * 100)

    pt_itr = sub.pivot_table(
        values="first_iter_all_fnd",
        index="num_patterns",
        columns="network_size",
    )
    if not np.all(np.isnan(pt_itr.values)):
        global_max_iter = max(global_max_iter, np.nanmax(pt_itr.values))

# 2️⃣ colormap for 2nd row with grey "no-data" colour
default_cmap_name = plt.rcParams["image.cmap"]
cmap_iter = mpl.cm.get_cmap(default_cmap_name).copy()
cmap_iter.set_bad(color="lightgrey")

# 3️⃣ make the figure
r = 1.1
# Adjust figure width based on number of columns
fig_width = max(9, 3 * n_cols) / r
fig, axes = plt.subplots(2, n_cols, figsize=(fig_width, 8 / r),
                         sharex=True, sharey=True, squeeze=False)

for i, param_val in enumerate(values_to_plot):
    sub = df_last.loc[df_last[VARYING_PARAM] == param_val].copy()
    sub["is_not_error_before_all_fnd"] = ~sub["is_error_before_all_fnd"]
    sub.loc[sub["is_error_before_all_fnd"], "first_iter_all_fnd"] = np.nan

    # first row: % sims without errors
    pt_err = sub.pivot_table(
        values="is_not_error_before_all_fnd",
        index="num_patterns",
        columns="network_size",
    )
    im1 = axes[0, i].imshow(pt_err * 100,
                            vmin=0, vmax=global_max_error)
    axes[0, i].set_title(rf"${param_label}={param_val}$")
    axes[0, i].invert_yaxis()
    axes[0, i].grid(False)

    # second row: first iteration (NaNs → grey)
    pt_itr = (
        sub.pivot_table(
            values="first_iter_all_fnd",
            index="num_patterns",
            columns="network_size",
        )
        .reindex(index=all_num_patterns, columns=all_net_sizes)
    )
    masked_itr = np.ma.masked_invalid(pt_itr.values)
    im2 = axes[1, i].imshow(masked_itr,
                            vmin=0, vmax=global_max_iter,
                            cmap=cmap_iter)
    axes[1, i].invert_yaxis()
    axes[1, i].grid(False)

# 4️⃣ tidy ticks
all_net_sizes = np.sort(df_last["network_size"].unique())
all_num_patterns = np.sort(df_last["num_patterns"].unique())
x_tick_indices = get_spaced_indices(1, len(all_net_sizes) - 1, 4)
y_tick_indices = get_spaced_indices(1, len(all_num_patterns) - 1, 7)

for row in axes:
    for ax in row:
        ax.tick_params(axis='both', which='both', bottom=True, left=True, top=False, right=False)
        ax.set_xticks(x_tick_indices, all_net_sizes[x_tick_indices])
        ax.set_yticks(y_tick_indices, all_num_patterns[y_tick_indices])

# Add single colorbar for first row (error rates)
cbar1_ax = fig.add_axes([0.92, 0.56, 0.02, 0.3])
cbar1 = fig.colorbar(im1, cax=cbar1_ax)
cbar1.set_ticks(np.linspace(0, 100, 5))
cbar1.set_ticklabels([f'{int(val)}' for val in np.linspace(0, 100, 5)])

# Add single colorbar for second row (first iteration)
cbar2_ax = fig.add_axes([0.92, 0.14, 0.02, 0.3])
cbar2 = fig.colorbar(im2, cax=cbar2_ax)
cbar2.set_ticks(np.linspace(0, global_max_iter, 5))
cbar2.set_ticklabels([f'{int(val)}' for val in np.linspace(1, global_max_iter, 5)])

fig.text(0.51, 0.04, 'Network size', ha='center', va='center')
fig.text(0.04, 0.49, 'Nb stored pattern', ha='left', va='center', rotation=90)

# Generate output filename and save if enabled
if SAVE_PLOTS:
    if OUTPUT_FILENAME is None:
        output_name = f"./plots/Fig_load_SR_{VARYING_PARAM}.png"
    else:
        output_name = OUTPUT_FILENAME
    plt.savefig(output_name, dpi=300)
    print(f"Saved main plot to: {output_name}")

plt.show()

#%%
# ── Detailed heatmap for a single parameter value ──────────────────────────
# Use the first value from values_to_plot for the detailed view
detail_param_val = values_to_plot[0]
sub = df_last.loc[df_last[VARYING_PARAM] == detail_param_val].copy()

# pivot: count rows where is_error_before_all_fnd == False (= success)
pivot_success = sub.pivot_table(
    values="is_error_before_all_fnd",
    index="num_patterns",
    columns="network_size",
    aggfunc=lambda x: (~x).sum(),
    fill_value=0
)

fig, ax = plt.subplots(figsize=(12, 12))
sns.heatmap(
    pivot_success,
    annot=True, fmt="d",
    ax=ax, annot_kws={"fontsize": 18},
    cbar=False
)

ax.set_xlabel("Network size", fontsize=22)
ax.set_ylabel("Nb stored patterns", fontsize=22)
ax.tick_params(axis='both', which='both', bottom=True, left=True, top=False, right=False)

plt.xticks(rotation=45, ha="right", fontsize=18)
plt.yticks(rotation=0, fontsize=18)
ax.invert_yaxis()
plt.tight_layout()

if SAVE_PLOTS:
    detail_output_name = f"./plots/Fig_detailed_recovery_{VARYING_PARAM}.png"
    plt.savefig(detail_output_name)
    print(f"Saved detailed plot to: {detail_output_name}")

plt.show()

# %%
