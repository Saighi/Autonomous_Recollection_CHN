#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
# Update the styling
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
def equally_spaced_from_array(arr, n, ratio_taken):
    return arr[np.linspace(0, (len(arr)*ratio_taken)-1, n, dtype=int)]

def relative_iter(row,eta):
    # Replace this with your specific condition
    return row['query_iter']==int(eta*row['num_patterns'])

def get_spaced_indices(a,n, num_ticks=4):
    return np.linspace(a, n, num_ticks, dtype=int)

#%%
# Read the CSV file
# Fig_load_SR_average_new_inh_plas_many_betta_larger_networks
# myDir = "../../data/all_data_splited/sleep_simulations/Fig_load_SR_average_new_inh_plas_many_betta_larger_networks"
# myDir = "../../data/all_data_splited/sleep_simulations/Fig_load_SR_average_new_inh_plas_big_simulations_many_beta"
myDir = "/home/saighi/Desktop/data/all_data_splited/trained_networks_fast/Fig_load_SR_many_leak_parameter"
data = pd.read_csv(myDir+'/all_simulation_data.csv')
# data = data[data['delta'] == 0.1]
#%%
# Calculate the ratio of successfully queried patterns
data['success_ratio'] = data['nb_fnd_pat'] / (data['num_patterns'])
data['num_patterns'] = data['num_patterns'].astype(int)
data['error_ratio'] = 1
data['is_error_before_all_fnd'] = False
#%%
all_num_patterns = np.sort(data['num_patterns'].unique())
all_net_sizes = np.sort(data['network_size'].unique())
all_beta = np.sort(data['beta'].unique())
# all_repetitions= np.sort(data['repetitions'].unique())
# nb_sim_one_parameter = len(all_repetitions)
#%%
x_tick_indices = get_spaced_indices(1,len(all_net_sizes)-1,4)
y_tick_indices = get_spaced_indices(1,len(all_num_patterns)-1,7)
#%%

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
# ── Heat-maps of (absence of) errors and first-found iteration ──────────────

# 0️⃣  choose which β values to display
# betas_to_plot = np.sort(df_last["beta"].unique())        # or slice / mask as you like
betas_to_plot = [0.05,0.1,1.0]        # or slice / mask as you like
n_cols        = len(betas_to_plot)

# 1️⃣  global colour-scale limits (so every panel shares one scale)
global_max_error = 0
global_max_iter  = 0

for beta in betas_to_plot:
    sub = df_last.loc[df_last["beta"] == beta].copy()
    sub["is_not_error_before_all_fnd"] = ~sub["is_error_before_all_fnd"]

    # success-rate (1st row)
    pt_err = sub.pivot_table(
        values="is_not_error_before_all_fnd",
        index="num_patterns",
        columns="network_size",
    )
    global_max_error = max(global_max_error, pt_err.values.max() * 100)

    # first-iteration (2nd row)
    pt_itr = sub.pivot_table(
        values="first_iter_all_fnd",
        index="num_patterns",
        columns="network_size",
    )
    global_max_iter = max(global_max_iter, np.nanmax(pt_itr.values))

#%%
# ── Heat-maps: % sims w/o error  &  first-iteration (grey = no convergence) ─

# 0️⃣  which β’s to show
# betas_to_plot = np.sort(df_last["beta"].unique())
betas_to_plot = [0.05,0.1,1.0]        # or slice / mask as you like

n_cols        = len(betas_to_plot)

# 1️⃣  global colour-scale limits (so all panels share one scale)
global_max_error = 0
global_max_iter  = 0

for beta in betas_to_plot:
    sub = df_last.loc[df_last["beta"] == beta].copy()
    sub["is_not_error_before_all_fnd"] = ~sub["is_error_before_all_fnd"]
    sub.loc[sub["is_error_before_all_fnd"], "first_iter_all_fnd"] = np.nan  # <-- drop

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

# 2️⃣  colormap for 2nd row with grey “no-data” colour
import matplotlib as mpl   # at top of your file if not already imported
default_cmap_name = plt.rcParams["image.cmap"]          # e.g. "viridis"
cmap_iter = mpl.cm.get_cmap(default_cmap_name).copy()   # copy keeps the original intact
cmap_iter.set_bad(color="lightgrey")        

# 3️⃣  make the figure
r = 1.1
fig, axes = plt.subplots(2, n_cols, figsize=(9 / r, 8 / r),
                         sharex=True, sharey=True)

for i, beta in enumerate(betas_to_plot):
    sub = df_last.loc[df_last["beta"] == beta].copy()
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
    axes[0, i].set_title(rf"$\beta={beta}$")
    axes[0, i].invert_yaxis()
    axes[0, i].grid(False)

    # second row: first iteration (NaNs → grey)
    pt_itr = (
        sub.pivot_table(
            values="first_iter_all_fnd",
            index="num_patterns",
            columns="network_size",
        )
        .reindex(index=all_num_patterns, columns=all_net_sizes)  # ← add this
    )
    masked_itr = np.ma.masked_invalid(pt_itr.values)
    im2 = axes[1, i].imshow(masked_itr,
                            vmin=0, vmax=global_max_iter,
                            cmap=cmap_iter)
    axes[1, i].invert_yaxis()
    axes[1, i].grid(False)

# 4️⃣  tidy ticks
all_net_sizes    = np.sort(df_last["network_size"].unique())
all_num_patterns = np.sort(df_last["num_patterns"].unique())
x_tick_indices   = get_spaced_indices(1, len(all_net_sizes) - 1, 4)
y_tick_indices   = get_spaced_indices(1, len(all_num_patterns) - 1, 7)

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
fig.text(0.04, 0.49, 'Nb stored pattern', ha='left', va='center',rotation=90)
plt.savefig("./plots/Fig_load_SR_average_betas_diag_inh.png",dpi=300)
plt.show()



#%%
beta_val = 0.1          # ← pick the β you want to inspect
sub = df_last.loc[df_last["beta"] == beta_val].copy()

# pivot: count rows where is_error_before_all_fnd == False  (= success)
pivot_success = sub.pivot_table(
    values="is_error_before_all_fnd",
    index="num_patterns",
    columns="network_size",
    aggfunc=lambda x: (~x).sum(),      # count successes in that bucket
    fill_value=0                       # optional: show 0 instead of NaN
)

fig, ax = plt.subplots(figsize=(12, 12))
sns.heatmap(
    pivot_success,
    annot=True, fmt="d",
    ax=ax, annot_kws={"fontsize": 18},
    cbar=False            # ← suppress the colour-bar
)

# cosmetics
ax.set_xlabel("Network size", fontsize=22)
ax.set_ylabel("Nb stored patterns", fontsize=22)
ax.tick_params(axis='both', which='both', bottom=True, left=True, top=False, right=False)

plt.xticks(rotation=45, ha="right", fontsize=18)
plt.yticks(rotation=0, fontsize=18)
ax.invert_yaxis()          # keep your preferred orientation
plt.tight_layout()

plt.savefig("./plots/Fig_detailed_recovery_data_diag_inh.png")
plt.show()

# %%
