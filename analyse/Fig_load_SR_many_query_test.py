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
myDir = "../../data/all_data_splited/sleep_simulations/Fig_load_SR_query_test"
data = pd.read_csv(myDir+'/all_simulation_data.csv')
# data = data[data['delta'] == 0.1]

#%%
all_num_patterns = np.sort(data['num_patterns'].unique())
all_net_sizes = np.sort(data['network_size'].unique())
all_ratio_query = np.sort(data["ratio_flip_querying"].unique())
# all_repetitions= np.sort(data['repetitions'].unique())
# nb_sim_one_parameter = len(all_repetitions)
#%%
x_tick_indices = get_spaced_indices(1,len(all_net_sizes)-1,4)
y_tick_indices = get_spaced_indices(1,len(all_num_patterns)-1,7)
#%%

# 2️⃣  colormap for 2nd row with grey “no-data” colour
import matplotlib as mpl   # at top of your file if not already imported
default_cmap_name = plt.rcParams["image.cmap"]          # e.g. "viridis"
cmap_iter = mpl.cm.get_cmap(default_cmap_name).copy()   # copy keeps the original intact
cmap_iter.set_bad(color="lightgrey")        

# 3️⃣  make the figure
r = 1.1
fig, axes = plt.subplots(1, 5, figsize=(14 / r, 4 / r),
                         sharex=True, sharey=True)

ratio_to_plots = all_ratio_query

for i, ratio in enumerate(ratio_to_plots):
    sub = data.loc[data["ratio_flip_querying"] == ratio].copy()
    print(sub)
    # first row: % sims without errors
    pt_err = sub.pivot_table(
        values="nb_found_patterns",
        index="num_patterns",
        columns="network_size",
    )
    im1 = axes[i].imshow(pt_err,
                            vmin=0, vmax=1)
    axes[i].set_title(rf"$\delta={ratio}$")
    axes[i].invert_yaxis()
    axes[i].grid(False)



x_tick_indices   = get_spaced_indices(1, len(all_net_sizes) - 1, 4)
y_tick_indices   = get_spaced_indices(1, len(all_num_patterns) - 1, 7)


for ax in axes:        
    ax.tick_params(axis='both', which='both', bottom=True, left=True, top=False, right=False)

    ax.set_xticks(x_tick_indices, all_net_sizes[x_tick_indices])
    ax.set_yticks(y_tick_indices, all_num_patterns[y_tick_indices])

# Add single colorbar for first row (error rates)
cbar1_ax = fig.add_axes([0.92, 0.16, 0.02, 0.6])
cbar1 = fig.colorbar(im1, cax=cbar1_ax)

# cbar1.set_ticks(np.linspace(0, 100, 5))
# cbar1.set_ticklabels([f'{int(val)}' for val in np.linspace(0, 100, 5)])


fig.text(0.51, 0.04, 'Network size', ha='center', va='center')
fig.text(0.04, 0.49, 'Nb stored pattern', ha='left', va='center',rotation=90)
plt.savefig("./plots/Fig_load_SR_many_query_test.png",dpi=300)
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
