#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Update the styling
sns.set_style("darkgrid")  # or "whitegrid", "dark", "white", "ticks"
sns.set_context("paper", font_scale=1.5)  # or "paper", "talk", "poster"
def equally_spaced_from_array(arr, n, ratio_taken):
    return arr[np.linspace(0, (len(arr)*ratio_taken)-1, n, dtype=int)]

def relative_iter(row,eta):
    # Replace this with your specific condition
    return row['query_iter']==int(eta*row['num_patterns'])

def get_spaced_indices(n, num_ticks=4):
    return np.linspace(0, n - 1, num_ticks, dtype=int)

plt.rcParams.update({'font.size': 15})
#%%
# Read the CSV file
# Fig_load_SR_average_new_inh_plas_many_betta_larger_networks
# myDir = "../../data/all_data_splited/sleep_simulations/Fig_load_SR_average_new_inh_plas_many_betta_larger_networks"
myDir = "../../data/all_data_splited/sleep_simulations/Fig_load_SR_average_new_inh_plas_big_simulations_many_correlations_2025"
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
all_correlation = np.sort(data['noise_level'].unique())
all_repetitions= np.sort(data['repetitions'].unique())
nb_sim_one_parameter = len(all_repetitions)
#%%
x_tick_indices = get_spaced_indices(len(all_net_sizes),4)
y_tick_indices = get_spaced_indices(len(all_num_patterns),10)
#%%
data_correlations_1 = dict()
for i,correlation in enumerate(all_correlation):
    data_one_correlation = data[data['noise_level']==correlation]
    sim_IDs=np.sort(data['sim_ID'].unique())
    for sim_id in sim_IDs:
        is_error = False
        mask = data_one_correlation['sim_ID']==sim_id
        sim_data = data_one_correlation[mask]
        sim_data_all_patterns_fnd = sim_data[sim_data['nb_fnd_pat']==sim_data['num_patterns']]
        first_iter_all_fnd= 0
        if len(sim_data_all_patterns_fnd)==0:
            is_error = True
        first_iter_all_fnd_data=sim_data_all_patterns_fnd[sim_data_all_patterns_fnd['query_iter']==sim_data_all_patterns_fnd["query_iter"].min()]
        first_iter_all_fnd = first_iter_all_fnd_data["query_iter"]+1

        data_one_correlation.loc[mask, 'is_error_before_all_fnd'] = is_error
        data_one_correlation.loc[mask, 'first_iter_all_fnd'] = first_iter_all_fnd
        data_one_correlation.loc[mask, 'noned'] = 1
        data_one_correlation.loc[mask, 'more_than_one_noned'] = 1
    data_correlations_1[correlation]=data_one_correlation
data_correlations = dict()
#%%
for i,correlation in enumerate(all_correlation):
    data_one_correlation=data_correlations_1[correlation]
    # Keep only the last iteration as representent of the simulation
    data_one_correlation = data_one_correlation.loc[data_one_correlation.groupby("sim_ID")["query_iter"].idxmax()]
    for n in all_num_patterns:
        for s in all_net_sizes:
            is_error = data_one_correlation.loc[(data_one_correlation['network_size'] == s) & (data_one_correlation['num_patterns'] == n),"is_error_before_all_fnd"]
            any_error = np.any(list(is_error))
            # more_than_one_error = len(list(is_error))>((nb_sim_one_parameter*80)/100)
            more_than_one_error = np.sum(list(is_error))>2
            if any_error:
                data_one_correlation.loc[(data_one_correlation['network_size'] == s) & (data_one_correlation['num_patterns'] == n),'noned']=None
            if more_than_one_error:
                data_one_correlation.loc[(data_one_correlation['network_size'] == s) & (data_one_correlation['num_patterns'] == n),'more_than_one_noned']=None

    data_one_correlation["first_iter_all_fnd_normalized"] = data_one_correlation['first_iter_all_fnd']/data_one_correlation["num_patterns"]

    data_correlations[correlation]=data_one_correlation # store the specific data for later
#%%
r = 1.1
fig, axes = plt.subplots(2,5, figsize=(20/r,8/r), sharey=True,sharex=True)
for i, correlation in enumerate(all_correlation):
    data_one_correlation = data_correlations[correlation]

    pivot_table = data_one_correlation.pivot_table(values='is_error_before_all_fnd', index='num_patterns', columns='network_size')
    im1 = axes[0][i].imshow(pivot_table*100, cmap="Reds", vmin=0)
    axes[0][i].set_xticks(x_tick_indices, all_net_sizes[x_tick_indices])
    axes[0][i].set_yticks(y_tick_indices, all_num_patterns[y_tick_indices])
    axes[0][i].grid(False)
    axes[0][i].invert_yaxis()
    axes[0][i].set_title(r'$\rho$ = '+str(1-correlation))

    plt.colorbar(im1, ax=axes[0][i], shrink=0.8)

    # FIRST ITER FIGS
    pivot_table = data_one_correlation.pivot_table(values='first_iter_all_fnd_normalized', 
                                                   index='num_patterns', columns='network_size')
    max_val = np.nanmax(pivot_table.values)
    im2 = axes[1][i].imshow(pivot_table, vmin=0, vmax=max_val)
    axes[1][i].set_xticks(x_tick_indices, all_net_sizes[x_tick_indices])
    axes[1][i].set_yticks(y_tick_indices, all_num_patterns[y_tick_indices])
    axes[1][i].grid(False)
    axes[1][i].invert_yaxis()
    cbar = plt.colorbar(im2, ax=axes[1][i], shrink=0.8)
    # Manually set ticks to include the max value
    cbar.set_ticks(np.linspace(0,max_val,5))
    cbar.set_ticklabels([f'{val:.1f}' for val in np.linspace(0,max_val,5)])

# fig.colorbar(im2, ax=axes[1, :],shrink=0.8)
fig.text(0.47, 0.02, 'Network size', ha='center', va='center')
fig.text(0.07, 0.49, 'Nb stored pattern', ha='left', va='center',rotation=90)
# %%
# %%
# At the end of your script:
# sns.set(font_scale=2)
# Create a pivot table that counts how many simulations ended with an error before all found
pivot_table = data_correlations[0.5].pivot_table(
    values='is_error_before_all_fnd', 
    index='num_patterns', 
    columns='network_size', 
    aggfunc='sum'  # Count how many True values we got
)

fig, ax = plt.subplots(figsize=(20, 20))  # Large figure size
heatmap = sns.heatmap(pivot_table, annot=True, fmt="g", cmap="Reds", ax=ax,annot_kws={"fontsize": 24},cbar=False)
# Adjust the colorbar font size

ax.set_xlabel("Network size", fontsize=28)
ax.set_ylabel("Nb stored pattern", fontsize=28)
plt.xticks(rotation=45, ha='right', fontsize=25)
plt.yticks(rotation=0, fontsize=25)
ax.invert_yaxis()
#plt.title("Number of simulations with errors before all patterns found", fontsize=24, pad=20)
plt.tight_layout()
plt.show()
# %%
