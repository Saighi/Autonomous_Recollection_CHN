#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Function to get approximately equally spaced indices
def get_spaced_indices(n, num_ticks=4):
    return np.linspace(0, n - 1, num_ticks, dtype=int)
plt.rcParams.update({'font.size': 25})
#%%
# Read the CSV file
myDir = '..\\..\\..\\data\\all_data_splited\\query_simulations\\FIG_capacity_CHN_5'
data = pd.read_csv(myDir+'\\all_simulation_data.csv')
# data = data[data['network_size'] != 250]
#%%
# Calculate the ratio of successfully queried patterns
data['success_ratio'] = data['nb_found_patterns'] / (data['num_patterns'])
data['num_patterns'] = data['num_patterns'].astype(int)
#%%
all_query_ratio_rnd_bits = np.sort(data['ratio_flips_querying'].unique())
#%%
all_query_ratio_rnd_bits = [0.01,0.02,0.03,0.15]
#%%
nb_plot = 4
#%%
all_num_patterns = np.sort(data['num_patterns'].unique())
all_net_sizes = np.sort(data['network_size'].unique())
#%%
all_num_patterns
data.loc[data['num_patterns'] == 5, 'success_ratio'] = 1
#%%
# Get indices for x and y ticks
x_tick_indices = get_spaced_indices(len(all_net_sizes))
y_tick_indices = get_spaced_indices(len(all_num_patterns))
#%%
size = 4
# fig, ax = plt.subplots(1, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [6, 1]}, sharex=True)
fig, axes = plt.subplots(1, nb_plot, figsize=(size*nb_plot,size), sharey=True,sharex=True)
for i,v in enumerate(all_query_ratio_rnd_bits):

    ratio_flip_data = data[data['ratio_flips_querying']==v]
    # fig, ax = plt.subplots(1, 1, figsize=(12, 12), sharex=True)
    # Create a pivot table for the phase diagram
    # print(data_iter['query_iter'])
    pivot_table = ratio_flip_data.pivot_table(values='success_ratio', index='num_patterns', columns='network_size')
    # pivot_table = pivot_table.sort_index(ascending=False)  # Sort index in descending order
    # Create the figure and axes
    # Plot the phase diagram
    im = axes[i].imshow(pivot_table, aspect='auto', origin='upper', cmap='viridis')
    axes[i].set_xticks(x_tick_indices)
    axes[i].set_yticks(y_tick_indices)
    axes[i].set_xticklabels(all_net_sizes[x_tick_indices])
    axes[i].set_yticklabels(all_num_patterns[y_tick_indices])
    axes[i].set_title("o="+str(v))

cbar_ax = fig.add_axes([0.93, 0.15, 0.009, 0.7])  # [left, bottom, width, height]
cbar = fig.colorbar(im,cax=cbar_ax)
cbar.set_label('Recovery Ratio',labelpad = 14)
# plt.tight_layout(rect=[0,0,1,1])
fig.text(0.5, -0.1, 'Network size', ha='center', va='center')
fig.text(0.045, 0.5, 'number of stored pattern', ha='left', va='center',rotation=90)
plt.show()

# %%
