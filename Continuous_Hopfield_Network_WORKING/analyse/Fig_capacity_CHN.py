#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def equally_spaced_from_array(arr, n, ratio_taken):
    return arr[np.linspace(0, (len(arr)*ratio_taken)-1, n, dtype=int)]

def relative_iter(row,eta):
    # Replace this with your specific condition
    return row['query_iter']==int(eta*row['num_patterns'])
#%%
# Read the CSV file
myDir = '..\\..\\..\\data\\all_data_splited\\query_simulations\\FIG_capacity_CHN_4'
data = pd.read_csv(myDir+'\\all_simulation_data.csv')
# data = data[data['network_size'] != 250]
#%%
# Calculate the ratio of successfully queried patterns
data['success_ratio'] = data['nb_found_patterns'] / (data['num_patterns'])
data['num_patterns'] = data['num_patterns'].astype(int)
#%%
all_query_ratio_rnd_bits = np.sort(list(set(data['ratio_flips_querying'])))[::3]
#%%
all_query_ratio_rnd_bits
#%%
nb_plot = 4
#%%
size = 4
plt.rcParams.update({'font.size': 25})
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
    im = axes[i].imshow(pivot_table, aspect='auto', origin='upper', cmap='viridis', extent=[
        pivot_table.columns.min(), pivot_table.columns.max(), pivot_table.index.max(), pivot_table.index.min()],vmin=0,vmax=1)
    # ax.set_ylabel('Number of stored patterns')
    # ax.set_xlabel('Network size')
    # y_ticks = [i for i in range(5,25)]
    # ax.set_yticks(y_ticks)
    # ax.set_yticklabels([f'{tick:.0f}' for tick in y_ticks])

    # x_min, x_max = pivot_table.columns.min(), pivot_table.columns.max()
    # x_range = x_max - x_min
    # x_ticks = np.linspace(x_min + x_range * 0.1, x_max, 6)  # Shift the first tick to the right
    # ax2.set_xticks(x_ticks)
    # ax2.set_xticklabels([f'{tick:.0f}' for tick in x_ticks])
    # Add colorbar to the right of the phase diagram

    # # Plot the crosstalk
    # ax2.plot(aggregated_data['nb_rand_bits'], aggregated_data['abs_mean_crosstalk'])
    # ax2.set_xlabel('Number of random bits',labelpad=20, fontsize = 22)
    # ax2.set_ylabel('Mean Crosstalk',labelpad = 20, fontsize=22)
    # ax2.set_xlim(ax1.get_xlim())  # Match x-axis limits with the phase diagram

    # # Set custom ticks for y-axis (Mean Crosstalk)
    # y_min, y_max = ax2.get_ylim()
    # y_ticks = np.linspace(y_min, y_max, 3)
    # ax2.set_yticks(y_ticks)
    # ax2.set_yticklabels([f'{tick:.2f}' for tick in y_ticks])
    # # ax2.tick_params(axis='both', which='major', labelsize=22)  # Adjust this value as needed
    # # Adjust layout and display
    # plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust the rect to make room for the colorbar
    # cbar_ax = fig.add_axes([0.92, 0.3, 0.02, 0.6])  # [left, bottom, width, height]
cbar_ax = fig.add_axes([0.93, 0.15, 0.009, 0.7])  # [left, bottom, width, height]
cbar = fig.colorbar(im,cax=cbar_ax)
cbar.set_label('Success Ratio',labelpad = 14)
# plt.tight_layout(rect=[0,0,1,1])
fig.text(0.5, 0.07, 'Network size', ha='center', va='center')
fig.text(0.08, 0.5, 'number of stored pattern', ha='left', va='center',rotation=90)
plt.show()

# %%
