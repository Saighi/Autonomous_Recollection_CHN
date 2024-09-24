#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%
# Read the CSV file
data = pd.read_csv('..\\..\\..\\data\\all_data_splited\\trained_networks_fast\\figure_query_continuous_new_format\\all_simulation_data.csv')
#%%
# Calculate the ratio of successfully queried patterns
data['success_ratio'] = data['nb_found_patterns'] / (data['relative_num_patterns']*data['network_size'])
# data['nb_rand_bits'] = data['ratio_rand_bits']*data['network_size']
# data['nb_pat'] = data['ratio_nb_patterns']*data['network_size']
#%%
data_all_recovered = data.loc[data['success_ratio'] == 1].sort_values('network_size')
data_all_recovered = data_all_recovered.groupby('network_size').agg({'nb_found_patterns':'max'}).reset_index() 

#%%
plt.plot(data_all_recovered['network_size'], data_all_recovered['nb_found_patterns']/data_all_recovered['network_size'])
#%%
# Create a pivot table for the phase diagram
# pivot_table = data.pivot_table(values='success_ratio', index='nb_pat', columns='nb_rand_bits')
# # pivot_table = pivot_table.sort_index(ascending=False)  # Sort index in descending order
# #%%
# # Calculate mean absolute crosstalk
# data['abs_mean_crosstalk'] = np.abs(data['mean_crosstalk'])
# aggregated_data = data.groupby(['nb_rand_bits']).agg({'abs_mean_crosstalk':'mean'}).reset_index()

# #%%
# # Create the figure and axes
# plt.rcParams.update({'font.size': 25})
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [6, 1]}, sharex=True)
# # Plot the phase diagram
# im = ax1.imshow(pivot_table, aspect='auto', origin='upper', cmap='viridis', extent=[
#     pivot_table.columns.min(), pivot_table.columns.max(), pivot_table.index.max(), pivot_table.index.min()])
# # ax1.set_xlabel('Number of random bits')
# ax1.set_ylabel('Number of Patterns', labelpad=20)
# # ax1.set_title('Successfully queried patterns for a network of 200 units')
# y_ticks = np.linspace(pivot_table.index.min(), pivot_table.index.max(), 6)
# ax1.set_yticks(y_ticks)
# ax1.set_yticklabels([f'{tick:.0f}' for tick in y_ticks])

# x_min, x_max = pivot_table.columns.min(), pivot_table.columns.max()
# x_range = x_max - x_min
# x_ticks = np.linspace(x_min + x_range * 0.1, x_max, 6)  # Shift the first tick to the right
# ax2.set_xticks(x_ticks)
# ax2.set_xticklabels([f'{tick:.0f}' for tick in x_ticks])
# # Add colorbar to the right of the phase diagram
# cbar_ax = fig.add_axes([0.92, 0.3, 0.02, 0.6])  # [left, bottom, width, height]
# cbar = fig.colorbar(im, cax=cbar_ax)
# cbar.set_label('Success Ratio',labelpad = 20)

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
# plt.show()
# #%%