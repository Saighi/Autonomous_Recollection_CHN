#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
def equally_spaced_from_array(arr, n, ratio_taken):
    return arr[np.linspace(0, (len(arr)*ratio_taken)-1, n, dtype=int)]

def relative_iter(row,eta):
    # Replace this with your specific condition
    return row['query_iter']==int(eta*row['num_patterns'])

def get_spaced_indices(n, num_ticks=4):
    return np.linspace(0, n - 1, num_ticks, dtype=int)

plt.rcParams.update({'font.size': 20})
#%%
# Read the CSV file
# Fig_load_SR_average_new_inh_plas_many_betta_larger_networks
myDir = "../../data/all_data_splited/sleep_simulations/Fig_load_SR_average_new_inh_plas_many_betta_larger_networks"
data = pd.read_csv(myDir+'/all_simulation_data.csv')
# data = data[data['delta'] == 0.1]
#%%
# Calculate the ratio of successfully queried patterns
data['success_ratio'] = data['nb_fnd_pat'] / (data['num_patterns'])
data['num_patterns'] = data['num_patterns'].astype(int)
data['error_ratio'] = 1
data['is_error_before_all_fnd'] = False
#%%
sim_IDs=np.sort(data['sim_ID'].unique())
for sim_id in sim_IDs:
    error_ratio = 1
    is_error = False
    mask = data['sim_ID']==sim_id
    sim_data = data[mask]
    sim_data_all_patterns_fnd = sim_data[sim_data['nb_fnd_pat']==sim_data['num_patterns']]
    first_iter_all_fnd= 0
    if len(sim_data_all_patterns_fnd)==0:
        is_error = True
    else:
        first_iter_all_fnd_data=sim_data_all_patterns_fnd[sim_data_all_patterns_fnd['query_iter']==sim_data_all_patterns_fnd["query_iter"].min()]
        first_iter_all_fnd = first_iter_all_fnd_data["query_iter"]
        error_ratio=list(first_iter_all_fnd_data['nb_spurious']/first_iter_all_fnd)[0]
        if error_ratio>0:
            is_error=True
    data.loc[mask, 'error_ratio_all_fnd'] = error_ratio
    data.loc[mask, 'is_error_before_all_fnd'] = is_error
    data.loc[mask, 'first_iter_all_fnd'] = first_iter_all_fnd
    data.loc[mask, 'first_iter_all_fnd_no_error'] = first_iter_all_fnd
data["first_iter_all_fnd_no_error_normalized"] = data['first_iter_all_fnd_no_error']/data["num_patterns"]
#%%
all_num_patterns = np.sort(data['num_patterns'].unique())
all_net_sizes = np.sort(data['network_size'].unique())
#%%
for n in all_num_patterns:
    for s in all_net_sizes:
        is_error = data.loc[(data['network_size'] == s) & (data['num_patterns'] == n),"is_error_before_all_fnd"]
        any_error = np.any(list(is_error))
        if any_error:
            data.loc[(data['network_size'] == s) & (data['num_patterns'] == n),'first_iter_all_fnd_no_error']=None
            data.loc[(data['network_size'] == s) & (data['num_patterns'] == n),'first_iter_all_fnd_no_error_normalized']=None

#%%
x_tick_indices = get_spaced_indices(len(all_net_sizes),4)
y_tick_indices = get_spaced_indices(len(all_num_patterns),10)
#%%
pivot_table = data.pivot_table(values='is_error_before_all_fnd', index='num_patterns', columns='network_size')
plt.imshow(pivot_table)
plt.xticks(x_tick_indices,all_net_sizes[x_tick_indices])
plt.yticks(y_tick_indices,all_num_patterns[y_tick_indices])
#%%
pivot_table = data.pivot_table(values='first_iter_all_fnd', index='num_patterns', columns='network_size')
# cmap = plt.cm.viridis.copy()
# cmap.set_bad('gray')
plt.imshow(pivot_table)
plt.xticks(x_tick_indices,all_net_sizes[x_tick_indices])
plt.yticks(y_tick_indices,all_num_patterns[y_tick_indices])
plt.colorbar()
#%%
pivot_table = data.pivot_table(values='first_iter_all_fnd_no_error', index='num_patterns', columns='network_size')
# cmap = plt.cm.viridis.copy()
# cmap.set_bad('gray')
plt.imshow(pivot_table)
plt.xticks(x_tick_indices,all_net_sizes[x_tick_indices])
plt.yticks(y_tick_indices,all_num_patterns[y_tick_indices])
plt.colorbar()
#%%
pivot_table = data.pivot_table(values='first_iter_all_fnd_no_error_normalized', index='num_patterns', columns='network_size')
# cmap = plt.cm.viridis.copy()
# cmap.set_bad('gray')
plt.imshow(pivot_table)
plt.xticks(x_tick_indices,all_net_sizes[x_tick_indices])
plt.yticks(y_tick_indices,all_num_patterns[y_tick_indices])
plt.colorbar()
#%%
number_plot=6
ratio_taken = 0.99
nb_iter_mult = data['nb_iter_mult'][0]
eta_list = np.linspace(0.01,ratio_taken*nb_iter_mult,number_plot) # We have been doing 3 times more iterations than number of patterns
all_iter_ordered = np.sort(data['query_iter'].unique())
all_iter_ordered= equally_spaced_from_array(all_iter_ordered,number_plot, ratio_taken)
#%%
size = 150
new_data = data[data["network_size"] == size]
nb_pattern=9
new_data = new_data[new_data["num_patterns"]==nb_pattern]
repetition = 3
new_data = new_data[new_data["repetitions"]==repetition]
nb_query = nb_pattern*list(data["nb_iter_mult"])[0]
nb_spurious_list = []
for it in range(nb_query):
    nb_spurious = list(new_data.loc[new_data["query_iter"]==it,"nb_spurious"])[0]
    nb_spurious_list.append(nb_spurious)
x_marker = list(new_data["first_iter_all_fnd_no_error"])[0]
plt.axvline(x=x_marker, color='r', linestyle=':', label=f'x={x_marker}')
plt.plot(nb_spurious_list)

nb_query = nb_pattern*list(data["nb_iter_mult"])[0]
nb_spurious_list = []
for it in range(nb_query):
    nb_spurious = list(new_data.loc[new_data["query_iter"]==it,"success_ratio"])[0]
    nb_spurious_list.append(nb_spurious)
x_marker = list(new_data["first_iter_all_fnd_no_error"])[0]
plt.axvline(x=x_marker, color='r', linestyle=':', label=f'x={x_marker}')
plt.plot(nb_spurious_list)
#%%
nb_pattern=6
new_data = data[data["network_size"] == size]
#%%
# #%%
# size = 4
# # fig, ax = plt.subplots(1, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [6, 1]}, sharex=True)
# fig, axes = plt.subplots(2, int(number_plot/2), figsize=(size*(number_plot/2),size*2), sharey=True,sharex=True)
# # fig, ax = plt.subplots(1, 1, figsize=(12, 12), sharex=True)
# for i,eta in enumerate(eta_list):
#     # Create a pivot table for the phase diagram
#     data_iter = data.loc[data.apply(lambda row: relative_iter(row,eta),axis=1)]
#     # print(data_iter['query_iter'])
#     pivot_table = data_iter.pivot_table(values='success_ratio', index='num_patterns', columns='network_size')
#     # pivot_table = pivot_table.sort_index(ascending=False)  # Sort index in descending order
#     # Create the figure and axes
#     # Plot the phase diagram
#     y_axis_i = int(i//(number_plot/2))
#     x_axis_i = int(i-(y_axis_i*(number_plot/2)))
#     im = axes[y_axis_i][x_axis_i].imshow(pivot_table, aspect='auto', origin='upper', cmap='viridis',vmin=0,vmax=1)
#     axes[y_axis_i][x_axis_i].set_title(f'η = {eta:.2f}',fontsize=14)
#     axes[y_axis_i][x_axis_i].set_xticks(x_tick_indices)
#     axes[y_axis_i][x_axis_i].set_yticks(y_tick_indices)
#     axes[y_axis_i][x_axis_i].set_xticklabels(all_net_sizes[x_tick_indices])
#     axes[y_axis_i][x_axis_i].set_yticklabels(all_num_patterns[y_tick_indices])

# cbar_ax = fig.add_axes([0.93, 0.15, 0.009, 0.7])  # [left, bottom, width, height]
# cbar = fig.colorbar(im,cax=cbar_ax)
# cbar.set_label('Success Ratio',labelpad = 14)
# # plt.tight_layout(rect=[0,0,1,1])
# fig.text(0.5, 0.035, 'Network size', ha='center', va='center')
# fig.text(0.05, 0.45, 'number of stored pattern', ha='left', va='center',rotation=90)
# plt.show()
#%%
# size = 4
# # fig, ax = plt.subplots(1, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [6, 1]}, sharex=True)
# fig, axes = plt.subplots(2, int(number_plot/2), figsize=(size*(number_plot/2),size*2), sharey=True,sharex=True)
# # fig, ax = plt.subplots(1, 1, figsize=(12, 12), sharex=True)
# for i,iter in enumerate(all_iter_ordered):
#     # Create a pivot table for the phase diagram
#     data_iter = data[data['query_iter']==iter]
#     # print(data_iter['query_iter'])
#     pivot_table = data_iter.pivot_table(values='success_ratio', index='num_patterns', columns='network_size')
#     # pivot_table = pivot_table.sort_index(ascending=False)  # Sort index in descending order
#     # Create the figure and axes
#     # Plot the phase diagram
#     y_axis_i = int(i//(number_plot/2))
#     x_axis_i = int(i-(y_axis_i*(number_plot/2)))
#     im = axes[y_axis_i][x_axis_i].imshow(pivot_table, aspect='auto', origin='upper', cmap='viridis',vmin=0,vmax=1)
#     axes[y_axis_i][x_axis_i].set_title(f'iter = {iter}',fontsize=14)
#     axes[y_axis_i][x_axis_i].set_xticks(x_tick_indices)
#     axes[y_axis_i][x_axis_i].set_yticks(y_tick_indices)
#     axes[y_axis_i][x_axis_i].set_xticklabels(all_net_sizes[x_tick_indices])
#     axes[y_axis_i][x_axis_i].set_yticklabels(all_num_patterns[y_tick_indices])
# cbar_ax = fig.add_axes([0.93, 0.15, 0.009, 0.7])  # [left, bottom, width, height]
# cbar = fig.colorbar(im,cax=cbar_ax)
# cbar.set_label('Success Ratio',labelpad = 14)
# # plt.tight_layout(rect=[0,0,1,1])
# fig.text(0.5, 0.035, 'Network size', ha='center', va='center')
# fig.text(0.05, 0.45, 'number of stored pattern', ha='left', va='center',rotation=90)
# plt.show()

#%%
# size = 4
# # fig, ax = plt.subplots(1, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [6, 1]}, sharex=True)
# fig, axes = plt.subplots(2, int(number_plot/2), figsize=(size*(number_plot/2),size*2), sharey=True, sharex=True)
# # fig, ax = plt.subplots(1, 1, figsize=(12, 12), sharex=True)
# for i,eta in enumerate(eta_list):
#     # Create a pivot table for the phase diagram
#     data_iter = data.loc[data.apply(lambda row: relative_iter(row,eta),axis=1)]
#     # print(data_iter['query_iter'])
#     pivot_table = data_iter.pivot_table(values='relative_spurious', index='num_patterns', columns='network_size')
#     # pivot_table = pivot_table.sort_index(ascending=False)  # Sort index in descending order
#     # Create the figure and axes
#     # Plot the phase diagram
#     y_axis_i = int(i//(number_plot/2))
#     x_axis_i = int(i-(y_axis_i*(number_plot/2)))
#     im = axes[y_axis_i][x_axis_i].imshow(pivot_table, aspect='auto', origin='upper', cmap='Reds',vmin=0,vmax=0.5)
#     axes[y_axis_i][x_axis_i].set_title(f'η = {eta:.2f}',fontsize=14)
#     axes[y_axis_i][x_axis_i].set_xticks(x_tick_indices)
#     axes[y_axis_i][x_axis_i].set_yticks(y_tick_indices)
#     axes[y_axis_i][x_axis_i].set_xticklabels(all_net_sizes[x_tick_indices])
#     axes[y_axis_i][x_axis_i].set_yticklabels(all_num_patterns[y_tick_indices])
# cbar_ax = fig.add_axes([0.93, 0.15, 0.009, 0.7])  # [left, bottom, width, height]
# cbar = fig.colorbar(im,cax=cbar_ax)
# cbar.set_label('Ratio Error',labelpad = 14)
# # plt.tight_layout(rect=[0,0,1,1])
# fig.text(0.5, 0.035, 'Network size', ha='center', va='center')
# fig.text(0.05, 0.45, 'number of stored pattern', ha='left', va='center',rotation=90)
# plt.show()
# #%%
# data_no_spurious = data[data['relative_spurious_capped']==0]
# data_full_recovery_no_spurious = data_no_spurious[data_no_spurious['success_ratio']==1]
# #%%
# data_full_recovery_no_spurious.groupby("network_size").agg({'num_patterns':['max']})