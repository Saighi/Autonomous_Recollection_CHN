#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots
# plt.style.use('science')
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
myDir = "../../data/all_data_splited/sleep_simulations/Fig_load_SR_average_new_inh_plas_big_simulations_many_beta"
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
#%%
x_tick_indices = get_spaced_indices(len(all_net_sizes),4)
y_tick_indices = get_spaced_indices(len(all_num_patterns),10)
#%%
data_betas = dict()
fig, axes = plt.subplots(2,5, figsize=(12,8), sharey=True,sharex=True)
for i,beta in enumerate(all_beta):
    data_one_beta = data[data['beta']==beta]
    sim_IDs=np.sort(data['sim_ID'].unique())
    for sim_id in sim_IDs:
        is_error = False
        mask = data_one_beta['sim_ID']==sim_id
        sim_data = data_one_beta[mask]
        sim_data_all_patterns_fnd = sim_data[sim_data['nb_fnd_pat']==sim_data['num_patterns']]
        first_iter_all_fnd= 0
        if len(sim_data_all_patterns_fnd)==0:
            is_error = True
        first_iter_all_fnd_data=sim_data_all_patterns_fnd[sim_data_all_patterns_fnd['query_iter']==sim_data_all_patterns_fnd["query_iter"].min()]
        first_iter_all_fnd = first_iter_all_fnd_data["query_iter"]+1

        data_one_beta.loc[mask, 'is_error_before_all_fnd'] = is_error
        data_one_beta.loc[mask, 'first_iter_all_fnd'] = first_iter_all_fnd

    # Keep only the last iteration as representent of the simulation
    data_one_beta = data_one_beta.loc[data_one_beta.groupby("sim_ID")["query_iter"].idxmax()]
    for n in all_num_patterns:
        for s in all_net_sizes:
            is_error = data_one_beta.loc[(data_one_beta['network_size'] == s) & (data_one_beta['num_patterns'] == n),"is_error_before_all_fnd"]
            any_error = np.any(list(is_error))
            if any_error:
                data_one_beta.loc[(data_one_beta['network_size'] == s) & (data_one_beta['num_patterns'] == n),'first_iter_all_fnd']=None
                data_one_beta.loc[(data_one_beta['network_size'] == s) & (data_one_beta['num_patterns'] == n),'first_iter_all_fnd']=None
    data_one_beta["first_iter_all_fnd_normalized"] = data_one_beta['first_iter_all_fnd']/data_one_beta["num_patterns"]

    data_betas[beta]=data_one_beta # store the specific data for later

    # pivot_table = data_one_beta.pivot_table(values='is_error_before_all_fnd', index='num_patterns', columns='network_size')
    # im = axes[0][i].imshow(pivot_table*100,cmap="Reds")
    # axes[0][i].set_xticks(x_tick_indices,all_net_sizes[x_tick_indices])
    # axes[0][i].set_yticks(y_tick_indices,all_num_patterns[y_tick_indices])
    # plt.colorbar(im, ax=axes[0][i],shrink=0.7)
    # pivot_table = data_one_beta.pivot_table(values='first_iter_all_fnd_normalized', index='num_patterns', columns='network_size')
    # # cmap = plt.cm.viridis.copy()
    # # cmap.set_bad('gray')
    # im = axes[1][i].imshow(pivot_table)
    # axes[1][i].set_xticks(x_tick_indices,all_net_sizes[x_tick_indices])
    # axes[1][i].set_yticks(y_tick_indices,all_num_patterns[y_tick_indices])
    plt.colorbar(im, ax=axes[1][i],shrink=0.7)
#fig.text(0.5, 0.425, 'Network size', ha='center', va='center')
#fig.text(0.05, 0.49, 'Nb stored pattern', ha='left', va='center',rotation=90)
#%%
fig, axes = plt.subplots(2,5, figsize=(12,8), sharey=True,sharex=True)
for i,beta in enumerate(all_beta):
    data_one_beta = data_betas[beta]
    pivot_table = data_one_beta.pivot_table(values='is_error_before_all_fnd', index='num_patterns', columns='network_size')
    im = axes[0][i].imshow(pivot_table*100,cmap="Reds")
    axes[0][i].set_xticks(x_tick_indices,all_net_sizes[x_tick_indices])
    axes[0][i].set_yticks(y_tick_indices,all_num_patterns[y_tick_indices])
    plt.colorbar(im, ax=axes[0][i],shrink=0.7)
    pivot_table = data_one_beta.pivot_table(values='first_iter_all_fnd_normalized', index='num_patterns', columns='network_size')
    # cmap = plt.cm.viridis.copy()
    # cmap.set_bad('gray')
    im = axes[1][i].imshow(pivot_table)
    axes[1][i].set_xticks(x_tick_indices,all_net_sizes[x_tick_indices])
    axes[1][i].set_yticks(y_tick_indices,all_num_patterns[y_tick_indices])
    plt.colorbar(im, ax=axes[1][i],shrink=0.7)
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