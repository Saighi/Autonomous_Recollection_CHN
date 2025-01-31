#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots
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
myDir = "../../data/all_data_splited/sleep_simulations/Fig_recovery_speed_size_and_spurious"
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
#%%
data = data.groupby(["network_size",'query_iter']).mean().reset_index()
#%%
fig, axes = plt.subplots(1,2, figsize=(10,5))
for size in all_net_sizes:
    recovery_traj = np.array(data.loc[data['network_size']==size,['nb_fnd_pat','query_iter']]).T
    x_axis = recovery_traj[1]
    y_axis = recovery_traj[0]
    axes[0].plot(x_axis,y_axis)
#axes[0].legend(all_net_sizes,title="Network Sizes")
for size in all_net_sizes:
    recovery_traj = np.array(data.loc[data['network_size']==size,['nb_spurious','query_iter']]).T
    x_axis = recovery_traj[1]
    y_axis = recovery_traj[0]
    axes[1].plot(x_axis,y_axis)
plt.legend(all_net_sizes,title="Network Sizes")
# %%
