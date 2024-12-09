#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Update the styling
sns.set_style("darkgrid")  # or "whitegrid", "dark", "white", "ticks"
sns.set_context("paper", font_scale=2)  # or "paper", "talk", "poster"

def equally_spaced_from_array(arr, n, ratio_taken):
    return arr[np.linspace(0, (len(arr)*ratio_taken)-1, n, dtype=int)]

def relative_iter(row,eta):
    return row['query_iter']==int(eta*row['num_patterns'])

def get_spaced_indices(n, num_ticks=4):
    return np.linspace(0, n - 1, num_ticks, dtype=int)

# Read the CSV file
myDir = "../../data/all_data_splited/sleep_simulations/Fig_compare_CHN_DHN_2"
data = pd.read_csv(myDir+'/all_simulation_data.csv')

# Calculate the ratio of successfully queried patterns
data['success_ratio'] = data['nb_fnd_pat'] / (data['num_patterns'])
data['num_patterns'] = data['num_patterns'].astype(int)

all_num_patterns = np.sort(data['num_patterns'].unique())
all_net_sizes = np.sort(data['network_size'].unique())

x_tick_indices = get_spaced_indices(len(all_net_sizes),4)
y_tick_indices = get_spaced_indices(len(all_num_patterns),10)

plt.figure(figsize=(8, 6))

# Find global minimum of spurious patterns
min_spurious = float('inf')
for nb_pat in all_num_patterns:
    data_one_nb_pat = data[data["num_patterns"]==nb_pat]
    data_one_nb_pat = data_one_nb_pat[data_one_nb_pat['nb_fnd_pat']==data_one_nb_pat['num_patterns']]
    data_one_nb_pat = data_one_nb_pat.loc[data_one_nb_pat.groupby('sim_ID')['query_iter'].idxmin()]
    min_spurious = min(min_spurious, data_one_nb_pat['nb_spurious'].min())
# Add minimum line
plt.axhline(y=min_spurious, color='red', linestyle='--', linewidth=2)

for i, nb_pat in enumerate(all_num_patterns):
    data_one_nb_pat = data[data["num_patterns"]==nb_pat]
    data_one_nb_pat = data_one_nb_pat[data_one_nb_pat['nb_fnd_pat']==data_one_nb_pat['num_patterns']]
    data_one_nb_pat = data_one_nb_pat.loc[data_one_nb_pat.groupby('sim_ID')['query_iter'].idxmin()]
    
    # Calculate mean and standard deviation for each network size
    stats = data_one_nb_pat.groupby('network_size').agg({
        'nb_spurious': ['mean', 'std']
    }).reset_index()
    
    x_axis = stats['network_size']
    y_mean = stats['nb_spurious']['mean']
    y_std = stats['nb_spurious']['std']
    
    # Plot mean line
    line = plt.plot(x_axis, y_mean, label=f'{nb_pat}', 
                    linewidth=2,
                    marker='o',
                    markersize=4)[0]
    color = line.get_color()
    
    # Add variance bands (±1 standard deviation)
    plt.fill_between(x_axis, 
                    y_mean - y_std,
                    y_mean + y_std,
                    alpha=0.2,
                    color=color)
plt.ylim(-1 , 120)
plt.xlabel('Network size')
#plt.ylabel('Nb spurious patterns')
#plt.legend(title="Nb stored patterns")
plt.grid(True)
plt.tight_layout()
# %%
