#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Modern seaborn styling
sns.set_style("darkgrid")
sns.set_context("notebook", font_scale=1.5)

#%%
# Read the CSV file
myDir = "../../data/all_data_splited/sleep_simulations/Fig_recovery_speed_size_and_spurious"
data = pd.read_csv(myDir+'/all_simulation_data.csv')

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
# Create figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# First subplot: Found patterns
for size in all_net_sizes:
    # Calculate mean and std for each query_iter
    stats = data[data['network_size'] == size].groupby('query_iter').agg({
        'nb_fnd_pat': ['mean', 'std']
    }).reset_index()
    
    x_axis = stats['query_iter']
    y_mean = stats['nb_fnd_pat']['mean']
    y_std = stats['nb_fnd_pat']['std']
    
    # Plot mean line
    line = axes[0].plot(x_axis, y_mean, 
                       label=f'Size: {size}',
                       linewidth=2)[0]
    color = line.get_color()
    
    # Add variance bands
    axes[0].fill_between(x_axis,
                        y_mean - y_std,
                        y_mean + y_std,
                        alpha=0.2,
                        color=color)

# Second subplot: Spurious patterns
for size in all_net_sizes:
    # Calculate mean and std for each query_iter
    stats = data[data['network_size'] == size].groupby('query_iter').agg({
        'nb_spurious': ['mean', 'std']
    }).reset_index()
    
    x_axis = stats['query_iter']
    y_mean = stats['nb_spurious']['mean']
    y_std = stats['nb_spurious']['std']
    
    # Plot mean line
    line = axes[1].plot(x_axis, y_mean,
                       label=f'{size}')[0]
    color = line.get_color()
    
    # Add variance bands
    axes[1].fill_between(x_axis,
                        y_mean - y_std,
                        y_mean + y_std,
                        alpha=0.2,
                        color=color)

# Customize first subplot
axes[0].set_xlabel('Query iteration')
axes[0].set_ylabel('Nb found patterns')
axes[0].grid(True, alpha=0.3)

# Customize second subplot
axes[1].set_xlabel('Query iteration')
axes[1].set_ylabel('Nb spurious patterns')
axes[1].grid(True, alpha=0.3)
axes[1].legend(title="Network size")

# Set y-axis to start at 0 for both plots
axes[0].set_ylim(bottom=0)
axes[1].set_ylim(bottom=0)

# Adjust layout to prevent overlapping
plt.tight_layout()