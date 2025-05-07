#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

# Update the styling
sns.set_style("darkgrid")
sns.set_context("paper", font_scale=1.5)
plt.rcParams.update({'font.size': 15})

def equally_spaced_from_array(arr, n, ratio_taken):
    return arr[np.linspace(0, (len(arr)*ratio_taken)-1, n, dtype=int)]

def relative_iter(row, eta):
    return row['query_iter'] == int(eta * row['num_patterns'])

def get_spaced_indices(n, num_ticks=4):
    return np.linspace(0, n - 1, num_ticks, dtype=int)

def parse_data_file(file_path):
    """Parse a data file with key=value format."""
    data = {}
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if '=' in line:
                key, value = line.split('=')
                try:
                    data[key] = float(value)
                except ValueError:
                    data[key] = value
    return data

#%%
# Directory with all simulation data
myDir = "../../data/all_data_splited/trained_networks_fast/Fig_load_SR_average_new_inh_plas_big_simulations_many_correlations_new_convergence_nb_iter"

# Collect data from all simulation folders
all_data = []
for sim_folder in sorted(os.listdir(myDir)):
    # Check if this is a simulation folder
    if not sim_folder.startswith('sim_nb_'):
        continue
    
    sim_path = os.path.join(myDir, sim_folder)
    results_path = os.path.join(sim_path, 'results.data')
    params_path = os.path.join(sim_path, 'parameters.data')
    
    # Skip if files don't exist
    if not os.path.exists(results_path) or not os.path.exists(params_path):
        continue
    
    # Parse data files
    results_data = parse_data_file(results_path)
    params_data = parse_data_file(params_path)
    
    # Combine data
    sim_data = {**params_data, **results_data}
    sim_data['sim_id'] = int(re.match(r'sim_nb_(\d+)', sim_folder).group(1))
    
    all_data.append(sim_data)

# Convert to DataFrame
df = pd.DataFrame(all_data)

# Extract unique values for plotting
all_net_sizes = np.array(sorted(df['network_size'].unique()))
all_num_patterns = np.array(sorted(df['num_patterns'].unique()))
noise_levels = np.array(sorted(df['noise_level'].unique()))
noise_levels = noise_levels[[0,3,-1]]
#%%
# Create a single row of subplots for each noise level
# Only showing the log visualization, similar to the example image
fig, axes = plt.subplots(1, len(noise_levels), figsize=(4 * len(noise_levels), 4))

# If there's only one noise level, make sure axes is still a 1D array
if len(noise_levels) == 1:
    axes = np.array([axes])

# Define specific x and y ticks to match the example
x_tick_indices = get_spaced_indices(len(all_net_sizes), 4)
y_tick_indices = get_spaced_indices(len(all_num_patterns), 10)

for i, noise_level in enumerate(noise_levels):
    # Filter data for this noise level
    noise_df = df[df['noise_level'] == noise_level]
    
    # Create a pivot table for nb_iter
    nb_iter_pivot = pd.pivot_table(
        noise_df, 
        values='nb_iter', 
        index='num_patterns', 
        columns='network_size',
        aggfunc=np.mean
    )
    
    # Log scale of nb_iter
    log_iter_pivot = np.log10(nb_iter_pivot)
    
    # Create the imshow with hot colormap
    im = axes[i].imshow(log_iter_pivot, aspect='auto', cmap='hot', 
                       vmin=2.0, vmax=5.0)  # Set consistent color scale
    
    # Set title with precise formatting
    axes[i].set_title(f'ρ = {noise_level:.2f}', 
                      fontsize=14, pad=10)
    
    # Set y-axis label only on first subplot to avoid redundancy
    if i == 0:  # First subplot
        axes[i].set_ylabel('Number of patterns', fontsize=14)
    
    
    # Create integer labels for x-axis (network sizes)
    x_tick_labels = [str(int(all_net_sizes[idx])) for idx in x_tick_indices]
    axes[i].set_xticks(x_tick_indices)
    axes[i].set_xticklabels(x_tick_labels)
    
    # Create integer labels for y-axis (pattern numbers)
    y_tick_labels = [str(int(all_num_patterns[idx])) for idx in y_tick_indices]
    axes[i].set_yticks(y_tick_indices)
    axes[i].set_yticklabels(y_tick_labels)
    
    # Remove grid lines
    axes[i].grid(False)
    axes[i].invert_yaxis()

# Add colorbar with specific styling
# cbar = plt.colorbar(im, ax=axes)
# cbar.set_label('Log10(Iterations)', fontsize=12)
# cbar.ax.tick_params(labelsize=10)
fig.text(0.55, 0.0, 'Network size', ha='center', va='center')
cbar_ax = fig.add_axes([1, 0.13, 0.02, 0.8])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label('Log10(nb Iter)', fontsize=12)


plt.tight_layout()
plt.savefig('nb_iter_visualization_fixed.png', dpi=300, bbox_inches='tight')
plt.show()
#%%