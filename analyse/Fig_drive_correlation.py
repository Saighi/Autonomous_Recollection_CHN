#%% Import libraries
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import pearsonr
import seaborn as sns

sns.set_style("darkgrid")  # or "whitegrid", "dark", "white", "ticks"
sns.set_context("paper", font_scale=2)  # or "paper", "talk", "poster"
#%% Define helper functions
def load_matrix(file_path):
    """Load a matrix from a space-separated data file."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    matrix = []
    for line in lines:
        row = [float(val) for val in line.strip().split()]
        matrix.append(row)
    
    return np.array(matrix)

def load_patterns(file_path):
    """Load patterns from a space-separated data file."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    patterns = []
    for line in lines:
        pattern = [bool(int(val)) for val in line.strip().split()]
        patterns.append(pattern)
    
    return np.array(patterns, dtype=float)

def compute_synaptic_drive(weight_matrix):
    """Compute the synaptic drive for each neuron (sum of incoming weights)."""
    return np.sum(weight_matrix, axis=0)

def compute_avg_correlation(synaptic_drive, patterns):
    """Compute average correlation between synaptic drive and patterns."""
    correlations = []
    
    for pattern in patterns:
        # Convert boolean pattern to numeric (0s and 1s)
        pattern_numeric = pattern.astype(float)
        
        # Calculate Pearson correlation
        correlation, _ = pearsonr(synaptic_drive, pattern_numeric)
        correlations.append(correlation)
    
    return np.mean(correlations)

def read_noise_level(param_file):
    """Read the noise_level parameter from a parameter file."""
    with open(param_file, 'r') as f:
        for line in f:
            if line.startswith('noise_level='):
                return float(line.strip().split('=')[1])
    return None

#%% Set up the analysis
# Base directory containing simulation results
base_dir = "/home/saighi/Desktop/data/all_data_splited/trained_networks_fast/Fig_drive_correlation"

# Lists to store results
sim_correlations = []
noise_levels = []
sim_indices = []

#%% Process each simulation
# Process each simulation directory
for i in range(5):  # Assuming sim_nb_0 through sim_nb_4
    sim_dir = os.path.join(base_dir, f"sim_nb_{i}")
    
    # Load weight matrix and patterns
    weights_path = os.path.join(sim_dir, "weights.data")
    patterns_path = os.path.join(sim_dir, "patterns.data")
    params_path = os.path.join(sim_dir, "parameters.data")
    
    # Read noise level from parameters file
    noise_level = read_noise_level(params_path)
    
    weight_matrix = load_matrix(weights_path)
    patterns = load_patterns(patterns_path)
    
    # Compute synaptic drive
    synaptic_drive = compute_synaptic_drive(weight_matrix)
    
    # Compute average correlation with patterns
    avg_correlation = compute_avg_correlation(synaptic_drive, patterns)
    
    # Store results
    sim_correlations.append(avg_correlation)
    noise_levels.append(noise_level)
    sim_indices.append(i)
    
    print(f"Sim {i} (Noise level = {noise_level:.2f}): Average correlation = {avg_correlation:.4f}")

# Sort results by noise level
sorted_indices = np.argsort(noise_levels)
sorted_noise_levels = [noise_levels[i] for i in sorted_indices]
sorted_correlations = [sim_correlations[i] for i in sorted_indices]

#%% Create visualization
# Create bar plot
plt.figure(figsize=(10, 6))
bar_labels = [str(1-n) for n in sorted_noise_levels]
bars = plt.bar(bar_labels, sorted_correlations)

# Customize the plot
plt.xlabel(r'$\rho$')
plt.ylabel(r'$\overline{r}$')
# plt.title('Correlation between Synaptic Drive and Stored Patterns')
plt.ylim(0, 1)  # Correlation ranges from -1 to 1
# plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add correlation values on top of bars
for i, corr in enumerate(sorted_correlations):
    plt.text(i, corr + 0.05 if corr >= 0 else corr - 0.1, 
             f'{corr:.3f}', 
             ha='center')

plt.tight_layout()
plt.savefig('drive_pattern_correlation.png')
plt.show()
# %%
