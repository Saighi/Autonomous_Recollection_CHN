#%%
import utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
sns.set(font_scale=1.5)
# Custom correlation function (dot product of normalized vectors)
def vector_correlation(a, b):
    """
    Calculate the correlation between two vectors as the dot product of normalized vectors.
    This is equivalent to cosine similarity.
    """
    # Normalize vectors
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b)
    # Compute dot product
    return np.dot(a_norm, b_norm)
#%%
# Read the CSV file
myDir = '../../data/all_data_splited/sleep_simulations/Fig_typical_recovery_nb_iter_biased_diagonal_inh'
sim_dir_name = 'sim_nb_'
data = pd.read_csv(myDir+'/all_simulation_data.csv')
#%%
# Calculate the ratio of successfully queried patterns
data['success_ratio'] = data['nb_fnd_pat'] / (data['num_patterns'])
data['num_patterns'] = data['num_patterns'].astype(int)
data['relative_spurious'] = data["nb_spurious"]/data['num_patterns']
max_spurious = 0.5
data['relative_spurious_capped'] = np.clip(data['relative_spurious'], 0, max_spurious)
nb_sim = max(data['sim_ID'])
#%%
set(data['num_patterns'])
#%%
nb_iter = 6
nb_iter_offset= 0
#%%
set(data['network_size'])
#%%
set(data['num_patterns'])
#%%
# Cibled simulation
num_pattern = 4
network_size = 30
sub_data = data[(data['network_size'] == network_size) & (data['num_patterns'] == num_pattern)].sort_values(by='query_iter')
#%%
sim_ID = list(sub_data['sim_ID'])[0]
folder_sim = myDir + "/" + sim_dir_name + "" + str(sim_ID)
patterns = np.loadtxt(folder_sim + "/patterns.data")
parameters = utils.parse_config_file(folder_sim + "/parameters.data")
nb_iter_biased = data[data['sim_ID']==sim_ID]['nb_iter_biased']
#%%
# Initialize an array to store all correlation coefficients
all_correlations = []
iteration_lengths = []
for j in range(nb_iter_offset,nb_iter):
    print(nb_iter_biased)
    print("nb query iter = "+str(j))
    traj_file = folder_sim + "/results_" + str(j) + ".data"
    traj = np.loadtxt(traj_file)
    # Calculate vector correlation for each pattern at each time step
    correlations = []
    for t in range(traj.shape[0]):
        pattern_correlations = []
        for p in range(patterns.shape[0]):
            # Replace pearsonr with our vector_correlation function
            corr = vector_correlation(traj[t], patterns[p])
            pattern_correlations.append(corr)
        correlations.append(pattern_correlations)
    # Append correlations for this iteration to the all_correlations list
    all_correlations.extend(correlations)
    iteration_lengths.append(len(correlations))
#%%
# Convert to numpy array for easier manipulation
all_correlations = np.array(all_correlations)
#%%
nb_iter_free_phases = data[data['sim_ID']==sim_ID]['nb_iter_free'].values[nb_iter_offset:]
#%%
# Plot the results
plt.rcParams.update({'font.size': 15})
plt.figure(figsize=(12, 6))

# Add yellow rectangles between orange and red lines
cumulative_length = 0
for i, length in enumerate(iteration_lengths[:-1]):  # We don't need after the last iteration

    cumulative_length += length

# Reset cumulative_length for the lines
cumulative_length = 0

# Plot the correlation lines
for p in range(patterns.shape[0]):
    plt.plot(all_correlations[:, p], label=f'Pattern {p+1}', linewidth=3)

# Add the vertical lines
cumulative_length = 0
for i, length in enumerate(iteration_lengths[:-1]):  # We don't need a line after the last iteration
    cumulative_length += length
    # Start position of rectangle is at the orange line
    rect_start = cumulative_length - nb_iter_free_phases[i]
    # End position of rectangle is at the red line
    rect_end = cumulative_length
    # Add yellow rectangle with low alpha (transparency)
    plt.axvspan(rect_start, rect_end, color='yellow', alpha=0.2)
    plt.axvline(x=cumulative_length-nb_iter_free_phases[i], color='orange', linestyle='--', linewidth=2.5)
    plt.axvline(x=cumulative_length, color='red', linestyle='--', alpha=1, linewidth=2.5)

plt.xlabel('t', fontsize=20)
plt.ylabel(r'Corr $<\mathbf{u}^{\mu}\mathbf{u}(t)>$')
# plt.title(f'Correlation between trajectory and patterns (first {nb_iter} iterations)')
# plt.legend()
plt.show()

#%%
# Print the shape of the resulting array
print(f"Shape of all_correlations: {all_correlations.shape}")
print(f"Number of iterations processed: {nb_iter}")

#%%
# Cibled simulation
num_pattern = 5
network_size = 8

sub_data = data[(data['network_size'] == network_size) & (data['num_patterns'] == num_pattern)].sort_values(by='query_iter')

#%%
set(data['network_size'])
#%%
sim_ID = list(sub_data['sim_ID'])[0]
folder_sim = myDir + "/" + sim_dir_name + "" + str(sim_ID)
patterns = np.loadtxt(folder_sim + "/patterns.data")
parameters = utils.parse_config_file(folder_sim + "/parameters.data")

#%%
# Initialize an array to store all correlation coefficients
all_correlations = []
iteration_lengths = []
for j in range(0,nb_iter):
    print("nb query iter = "+str(j))
    traj_file = folder_sim + "/results_" + str(j) + ".data"
    traj = np.loadtxt(traj_file)
    # Calculate vector correlation for each pattern at each time step
    correlations = []
    for t in range(traj.shape[0]):
        pattern_correlations = []
        for p in range(patterns.shape[0]):
            # Replace pearsonr with our vector_correlation function
            corr = vector_correlation(traj[t], patterns[p])
            pattern_correlations.append(corr)
        correlations.append(pattern_correlations)
 
    # Append correlations for this iteration to the all_correlations list
    all_correlations.extend(correlations)
    iteration_lengths.append(len(correlations))

#%%
# Convert to numpy array for easier manipulation
all_correlations = np.array(all_correlations)
#%%
nb_iter_free_phases = data[data['sim_ID']==sim_ID]['nb_iter_free'].values
#%%
# Plot the results
plt.rcParams.update({'font.size': 15})
plt.figure(figsize=(12, 6))
for p in range(patterns.shape[0]):
    plt.plot(all_correlations[:, p], label=f'Pattern {p+1}',linewidth=3)

plt.xlabel('t')
plt.ylabel(r'Corr $<u^{\mu},u(t)>$')
# plt.title(f'Correlation between trajectory and patterns (first {nb_iter} iterations)')
# plt.legend()
# Add dotted red lines to separate iterations
cumulative_length = 0
for length in iteration_lengths[:-1]:  # We don't need a line after the last iteration
    cumulative_length += length
    plt.axvline(x=cumulative_length, color='red', linestyle=':', alpha=0.7)
plt.show()

#%%
# Print the shape of the resulting array
print(f"Shape of all_correlations: {all_correlations.shape}")
print(f"Number of iterations processed: {nb_iter}")

#%%
# Cibled simulation
num_pattern = 40
network_size = 200

sub_data = data[(data['network_size'] == network_size) & (data['num_patterns'] == num_pattern)].sort_values(by='query_iter')

#%%
set(data['network_size'])
#%%
sim_ID = list(sub_data['sim_ID'])[0]
folder_sim = myDir + "/" + sim_dir_name + "" + str(sim_ID)
patterns = np.loadtxt(folder_sim + "/patterns.data")
parameters = utils.parse_config_file(folder_sim + "/parameters.data")

#%%
# Initialize an array to store all correlation coefficients
all_correlations = []
iteration_lengths = []
for j in range(0,nb_iter):
    print("nb query iter = "+str(j))
    traj_file = folder_sim + "/results_" + str(j) + ".data"
    traj = np.loadtxt(traj_file)
    # Calculate vector correlation for each pattern at each time step
    correlations = []
    for t in range(traj.shape[0]):
        pattern_correlations = []
        for p in range(patterns.shape[0]):
            # Replace pearsonr with our vector_correlation function
            corr = vector_correlation(traj[t], patterns[p])
            pattern_correlations.append(corr)
        correlations.append(pattern_correlations)
 
    # Append correlations for this iteration to the all_correlations list
    all_correlations.extend(correlations)
    iteration_lengths.append(len(correlations))

#%%
# Convert to numpy array for easier manipulation
all_correlations = np.array(all_correlations)

#%%
# Plot the results
plt.rcParams.update({'font.size': 15})
plt.figure(figsize=(12, 6))
for p in range(patterns.shape[0]):
    plt.plot(all_correlations[:, p], label=f'Pattern {p+1}',linewidth=3)

plt.xlabel('t')
plt.ylabel(r'Corr $<\mathbf{u}^{\mu},\mathbf{u}(t)>$')
# plt.title(f'Correlation between trajectory and patterns (first {nb_iter} iterations)')
# plt.legend()
# Add dotted red lines to separate iterations
cumulative_length = 0
for length in iteration_lengths[:-1]:  # We don't need a line after the last iteration
    cumulative_length += length
    plt.axvline(x=cumulative_length, color='red', linestyle=':', alpha=0.7)
plt.show()

#%%
# Print the shape of the resulting array
print(f"Shape of all_correlations: {all_correlations.shape}")
print(f"Number of iterations processed: {nb_iter}")
# %%