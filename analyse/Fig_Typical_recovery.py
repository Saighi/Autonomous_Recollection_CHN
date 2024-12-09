#%%
import utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import seaborn as sns
sns.set_theme()
sns.set(font_scale=1.5)
#%%
# Read the CSV file
myDir = '../../data/all_data_splited/sleep_simulations/Fig_typical_recovery'
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
nb_iter = 11
#%%
# Cibled simulation
num_pattern = 5
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
    # Calculate Pearson correlation for each pattern at each time step
    correlations = []
    for t in range(traj.shape[0]):
        pattern_correlations = []
        for p in range(patterns.shape[0]):
            corr, _ = pearsonr(traj[t], patterns[p])
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
plt.ylabel('PCC')
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
num_pattern = 21
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
    # Calculate Pearson correlation for each pattern at each time step
    correlations = []
    for t in range(traj.shape[0]):
        pattern_correlations = []
        for p in range(patterns.shape[0]):
            corr, _ = pearsonr(traj[t], patterns[p])
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
plt.ylabel('PCC')
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
    # Calculate Pearson correlation for each pattern at each time step
    correlations = []
    for t in range(traj.shape[0]):
        pattern_correlations = []
        for p in range(patterns.shape[0]):
            corr, _ = pearsonr(traj[t], patterns[p])
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
plt.ylabel('PCC')
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
