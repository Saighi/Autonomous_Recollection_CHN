#%%
import utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter,MaxNLocator

# Update the styling
# sns.set_style("darkgrid")  # or "whitegrid", "dark", "white", "ticks"
# sns.set_context("paper", font_scale=1.5)  # or "paper", "talk", "poster"
sns.set_theme(style="ticks")

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times']
plt.rcParams.update({
    'font.size': 20,
    'axes.labelsize': 20,
    'axes.titlesize': 20,
    'xtick.labelsize': 17,
    'ytick.labelsize': 17,
    'legend.fontsize': 20,
    'figure.titlesize': 20,
    'lines.linewidth': 3,
    'axes.linewidth': 2,
    'axes.grid': False,
    'font.weight' : 'bold'
})

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
#%%
# Read the CSV file
myDir = '../../data/all_data_splited/sleep_simulations/Fig_typical_recovery_nb_iter_biased' # size 60 network
# myDir = '../../data/all_data_splited/sleep_simulations/Fig_typical_recovery_nb_iter_biased_small_network' 
# myDir = '../../data/all_data_splited/sleep_simulations/Fig_typical_recovery_size_30_network' 

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
num_pattern = 5
network_size = 60
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
plt.figure(figsize=(9, 4))

# Add yellow rectangles between orange and red lines
cumulative_length = 0
for i, length in enumerate(iteration_lengths[:-1]):  # We don't need after the last iteration

    cumulative_length += length

# Reset cumulative_length for the lines
cumulative_length = 0

# Plot the correlation lines
for p in range(patterns.shape[0]):
    plt.plot(all_correlations[:, p], label=f'Pattern {p+1}', linewidth=2.5)

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
    plt.axvline(x=cumulative_length-nb_iter_free_phases[i], color='orange', linestyle='--', linewidth=2)
    plt.axvline(x=cumulative_length, color='red', linestyle='--', alpha=1, linewidth=2)

# plt.xlabel('t', fontsize=20)
ax = plt.gca()
ax.xaxis.set_major_formatter(FuncFormatter(utils.thousands_formatter))
plt.ylabel(r'Corr $<\mathbf{u}^{\mu}\mathbf{u}(t)>$',labelpad=12)
# plt.title(f'Correlation between trajectory and patterns (first {nb_iter} iterations)')
# plt.legend()
sns.despine(top=True, right=True)
plt.yticks([0.4,0.6,0.8,1])
plt.savefig("./plots/Fig_typicall_recovery_scenario_1.png",dpi=300,bbox_inches='tight')
plt.show()
#%%
# Read the CSV file
# myDir = '../../data/all_data_splited/sleep_simulations/Fig_typical_recovery_nb_iter_biased' # size 60 network
# myDir = '../../data/all_data_splited/sleep_simulations/Fig_typical_recovery_nb_iter_biased_small_network' 
myDir = '../../data/all_data_splited/sleep_simulations/Fig_typical_recovery_size_30_network' 

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
nb_iter = 40
nb_iter_offset= 33
#%%
set(data['network_size'])
#%%
set(data['num_patterns'])
#%%
# Cibled simulation
num_pattern = 6
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
plt.figure(figsize=(9, 4))

# Add yellow rectangles between orange and red lines
cumulative_length = 0
for i, length in enumerate(iteration_lengths[:-1]):  # We don't need after the last iteration

    cumulative_length += length

# Reset cumulative_length for the lines
cumulative_length = 0

# Plot the correlation lines
for p in range(patterns.shape[0]):
    plt.plot(all_correlations[:, p], label=f'Pattern {p+1}', linewidth=2.5)

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
    plt.axvline(x=cumulative_length-nb_iter_free_phases[i], color='orange', linestyle='--', linewidth=2)
    plt.axvline(x=cumulative_length, color='red', linestyle='--', alpha=1, linewidth=2)

# plt.xlabel('t', fontsize=20)
ax = plt.gca()
ax.xaxis.set_major_formatter(FuncFormatter(utils.thousands_formatter))
plt.ylabel(r'Corr $<\mathbf{u}^{\mu}\mathbf{u}(t)>$',labelpad=12)
sns.despine(top=True, right=True)
plt.yticks([0.4,0.6,0.8,1])
plt.savefig("./plots/Fig_typicall_recovery_scenario_2.png",dpi=300,bbox_inches='tight')
plt.show()


# %%
#%%
# Read the CSV file
myDir = '../../data/all_data_splited/sleep_simulations/Fig_typical_recovery_nb_iter_biased' # size 60 network
# myDir = '../../data/all_data_splited/sleep_simulations/Fig_typical_recovery_nb_iter_biased_small_network' 
# myDir = '../../data/all_data_splited/sleep_simulations/Fig_typical_recovery_size_30_network' 

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
num_pattern = 16
network_size = 60
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
plt.figure(figsize=(9, 4))

# Add yellow rectangles between orange and red lines
cumulative_length = 0
for i, length in enumerate(iteration_lengths[:-1]):  # We don't need after the last iteration

    cumulative_length += length

# Reset cumulative_length for the lines
cumulative_length = 0

# Plot the correlation lines
for p in range(patterns.shape[0]):
    plt.plot(all_correlations[:, p], label=f'Pattern {p+1}', linewidth=2.5)

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
    plt.axvline(x=cumulative_length-nb_iter_free_phases[i], color='orange', linestyle='--', linewidth=2)
    plt.axvline(x=cumulative_length, color='red', linestyle='--', alpha=1, linewidth=2)

plt.xlabel('t', fontsize=20)
ax = plt.gca()
ax.xaxis.set_major_formatter(FuncFormatter(utils.thousands_formatter))
plt.ylabel(r'Corr $<\mathbf{u}^{\mu}\mathbf{u}(t)>$',labelpad=12)
# plt.title(f'Correlation between trajectory and patterns (first {nb_iter} iterations)')
# plt.legend()
sns.despine(top=True, right=True)
plt.yticks([0.4,0.6,0.8,1])
plt.savefig("./plots/Fig_typicall_recovery_scenario_3.png",dpi=300,bbox_inches='tight')
plt.show()
#%%
# Read the CSV file
myDir = '../../data/all_data_splited/sleep_simulations/Fig_incremental_learning' # size 60 network
# myDir = '../../data/all_data_splited/sleep_simulations/Fig_typical_recovery_nb_iter_biased_small_network' 
# myDir = '../../data/all_data_splited/sleep_simulations/Fig_typical_recovery_size_30_network' 

sim_dir_name = 'sim_nb_'
data = pd.read_csv(myDir+'/all_simulation_data.csv')
#%%
#%%
# Cibled simulation
num_pattern = 2
network_size = 100
sub_data = data[(data['network_size'] == network_size) & (data['num_patterns'] == num_pattern)].sort_values(by='query_iter')
#%%
sim_ID = list(sub_data['sim_ID'])[0]
folder_sim = myDir + "/" + sim_dir_name + "" + str(sim_ID)
patterns = np.loadtxt(folder_sim + "/patterns.data")
parameters = utils.parse_config_file(folder_sim + "/parameters.data")
nb_iter_biased = data[data['sim_ID']==sim_ID]['nb_iter_biased']
#%%
nb_iter = 2
nb_iter_offset= 0
#%%
# Initialize an array to store all correlation coefficients
all_correlations = []
iteration_lengths = []
for j in range(nb_iter_offset,nb_iter):
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
nb_iter_free_phases = data[data['sim_ID']==sim_ID]['nb_iter_free'].values[nb_iter_offset:nb_iter]
# %%
# Plot the results
plt.figure(figsize=(6, 3))

# Add yellow rectangles between orange and red lines
cumulative_length = 0
for i, length in enumerate(iteration_lengths[:-1]):  # We don't need after the last iteration
    cumulative_length += length

# Reset cumulative_length for the lines
cumulative_length = 0

# Plot the correlation lines
for p in range(patterns.shape[0]):
    plt.plot(all_correlations[:, p], label=f'Pattern {p+1}', linewidth=2.5)

# Add the vertical lines
cumulative_length = 0
for i, length in enumerate(iteration_lengths[:-1]):  # We don't need a line after the last iteration
    cumulative_length += length
    # Start position of rectangle is at the orange line
    rect_start = cumulative_length - nb_iter_free_phases[i]
    # End position of rectangle is at the red line
    rect_end = cumulative_length
    # Add yellow rectangle with low alpha (transparency)
    # plt.axvspan(rect_start, rect_end, color='yellow', alpha=0.2)
    # plt.axvline(x=cumulative_length-nb_iter_free_phases[i], color='orange', linestyle='--', linewidth=2)
    plt.axvline(x=cumulative_length, color='red', linestyle='--', alpha=1, linewidth=2)

plt.xlabel('t', fontsize=20)
ax = plt.gca()
plt.xticks([0,1000,2000])
ax.xaxis.set_major_formatter(FuncFormatter(utils.thousands_formatter))
plt.ylabel(r'Corr $<\mathbf{u}^{\mu}\mathbf{u}(t)>$',labelpad=12)
# plt.title(f'Correlation between trajectory and patterns (first {nb_iter} iterations)')
# plt.legend()
sns.despine(top=True, right=True)
plt.yticks([0.4,0.6,0.8,1])
plt.savefig("./plots/Fig_incremental_learning_1.png",dpi=300,bbox_inches='tight')
plt.show()
#%%
# Cibled simulation
num_pattern = 5
network_size = 100
sub_data = data[(data['network_size'] == network_size) & (data['num_patterns'] == num_pattern)].sort_values(by='query_iter')
#%%
sim_ID = list(sub_data['sim_ID'])[0]
folder_sim = myDir + "/" + sim_dir_name + "" + str(sim_ID)
patterns = np.loadtxt(folder_sim + "/patterns.data")
parameters = utils.parse_config_file(folder_sim + "/parameters.data")
nb_iter_biased = data[data['sim_ID']==sim_ID]['nb_iter_biased']
#%%
nb_iter = 5
nb_iter_offset= 0
#%%
# Initialize an array to store all correlation coefficients
all_correlations = []
iteration_lengths = []
for j in range(nb_iter_offset,nb_iter):
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
nb_iter_free_phases = data[data['sim_ID']==sim_ID]['nb_iter_free'].values[nb_iter_offset:nb_iter]
# %%
# Plot the results
plt.figure(figsize=(6, 3))

# Add yellow rectangles between orange and red lines
cumulative_length = 0
for i, length in enumerate(iteration_lengths[:-1]):  # We don't need after the last iteration
    cumulative_length += length

# Reset cumulative_length for the lines
cumulative_length = 0

# Plot the correlation lines
for p in range(patterns.shape[0]):
    plt.plot(all_correlations[:, p], label=f'Pattern {p+1}', linewidth=2.5)

# Add the vertical lines
cumulative_length = 0
for i, length in enumerate(iteration_lengths[:-1]):  # We don't need a line after the last iteration
    cumulative_length += length
    # Start position of rectangle is at the orange line
    rect_start = cumulative_length - nb_iter_free_phases[i]
    # End position of rectangle is at the red line
    rect_end = cumulative_length
    # Add yellow rectangle with low alpha (transparency)
    # plt.axvspan(rect_start, rect_end, color='yellow', alpha=0.2)
    # plt.axvline(x=cumulative_length-nb_iter_free_phases[i], color='orange', linestyle='--', linewidth=2)
    plt.axvline(x=cumulative_length, color='red', linestyle='--', alpha=1, linewidth=2)

plt.xlabel('t', fontsize=20)
ax = plt.gca()
plt.xticks([0,1000,2000,3000,4000])
ax.xaxis.set_major_formatter(FuncFormatter(utils.thousands_formatter))
plt.ylabel(r'Corr $<\mathbf{u}^{\mu}\mathbf{u}(t)>$',labelpad=12)
# plt.title(f'Correlation between trajectory and patterns (first {nb_iter} iterations)')
# plt.legend()
sns.despine(top=True, right=True)
plt.yticks([0.4,0.6,0.8,1])
plt.savefig("./plots/Fig_incremental_learning_2.png",dpi=300,bbox_inches='tight')
plt.show()
# %%
# Cibled simulation
num_pattern = 6
network_size = 100
sub_data = data[(data['network_size'] == network_size) & (data['num_patterns'] == num_pattern)].sort_values(by='query_iter')
#%%
sim_ID = list(sub_data['sim_ID'])[0]
folder_sim = myDir + "/" + sim_dir_name + "" + str(sim_ID)
patterns = np.loadtxt(folder_sim + "/patterns.data")
parameters = utils.parse_config_file(folder_sim + "/parameters.data")
nb_iter_biased = data[data['sim_ID']==sim_ID]['nb_iter_biased']
#%%
nb_iter = 6
nb_iter_offset= 0
#%%
# Initialize an array to store all correlation coefficients
all_correlations = []
iteration_lengths = []
for j in range(nb_iter_offset,nb_iter):
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
nb_iter_free_phases = data[data['sim_ID']==sim_ID]['nb_iter_free'].values[nb_iter_offset:nb_iter]
# %%
# Plot the results
plt.figure(figsize=(6, 3))

# Add yellow rectangles between orange and red lines
cumulative_length = 0
for i, length in enumerate(iteration_lengths[:-1]):  # We don't need after the last iteration
    cumulative_length += length

# Reset cumulative_length for the lines
cumulative_length = 0

# Plot the correlation lines
for p in range(patterns.shape[0]):
    plt.plot(all_correlations[:, p], label=f'Pattern {p+1}', linewidth=2.5)

# Add the vertical lines
cumulative_length = 0
for i, length in enumerate(iteration_lengths[:-1]):  # We don't need a line after the last iteration
    cumulative_length += length
    # Start position of rectangle is at the orange line
    rect_start = cumulative_length - nb_iter_free_phases[i]
    # End position of rectangle is at the red line
    rect_end = cumulative_length
    # Add yellow rectangle with low alpha (transparency)
    # plt.axvspan(rect_start, rect_end, color='yellow', alpha=0.2)
    # plt.axvline(x=cumulative_length-nb_iter_free_phases[i], color='orange', linestyle='--', linewidth=2)
    plt.axvline(x=cumulative_length, color='red', linestyle='--', alpha=1, linewidth=2)

plt.xlabel('t', fontsize=20)
ax = plt.gca()
plt.xticks([0,1000,2000,3000,4000,5000])
ax.xaxis.set_major_formatter(FuncFormatter(utils.thousands_formatter))
plt.ylabel(r'Corr $<\mathbf{u}^{\mu}\mathbf{u}(t)>$',labelpad=12)
# plt.title(f'Correlation between trajectory and patterns (first {nb_iter} iterations)')
# plt.legend()
sns.despine(top=True, right=True)
plt.yticks([0.4,0.6,0.8,1])
plt.savefig("./plots/Fig_incremental_learning_3.png",dpi=300,bbox_inches='tight')
plt.show()
# %%
