#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import regex as re
import utils 
import matplotlib.animation as animation
from scipy.stats import pearsonr
plt.rcParams.update({'font.size': 20})
#%%
nb_iter = 20
size_picture = (20,16)
network_size = size_picture[0]*size_picture[1]
sim_dir_name = 'sim_nb_0'
myDir = '..\\..\\data\\all_data_splited\\sleep_simulations\\Fig_Spontaneous_Recollection_rand_pat'
#%%
data_trajs_depressed = utils.load_simulation_trajectories(myDir,'results_depressed_')
data_trajs_not_depressed = utils.load_simulation_trajectories(myDir,'results_')
data_inhib_mats = utils.load_simulation_trajectories(myDir,'inhib_matrix_')
#%%
folder_sim = myDir + "\\" + sim_dir_name
patterns = np.loadtxt(folder_sim + "\\patterns.data")
parameters = utils.parse_config_file(folder_sim + "\\parameters.data")
#%%
vmin =1
vmax =0
for i in range(len(data_trajs_depressed[0])):
    for j in data_trajs_depressed[0][i]:
        min_j = min(j) 
        max_j = max(j)
        if vmin>min_j:
            vmin = min_j
        if vmax<max_j:
            vmax = max_j
for i in range(len(data_trajs_not_depressed[0])):
    for j in data_trajs_not_depressed[0][i]:
        min_j = min(j) 
        max_j = max(j)
        if vmin>min_j:
            vmin = min_j
        if vmax<max_j:
            vmax = max_j
#%%
all_inhib = []
for i in range(len(data_trajs_depressed[0])):
    inhib_drive = np.full(network_size,0.0)
    for k in range(network_size):
        for l in range(network_size):
            inhib_drive[k]+=data_inhib_mats[0][i][k][l]
    for k in inhib_drive:
        all_inhib.append(k)
#%%
nb_plot_depressed = 3 # Has to be pair to deal with depressed and none depressed
nb_plot_not_depressed = 3 # Has to be pair to deal with depressed and none depressed
fig, axes = plt.subplots(nb_iter, nb_plot_depressed+nb_plot_not_depressed+1, figsize=(25, 25), sharey=True)
for i in range(len(data_trajs_depressed[0])):
    inhib_drive = np.full(network_size,0.0)
    for k in range(network_size):
        for l in range(network_size):
            inhib_drive[k]+=data_inhib_mats[0][i][k][l]
    inhib_drive = (inhib_drive.reshape(size_picture[0],size_picture[1])-np.min(all_inhib))/(np.max(all_inhib)-np.min(all_inhib))
    activity_data_depressed = data_trajs_depressed[0][i]
    activity_data_not_depressed = data_trajs_not_depressed[0][i]
    times_depressed = np.linspace(len(activity_data_depressed)/len(axes[i]), len(activity_data_depressed)-1, nb_plot_depressed)
    times_not_depressed = np.linspace(len(activity_data_not_depressed)/len(axes[i]), len(activity_data_not_depressed)-1, nb_plot_not_depressed)

    for j in range(nb_plot_depressed):
        ax = axes[i][j+1]
        im = ax.imshow(activity_data_depressed[int(times_depressed[j])].reshape((size_picture[0], size_picture[1])), vmin=vmin,vmax=vmax)
        if i ==0:
            if j ==0:
                ax.set_title("t="+str(int(times_depressed[j])))
            else :
                ax.set_title("t="+str(int(times_depressed[j])+1))
            

    for j in range(nb_plot_not_depressed):
        ax = axes[i][j+nb_plot_depressed+1]
        im = ax.imshow(activity_data_not_depressed[int(times_not_depressed[j])].reshape((size_picture[0], size_picture[1])), vmin=vmin,vmax=vmax)
        if i ==0:
            if j ==0:
                ax.set_title("t="+str(int(times_depressed[j])))
            else :
                ax.set_title("t="+str(int(times_depressed[j])+1))
    
    ax = axes[i][0]
    im_inhib = ax.imshow(inhib_drive,cmap='Reds',vmin=0,vmax=1)

cbar = fig.colorbar(im, ax=axes,shrink=0.4,orientation='horizontal', location = 'top', pad = 0.025)
cbar.set_label('Activity Level',labelpad=20)
cbar_inhib = fig.colorbar(im_inhib, ax=axes,shrink=0.4,orientation='horizontal', location = 'bottom',pad=0.05,format='%.1f' )
cbar_inhib.set_label('Normalized Inhibition',labelpad=20)
#%%
# PEARSON COEFFICIENTS

#%%
# Initialize an array to store all correlation coefficients
all_correlations = []
iteration_lengths = []
for j in range(0,nb_iter):
    print("nb query iter = "+str(j))
    traj_file_1 = folder_sim + "\\results_" + str(j) + ".data"
    traj_1 = np.loadtxt(traj_file_1,ndmin=2)
    traj_file_2 = folder_sim + "\\results_depressed_" + str(j) + ".data"
    traj_2 = np.loadtxt(traj_file_2,ndmin=2)
    traj = np.append(traj_2,traj_1,axis=0)
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
    plt.plot(all_correlations[:, p], label=f'Pattern {p+1}')

plt.xlabel('Time steps (concatenated across iterations)')
plt.ylabel('Pearson correlation coefficient')
plt.title(f'Correlation between trajectory and patterns over time (first {nb_iter} iterations)')
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
