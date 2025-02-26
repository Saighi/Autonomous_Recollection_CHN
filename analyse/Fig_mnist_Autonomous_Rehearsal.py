#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import regex as re
import utils 
import matplotlib.animation as animation
import seaborn as sns

sns.set_style("white")
sns.set_context("paper", font_scale=3)
#%%

size_picture = (28,28)
network_size = size_picture[0]*size_picture[1]
myDir = '/home/saighi/Desktop/data/all_data_splited/sleep_simulations/Fig_mnist_Autonomous_Rehearsal'
#%%
data_trajs_depressed = utils.load_simulation_trajectories(myDir,'depressed_traj_')
data_trajs_not_depressed = utils.load_simulation_trajectories(myDir,'free_traj_')
data_inhib_mats = utils.load_simulation_trajectories(myDir,'inhib_matrix_')
#%%
data_trajs_depressed
#%%
# Removed vmin/vmax calculation since we're setting fixed values
vmin = 0
vmax = 1
#%%
all_inhib = []
max_inhib_drive = 0
min_inhib_drive = 1000
for i in range(len(data_trajs_depressed)):
    inhib_drive = np.full(network_size,0.0)
    for k in range(network_size):
        for l in range(network_size):
            inhib_drive[k]+=data_inhib_mats[i][k][l]
    for k in inhib_drive:
        max_inhib_drive = max(max_inhib_drive,k)
        min_inhib_drive = min(min_inhib_drive,k)
        all_inhib.append(k)
#%%
times_depressed = [0,555]
nb_plot_depressed = len(times_depressed) # Has to be pair to deal with depressed and none depressed
nb_plot_not_depressed = 1 # Has to be pair to deal with depressed and none depressed
fig, axes = plt.subplots(5, nb_plot_depressed+nb_plot_not_depressed+1, figsize=(12, 25), sharey=True)
for i in range(len(data_trajs_depressed)):
    inhib_drive = np.full(network_size,0.0)
    for k in range(network_size):
        for l in range(network_size):
            inhib_drive[k]+=data_inhib_mats[i][k][l]
    inhib_drive = inhib_drive.reshape(size_picture[0],size_picture[1])

    activity_data_depressed = data_trajs_depressed[i]
    activity_data_not_depressed = data_trajs_not_depressed[i]
    times_depressed = np.linspace(len(activity_data_depressed)/len(axes[i]), len(activity_data_depressed)-1, nb_plot_depressed)
    times_not_depressed = np.linspace(len(activity_data_not_depressed)/len(axes[i]), len(activity_data_not_depressed), nb_plot_not_depressed)

    for j in range(nb_plot_depressed):
        ax = axes[i][j+1]
        im = ax.imshow(activity_data_depressed[int(times_depressed[j])].reshape((size_picture[0], size_picture[1])), 
                      vmin=vmin, vmax=vmax, cmap='viridis')
        ax.set_xticks([])  # Remove x ticks
        ax.set_yticks([])  # Remove y ticks
        if i ==0:
            if j ==0:
                ax.set_title("t="+str(int(times_depressed[j])))
            else :
                ax.set_title("t="+str(int(times_depressed[j])+1))

    for j in range(nb_plot_not_depressed):
        ax = axes[i][j+nb_plot_depressed+1]
        im = ax.imshow(activity_data_not_depressed[-1].reshape((size_picture[0], size_picture[1])), 
                      vmin=vmin, vmax=vmax, cmap='viridis')
        # im = ax.imshow(activity_data_not_depressed[int(times_not_depressed[j])].reshape((size_picture[0], size_picture[1])), 
        #               vmin=vmin, vmax=vmax, cmap='viridis')
        ax.set_xticks([])  # Remove x ticks
        ax.set_yticks([])  # Remove y ticks
        if i ==0:
            if j ==0:
                ax.set_title("t="+str(int(times_depressed[j])))
            else :
                ax.set_title("t="+str(int(times_depressed[j])+1))
    
    ax = axes[i][0]
    im_inhib = ax.imshow(inhib_drive, cmap='Reds', vmin=min_inhib_drive, vmax=max_inhib_drive)
    ax.set_xticks([])  # Remove x ticks
    ax.set_yticks([])  # Remove y ticks

cbar = fig.colorbar(im, ax=axes, shrink=0.4, orientation='horizontal', location='top', pad=0.1)
cbar.set_label('Rate', labelpad=20)

cbar_inhib = fig.colorbar(im_inhib, ax=axes, shrink=0.4, orientation='horizontal', location='bottom', pad=0.05, format='%d')
cbar_inhib.set_label('Inhibitory drive', labelpad=20)
#%%