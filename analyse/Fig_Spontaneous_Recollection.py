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

size_picture = (20,16)
network_size = size_picture[0]*size_picture[1]
myDir = '/home/saighi/Desktop/data/all_data_splited/sleep_simulations/Fig_Spontaneous_Recollection_spurious'
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
nb_plot_depressed = len(times_depressed)
nb_plot_not_depressed = 1
wi_ratios = [1]+[0.1]+[1]*(nb_plot_depressed+nb_plot_not_depressed)

# Create figure without colorbars first
fig, axes = plt.subplots(3, nb_plot_depressed+nb_plot_not_depressed+2, figsize=(15, 10), sharey=True,
                         gridspec_kw={'width_ratios': wi_ratios})

# Your existing plotting code
for i in range(len(data_trajs_depressed)-2):
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
        ax = axes[i][j+2]
        im = ax.imshow(activity_data_depressed[int(times_depressed[j])].reshape((size_picture[0], size_picture[1])),
                      vmin=vmin, vmax=vmax, cmap='viridis')
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 0:
            if j == 0:
                ax.set_title("t="+str(int(times_depressed[j])), fontsize=25)
            else:
                ax.set_title("t="+str(int(times_depressed[j])+2), fontsize=25)
    
    for j in range(nb_plot_not_depressed):
        ax = axes[i][j+nb_plot_depressed+2]
        im = ax.imshow(activity_data_not_depressed[-1].reshape((size_picture[0], size_picture[1])),
                      vmin=vmin, vmax=vmax, cmap='viridis')
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 0:
            if j == 0:
                ax.set_title("t="+str(int(times_depressed[j])), fontsize=25)
            else:
                ax.set_title("t="+str(int(times_depressed[j])+2), fontsize=25)
    
    ax = axes[i][0]
    im_inhib = ax.imshow(inhib_drive, cmap='Reds', vmin=min_inhib_drive, vmax=max_inhib_drive)
    ax.set_xticks([]) 
    ax.set_yticks([]) 
    
    for ax in axes[:, 1]:
        ax.axis("off")

# Apply tight_layout to properly organize the subplots first
plt.tight_layout()

# Now, create separate figure-level axes for colorbars that won't disturb the main layout
# Add colorbar for Rate (viridis)
cbar_ax = fig.add_axes([0.92, 0.3, 0.02, 0.4])  # [left, bottom, width, height]
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label('Rate', labelpad=15, fontsize=30)

# Add colorbar for Inhibitory drive (Reds)
cbar_inhib_ax = fig.add_axes([0.04, 0.3, 0.02, 0.4])  # [left, bottom, width, height]
cbar_inhib = fig.colorbar(im_inhib, cax=cbar_inhib_ax, format='%d')
cbar_inhib.set_label(r'$\mathbf{I}^{inh}$', labelpad=15, fontsize=40)
cbar_inhib.ax.yaxis.set_ticks_position('left')  # Move ticks to the left
cbar_inhib.ax.yaxis.set_label_position('left')  # Move label to the left
# Adjust the layout to accommodate the colorbars
fig.subplots_adjust(left=0.08, right=0.9)
#%%