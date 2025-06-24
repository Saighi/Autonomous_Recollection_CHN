#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import regex as re
import utils 
import matplotlib.animation as animation
import seaborn as sns

sns.set_theme(style="ticks")

# plt.rcParams['text.usetex'] = True
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['Computer Modern Roman']
plt.rcParams.update({
    'text.usetex': True,
    'font.size': 17,
    'axes.labelsize': 18,
    'axes.titlesize': 16,
    'xtick.labelsize': 17,
    'ytick.labelsize': 17,
    'legend.fontsize': 14,
    'figure.titlesize': 20,
    'lines.linewidth': 2.5,
    'axes.linewidth': 1.5,
    'axes.grid': False
})
#%%

size_picture = (20,16)
network_size = size_picture[0]*size_picture[1]
myDir = '/home/saighi/Desktop/data/all_data_splited/sleep_simulations/Fig_Spontaneous_Recollection'
#%%
data_trajs_depressed = utils.load_simulation_trajectories(myDir,'depressed_traj_')[:-1]
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
fig, axes = plt.subplots(len(data_trajs_depressed), nb_plot_depressed+nb_plot_not_depressed+1, figsize=(10, 8), sharey=True)
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
                      vmin=vmin, vmax=vmax, cmap='grey')
        ax.set_xticks([])  # Remove x ticks
        ax.set_yticks([])  # Remove y ticks
        # if i ==0:
        #     if j ==0:
        #         ax.set_title("t="+str(int(times_depressed[j])))
        #     else :
        #         ax.set_title("t="+str(int(times_depressed[j])+1))

    for j in range(nb_plot_not_depressed):
        ax = axes[i][j+nb_plot_depressed+1]
        im = ax.imshow(activity_data_not_depressed[-1].reshape((size_picture[0], size_picture[1])), 
                      vmin=vmin, vmax=vmax, cmap='grey')
        # im = ax.imshow(activity_data_not_depressed[int(times_not_depressed[j])].reshape((size_picture[0], size_picture[1])), 
        #               vmin=vmin, vmax=vmax, cmap='viridis')
        ax.set_xticks([])  # Remove x ticks
        ax.set_yticks([])  # Remove y ticks
        # if i ==0:
        #     if j ==0:
        #         ax.set_title(r"$t={:.1f}".format(int(times_depressed[j])))
        #     else :
        #         ax.set_title(r"$t={:.1f}".format(int(times_depressed[j]+1)))
    
    ax = axes[i][0]
    im_inhib = ax.imshow(inhib_drive, cmap='Reds', vmin=min_inhib_drive, vmax=max_inhib_drive)
    ax.set_xticks([])  # Remove x ticks
    ax.set_yticks([])  # Remove y ticks

# cbar = fig.colorbar(im, ax=axes, shrink=0.3, orientation='horizontal', location='bottom', pad=0.05)
# cbar.set_label('v(t)', labelpad=20)

cbar_ax = fig.add_axes([0.453, 0.05, 0.4, 0.015])  # Adjust 'left' (0.85) to shift horizontally
cbar = fig.colorbar(im, cax=cbar_ax, shrink=0.3, orientation='horizontal', location='bottom', pad=0.05)
cbar.set_label(r'$v(t)$', labelpad=6)

cbar_inhib = fig.colorbar(im_inhib, ax=axes, shrink=0.4, orientation='vertical', location='left', pad=0.05)
cbar_inhib.set_label(r'$\mathbf{I}^{inh}$', labelpad=8)
plt.tight_layout(rect=[1, 0.07, 1, 1.5])  # leave space at bottom for horizontal colorbar

# Add iteration labels on the right of each row
n_rows = len(data_trajs_depressed)
for i in range(n_rows):
    # Compute vertical position in figure coordinates
    ypos = 0.8 - i * (0.8 / n_rows)  # empirically adjusted
    fig.text(0.91, ypos, fr"iter n°{i+1}", va='center', ha='left')

fig.savefig("./plots/Fig_mnist_autonomous_rehearsal.png",transparent=True,dpi=300)
#%%