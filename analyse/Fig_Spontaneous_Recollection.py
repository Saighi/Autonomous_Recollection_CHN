#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import regex as re
import utils 
import matplotlib.animation as animation
plt.rcParams.update({'font.size': 20})
#%%

size_picture = (20,16)
network_size = size_picture[0]*size_picture[1]
myDir = '..\\..\\data\\all_data_splited\\sleep_simulations\\Fig_Spontaneous_Recollection'
#%%
data_trajs_depressed = utils.load_simulation_trajectories(myDir,'results_depressed_')
data_trajs_not_depressed = utils.load_simulation_trajectories(myDir,'results_')
data_inhib_mats = utils.load_simulation_trajectories(myDir,'inhib_matrix_')
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
fig, axes = plt.subplots(5, nb_plot_depressed+nb_plot_not_depressed+1, figsize=(25, 25), sharey=True)
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
# %%
