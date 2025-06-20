#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import regex as re
import utils 
import matplotlib.animation as animation
#%%
plt.rcParams.update({'font.size': 22})
#%%
size_picture = (20,16)
myDir = '/home/saighi/Desktop/data/all_data_splited/trained_networks_fast/Fig_Query_continuous'
#%%
# Read the CSV file
data = pd.read_csv(myDir+'/all_simulation_data.csv')
#%%
# data_trajs = utils.load_simulation_trajectories(myDir,'results')
#%%
results = np.loadtxt(myDir+"/sim_nb_0/results_0.data")
#%%
plt.imshow(results[0].reshape((size_picture[0], size_picture[1])))
#%%
#%%
fig, axes = plt.subplots(1, 4, figsize=(25, 5), sharey=True)
for i,ax in enumerate(axes):
    print(int(i*(len(results)/len(axes))))
    im =ax.imshow(results[int(i*(len(results)/len(axes)))].reshape((size_picture[0], size_picture[1])))
    ax.set_title("t="+ str(int(i*(len(results)/len(axes)))))
cbar = fig.colorbar(im, ax=axes, pad=0.02)
cbar.set_label('Rate',fontsize=25)
# %%
