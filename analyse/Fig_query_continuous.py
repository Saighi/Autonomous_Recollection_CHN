#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import regex as re
import utils 
import matplotlib.animation as animation
import seaborn as sns 
#%%
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
    'lines.linewidth': 2.5,
    'axes.linewidth': 1.5,
    'axes.grid': False,
    'font.weight' : 'bold'
})
#%%
size_picture = (20,16)
myDir = '/home/saighi/Desktop/data/all_data_splited/trained_networks_fast/Fig_Query_continuous'
#%%
# Read the CSV file
data = pd.read_csv(myDir+'/all_simulation_data.csv')
#%%
# data_trajs = utils.load_simulation_trajectories(myDir,'results')
#%%
folder = myDir+"/sim_nb_0/"
results_0 = np.loadtxt(myDir+"/sim_nb_0/results_0.data")
results_1 = np.loadtxt(myDir+"/sim_nb_0/results_1.data")
fig, axes = plt.subplots(2, 4, figsize=(10, 4), sharey=True)
for i,ax in enumerate(axes[0]):
    print(int(i*(len(results_0)/len(axes[0]))))
    im =ax.imshow(results_0[int(i*(len(results_0)/len(axes[0])))].reshape((size_picture[0], size_picture[1])), cmap='gray')
    ax.set_title("t="+ str(int(i*(len(results_0)/len(axes[0])))))
    ax.set_xticks([])
    ax.set_yticks([])
cbar = fig.colorbar(im, ax=axes, pad=0.04,shrink=0.8)
cbar.set_label(r'$v(t)$',fontsize=18,labelpad=10)
for i,ax in enumerate(axes[1]):
    print(int(i*(len(results_1)/len(axes))))
    im =ax.imshow(results_1[int(i*(len(results_1)/len(axes[1])))].reshape((size_picture[0], size_picture[1])), cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
fig.text(0.1, 0.7, 'Q1', ha='left', va='center')
fig.text(0.1, 0.3, 'Q2', ha='left', va='center')
fig.savefig(fname="./plots/Fig_querying",transparent=True,dpi=300)

# %%
