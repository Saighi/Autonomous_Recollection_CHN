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
myDir = "/home/saighi/Desktop/data/all_data_splited/sleep_simulations/Fig_Spontaneous_Recollection_perfect_spurious/sim_nb_0"
#%%
data_trajs = np.loadtxt(myDir+'/results_depressed_4.data')
#%%
picture = data_trajs[-1].reshape(size_picture)
#%%
plt.imshow(picture,cmap="viridis")

# %%
