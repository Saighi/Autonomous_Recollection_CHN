#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import regex as re
import utils 
import matplotlib.animation as animation

#%%
size_picture = (20,16)
myDir = '..\\..\\..\\data\\all_data_splited\\trained_networks_fast\\Fig_Query_continuous'
#%%
# Read the CSV file
data = pd.read_csv(myDir+'\\all_simulation_data.csv')
#%%
data_trajs = utils.load_simulation_trajectories(myDir,'results')
#%%
activity_data = data_trajs[0][0]

# Create a figure and axis for the plot
fig, ax = plt.subplots()

# Initialize the grid with the first time step
initial_activity = activity_data[0].reshape((size_picture[0], size_picture[1]))
im = ax.imshow(initial_activity, cmap='viridis', interpolation='nearest')

# Add a colorbar to show the mapping from activity levels to colors
cbar = fig.colorbar(im, ax=ax)
cbar.set_label('Activity Level')

# Function to update the grid for each frame
def update(frame):
    activity = activity_data[frame].reshape((size_picture[0], size_picture[1]))
    im.set_array(activity)
    return [im]

# Create the animation
ani = animation.FuncAnimation(
    fig, update, frames=len(activity_data), blit=True, interval=100
)

# To display the animation inline (if using a Jupyter notebook)
# from IPython.display import HTML
# HTML(ani.to_jshtml())

# To save the animation to a file (e.g., MP4 video)
ani.save('network_activity.mp4', writer='ffmpeg')

# Add a colorbar to show the mapping from activity levels to colors
cbar = fig.colorbar(im, ax=ax)
cbar.set_label('Activity Level')

plt.show()
#%%
plt.imshow(activity_data[0].reshape((size_picture[0], size_picture[1])))
#%%
#%%
fig, axes = plt.subplots(1, 5, figsize=(25, 5), sharey=True)
for i,ax in enumerate(axes):
    print(int(i*(len(activity_data)/len(axes))))
    ax.imshow(activity_data[int(i*(len(activity_data)/len(axes)))].reshape((size_picture[0], size_picture[1])))
cbar = fig.colorbar(im, ax=axes)
cbar.set_label('Activity Level')
# %%
