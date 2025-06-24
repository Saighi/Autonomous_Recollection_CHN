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

# plt.rcParams['text.usetex'] = True
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['Computer Modern Roman']
plt.rcParams.update({
    'text.usetex': True,
    'font.size': 17,
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'xtick.labelsize': 13,
    'ytick.labelsize': 15,
    'legend.fontsize': 14,
    'figure.titlesize': 18,
    'lines.linewidth': 2.5,
    'axes.linewidth': 1.5,
    'axes.grid': False
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
cbar = fig.colorbar(im, ax=axes, pad=0.02)
cbar.set_label(r'$v(t)$',fontsize=18,labelpad=10)
for i,ax in enumerate(axes[1]):
    print(int(i*(len(results_1)/len(axes))))
    im =ax.imshow(results_1[int(i*(len(results_1)/len(axes[1])))].reshape((size_picture[0], size_picture[1])), cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
fig.savefig(fname="./plots/Fig_querying",transparent=True,dpi=300)
# # %%
# def plot_dotproduct_interpolate_plane(ax, filename, display_pattern_numbers = True):
#     """
#     Plots a quiver + stream field in the 2D plane spanned (affinely) by:
#       - pattern_1_rate → fully_inactivated
#       - pattern_2_rate → fully_activated

#     Each row in 'filename' has:
#         x   y   dx   dy   v0 ... vN-1
#     where x,y in [0..N-1] or so, then we'll rescale to [0..1]. 
#     dx,dy are the projected derivatives in that plane.
#     """
#     # 1) Load data
#     data = np.loadtxt(filename)
#     x  = data[:, 0]
#     y  = data[:, 1]
#     dx = data[:, 2]
#     dy = data[:, 3]
    
#     # 2) Determine grid size N
#     total_points = len(x)
#     N = int(np.sqrt(total_points))
#     if N*N != total_points:
#         raise ValueError(f"Data has {total_points} rows, not an NxN grid.")

#     # Reshape for NxN
#     X = x.reshape(N, N).T
#     Y = y.reshape(N, N).T
#     DX = dx.reshape(N, N).T
#     DY = dy.reshape(N, N).T

#     # Ensure ascending Y
#     if not np.all(np.diff(Y[:,0]) > 0):
#         X = np.flipud(X)
#         Y = np.flipud(Y)
#         DX = np.flipud(DX)
#         DY = np.flipud(DY)

#     # Create evenly spaced grid for streamplot
#     x_min, x_max = np.min(X), np.max(X)
#     y_min, y_max = np.min(Y), np.max(Y)
    
#     # Create new grid with evenly spaced points
#     xi = np.linspace(x_min, x_max, N)
#     yi = np.linspace(y_min, y_max, N)
    
#     # Stream plot with interpolated data
#     strm = ax.streamplot(xi, yi, DX, DY, density=1, color='tab:blue', arrowsize=1.7)
    
#     if display_pattern_numbers:
#         # Add points and labels
#         ax.plot(0, 0, 'o', markersize=10, c="red") 
#         ax.plot(1, 0, 'o', markersize=4.5, c="red") 
#         ax.plot(0, 1, 'o', markersize=4.5, c="red") 

#         ax.text(0.96, +0.08, r"$\mathbf{2}$", c="red", fontsize=20)
#         ax.text(+0.08, 0.95, r"$\mathbf{1}$", c="red", fontsize=20)
        
#     return xi, yi

# def plot_trajectory(ax, patterns, traj):
#     coordinates = (((patterns*2)-1) @ ((traj.T*2)-1) / len(patterns[1]))
#     x = coordinates[0]
#     y = coordinates[1]
#     return ax.plot(x, y, c='red', linewidth=2.0)[0]

# # %%
# post_training_file = folder+ "vector_field_two_patterns_post_train.txt"
# patterns_file = folder+"patterns.data"
# first_trajectory_file = folder + "results_0.data"
# second_trajectory_file = folder + "results_1.data"
# # %%
# patterns = np.loadtxt(patterns_file)
# first_trajectory = np.loadtxt(first_trajectory_file)
# second_trajectory = np.loadtxt(second_trajectory_file)

# # %%

# fig, axs = plt.subplots(1, 1, figsize=(5, 5), sharey=True, gridspec_kw={'wspace': 0.05})
# # Plot 1: Pre-training field
# plot_dotproduct_interpolate_plane(axs, post_training_file, False)
# axs.set_xlabel(r"$\lambda_1$")
# axs.set_ylabel(r"$\lambda_2$")
# # axs[0].set_title(titles[0])
# # %%

# %%
