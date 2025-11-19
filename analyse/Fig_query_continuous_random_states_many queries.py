#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import regex as re
import utils
import matplotlib.animation as animation
import seaborn as sns
import glob
from mpl_toolkits.axes_grid1 import make_axes_locatable 
#%%
sns.set_theme(style="ticks")

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern Roman']
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
# myDir = '/home/saighi/Desktop/data/all_data_splited/trained_networks_fast/Fig_Query_continuous'
myDir = '../../data/all_data_splited/trained_networks_fast/Fig_Query_continuous_random_states'
#%%
# Read the CSV file
data = pd.read_csv(myDir+'/all_simulation_data.csv')
#%%
# data_trajs = utils.load_simulation_trajectories(myDir,'results')
#%%
folder = myDir+"/sim_nb_0/"

# Auto-discover all results_*.data files
results_files = sorted(glob.glob(folder + "results_*.data"),
                       key=lambda x: int(re.search(r'results_(\d+)\.data', x).group(1)))
print(f"Found {len(results_files)} results files")

# Load all results data
results_data = [np.loadtxt(f) for f in results_files]

# Dynamic layout calculation
n_rows = len(results_data)
n_cols = 4  # Keep 4 time snapshots per row

# Smart figure sizing - 2 inches per row for good proportions
fig_height = 2 * n_rows
fig_width = 10

# Adaptive font scaling for many rows
if n_rows >= 4:
    font_scale = 0.85
else:
    font_scale = 1.0

# Create subplots with adaptive spacing
fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height),
                         sharey=True, gridspec_kw={'hspace': 0.15, 'wspace': 0.05})

# Ensure axes is 2D even for single row
if n_rows == 1:
    axes = axes.reshape(1, -1)

# Plot each results file in its own row
for row_idx, results in enumerate(results_data):
    for col_idx, ax in enumerate(axes[row_idx]):
        time_idx = int(col_idx * (len(results) / n_cols))
        im = ax.imshow(results[time_idx].reshape(size_picture), cmap='gray', vmin=0, vmax=1)

        # Add title only to first row
        if row_idx == 0:
            ax.set_title("t=" + str(time_idx), fontsize=int(16 * font_scale))

        ax.set_xticks([])
        ax.set_yticks([])

# Improved colorbar positioning - fixed height, not spanning all rows
# Create a small axis for the colorbar on the right side
cbar_ax = fig.add_axes([0.92, 0.5 - 0.15, 0.02, 0.3])  # [left, bottom, width, height]
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label(r'$v(t)$', fontsize=int(18 * font_scale), labelpad=10)
cbar.mappable.set_clim(0, 1)  # Ensure colorbar shows 0 to 1 range

# Adjust layout to prevent colorbar overlap
plt.subplots_adjust(right=0.90)

fig.savefig(fname="./plots/Fig_query_continuous_random_states_many_queries", transparent=True, dpi=300, bbox_inches='tight')
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
