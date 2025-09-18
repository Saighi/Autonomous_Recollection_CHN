#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="ticks")

# sns.set_style("darkgrid")  # or "whitegrid", "dark", "white", "ticks"
# sns.set_context("paper", font_scale=1.5)  # or "paper", "talk", "poster"
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times']
plt.rcParams.update({
    'font.size': 30,
    'axes.labelsize': 30,
    'axes.titlesize': 30,
    'xtick.labelsize': 25,
    'ytick.labelsize': 25,
    'legend.fontsize': 25,
    'figure.titlesize': 20,
    'lines.linewidth': 2.5,
    'axes.linewidth': 1.5,
    'axes.grid': False,
    'font.weight' : 'bold'
})

def plot_dotproduct_interpolate_plane(ax, filename, display_pattern_numbers = True):
    """
    Plots a quiver + stream field in the 2D plane spanned (affinely) by:
      - pattern_1_rate → fully_inactivated
      - pattern_2_rate → fully_activated

    Each row in 'filename' has:
        x   y   dx   dy   v0 ... vN-1
    where x,y in [0..N-1] or so, then we'll rescale to [0..1]. 
    dx,dy are the projected derivatives in that plane.
    """
    # 1) Load data
    data = np.loadtxt(filename)
    x  = data[:, 0]
    y  = data[:, 1]
    dx = data[:, 2]
    dy = data[:, 3]
    
    # 2) Determine grid size N
    total_points = len(x)
    N = int(np.sqrt(total_points))
    if N*N != total_points:
        raise ValueError(f"Data has {total_points} rows, not an NxN grid.")

    # Reshape for NxN
    X = x.reshape(N, N).T
    Y = y.reshape(N, N).T
    DX = dx.reshape(N, N).T
    DY = dy.reshape(N, N).T

    # Ensure ascending Y
    if not np.all(np.diff(Y[:,0]) > 0):
        X = np.flipud(X)
        Y = np.flipud(Y)
        DX = np.flipud(DX)
        DY = np.flipud(DY)

    # Create evenly spaced grid for streamplot
    x_min, x_max = np.min(X), np.max(X)
    y_min, y_max = np.min(Y), np.max(Y)
    
    # Create new grid with evenly spaced points
    xi = np.linspace(x_min, x_max, N)
    yi = np.linspace(y_min, y_max, N)
    
    # Stream plot with interpolated data
    strm = ax.streamplot(xi, yi, DX, DY, density=0.9, color='tab:blue', arrowsize=2.5, linewidth = 2)
    
    if display_pattern_numbers:
        # Add points and labels
        ax.plot(0, 0, 'o', markersize=10, c="red") 
        ax.plot(1, 0, 'o', markersize=4.5, c="red") 
        ax.plot(0, 1, 'o', markersize=4.5, c="red") 

        ax.text(0.92, +0.07, r" $\mathbf{1}$", c="red", fontsize=31,fontfamily="monospace")
        ax.text(+0.08, 0.95, r" $\mathbf{2}$", c="red", fontsize=31,fontfamily="monospace")
        
    return xi, yi

def plot_trajectory(ax, patterns, traj):
    coordinates = (((patterns*2)-1) @ ((traj.T*2)-1) / len(patterns[0]))
    x = coordinates[0]
    y = coordinates[1]
    return ax.plot(x, y, c='red', linewidth=3.0)[0]

#%%
# Define file paths
folder = "../../data/all_data_splited/trained_networks_fast/Fig_vector_fields_patterns_inhib_and_exc/sim_nb_0/"
file_name = "vector_field_two_patterns_"
pre_file = folder + file_name + "pre_train.txt"
post_training_file = folder + file_name + "post_train.txt"
post_inhib_file = folder + file_name + "post_inhib.txt"
patterns_file = folder + "patterns.data"
first_trajectory_file = folder + "results_evolution_1.data"
second_trajectory_file = folder + "results_evolution_2.data"

# Load patterns and trajectories
patterns = np.loadtxt(patterns_file)
first_trajectory = np.loadtxt(first_trajectory_file)
second_trajectory = np.loadtxt(second_trajectory_file)

# Create figure with shared y-axis
fig, axs = plt.subplots(1, 3, figsize=(17, 6), sharey=True, gridspec_kw={'wspace': 0.05})

# Plot titles
titles = ["Pre-training", "Post-training", "Post-inhibition"]
sns.despine(left=True,bottom=True,top=True, right=True)
# Plot 1: Pre-training field
plot_dotproduct_interpolate_plane(axs[0], pre_file, False)
# axs[0].set_xlabel(r"$\lambda_1$")
axs[0].set_ylabel(r"$\lambda_2$",rotation=90)
axs[0].set_title(titles[0])
axs[0].set_xticks([0,0.5,1])
axs[0].set_yticks([0,0.5,1])
# Plot 2: Post-training field with first trajectory
plot_dotproduct_interpolate_plane(axs[1], post_training_file)
plot_trajectory(axs[1], patterns, first_trajectory)
# axs[1].set_xlabel(r"$\lambda_1$")
axs[1].set_title(titles[1])
axs[1].set_xticks([0,0.5,1])

# Plot 3: Post-inhibition field with second trajectory
plot_dotproduct_interpolate_plane(axs[2], post_inhib_file)
plot_trajectory(axs[2], patterns, second_trajectory)
# axs[2].set_xlabel(r"$\lambda_1$")
axs[2].set_title(titles[2])
axs[2].set_xticks([0,0.5,1])
fig.text(0.5,0.01,r"$\lambda_1$",ha="left",va="center")
# Ensure consistent aspect ratio and limits across all plots
for ax in axs:
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_aspect('equal')

# Save figure
plt.savefig('plots/Fig_pattern_vector_field_3plots.png', dpi=300, bbox_inches='tight')
plt.show()

if __name__ == "__main__":
    print("Figure with three vector field plots created and saved.")
# %%
