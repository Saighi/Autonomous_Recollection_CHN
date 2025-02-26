#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid") 
sns.set_context("paper", font_scale=1.5)

def plot_dotproduct_interpolate_plane(filename, scale):
    """
    Plots a quiver + stream field in the 2D plane spanned (affinely) by:
      - pattern_1_rate → fully_inactivated
      - pattern_2_rate → fully_activated

    Each row in 'filename' has:
        x   y   dx   dy   v0 ... vN-1
    where x,y in [0..N-1] or so, then we'll rescale to [0..1]. 
    dx,dy are the projected derivatives in that plane.

    CAPTION:
    This figure shows how the local derivative flows in the 2D space parameterized
    by λ for pattern1 → 0 (horizontal axis) and λ for pattern2 → 1 (vertical axis).
    """
    # 1) Load data
    data = np.loadtxt(filename)
    x  = data[:, 0]
    y  = data[:, 1]
    dx = data[:, 2]
    dy = data[:, 3]
    # If needed, the rest: rates = data[:, 4:] 

    # 2) Determine grid size N
    total_points = len(x)
    N = int(np.sqrt(total_points))
    if N*N != total_points:
        raise ValueError(f"Data has {total_points} rows, not an NxN grid.")

    # # 3) Rescale x,y => [0..1]
    # x /= x.max()
    # y /= y.max()

    # 4) Reshape for NxN
    X  = x.reshape(N, N).T
    Y  = y.reshape(N, N).T
    DX = dx.reshape(N, N).T
    DY = dy.reshape(N, N).T

    # 5) Ensure ascending Y
    if not np.all(np.diff(Y[:,0]) > 0):
        X  = np.flipud(X)
        Y  = np.flipud(Y)
        DX = np.flipud(DX)
        DY = np.flipud(DY)

    # 6) Build subplots
    # fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6))
    fig, ax0 = plt.subplots(1, 1, figsize=(6, 6))

    # --- (a) Quiver ---
    # Q = ax0.quiver(X, Y, DX, DY, angles='xy', scale_units='xy',
    #                scale=scale, color='blue')
    # x-axis: λ for p1→0, y-axis: λ for p2→1
    # ax1.set_xlabel(r"$\lambda$ (0 $\to$ p1)")

    # --- (b) Stream ---    

    strm = ax0.streamplot(X, Y, DX,DY, density=0.8, color='tab:blue',arrowsize=1.7)
    ax0.set_xlabel(r"$\lambda_1$")
    ax0.set_ylabel(r"$\lambda_2$")
    # ax0.set(xlim=(-0.25, 1), ylim=(-0.25, 1))

    # Create a twin axes for the top and right spines
    # ax_top = ax0.twiny()  # Create twin axes for top x-axis
    # ax_right = ax0.twinx()  # Create twin axes for right y-axis
    # ax_top.set(ylim=(-0.25, 1))
    # ax_right.set(xlim=(-0.25, 1))
    # ax_top.set_xlabel(r"$\lambda$ for $\mathcal{I}(\lambda,\mathbf{v}^2,\mathbf{1}_N$)")
    # ax_right.set_ylabel(r"$\lambda$ for $\mathcal{I}(\lambda,\mathbf{v}^1,\mathbf{1}_N$)")

    # ax_top.grid(False)
    # ax_right.grid(False)

    ax0.plot(0, 0   , 'o', markersize=10, c="red") 
    ax0.plot(1, 0   , 'o', markersize=4.5, c="red") 
    ax0.plot(0, 1   , 'o', markersize=4.5, c="red") 

    ax0.text(0.96, -0.08, r"$\mathbf{1}$",c="red",fontsize=15)
    ax0.text(-0.015, 1.03, r"$\mathbf{2}$",c="red",fontsize=15)
    # ax0.text(0.93, -0.01, r"$\mathbf{1}$",c="red",fontsize=15)
    # ax0.text(-0.015, 1.03, r"$\mathbf{2}$",c="red",fontsize=15)

    # plt.tight_layout()
    
#%%
# Example usage
#%%
folder = "../../data/all_data_splited/trained_networks_fast/Fig_patterns_vector_field/sim_nb_0/"
file_name = "vector_field_two_patterns_"
pre_file  = folder +file_name+"pre_train.txt"
post_training_file = folder +file_name+"post_train.txt"
post_inhib_file = folder +file_name+ "post_inhib.txt"
post_inhib_2_file = folder +file_name+ "post_inhib_2.txt"
post_weight_sum_null_file = folder +file_name+ "post_null_w.txt"
# Plot the 'pre-training' field

plot_dotproduct_interpolate_plane(pre_file, scale=30)
plt.show()
# Plot the 'post-training' field
plot_dotproduct_interpolate_plane(post_training_file, scale=30)
# patterns = np.loadtxt(folder+"patterns.data")
# first_trajectory = np.loadtxt(folder+"results_evolution_1.data")
# project_traj_pat_1 = np.dot(first_trajectory,patterns[0])/np.sqrt(np.sum(first_trajectory[0])*np.sum(patterns[0]))
# project_traj_pat_2 = np.dot(first_trajectory,patterns[1])/np.sqrt(np.sum(first_trajectory[1])*np.sum(patterns[1]))
# plt.plot(project_traj_pat_1,project_traj_pat_2)
# print(project_traj_pat_1)
plt.show()
# Plot the 'post-inhib' field
plot_dotproduct_interpolate_plane(post_inhib_file, scale=30)
plt.show()
# Plot the 'post-inhib_2' field
plot_dotproduct_interpolate_plane(post_inhib_2_file, scale=30)
plt.show()
plot_dotproduct_interpolate_plane(post_weight_sum_null_file, scale=30)
plt.show()
# %%
