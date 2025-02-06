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
    X  = x.reshape(N, N)
    Y  = y.reshape(N, N)
    DX = dx.reshape(N, N)
    DY = dy.reshape(N, N)

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
    print(X)
    print(Y)
    strm = ax0.streamplot(X, Y, DX, DY, density=8.0, color='tab:blue',arrowsize=1.5)

    ax0.set_xlabel(r"$\lambda_1$")
    ax0.set_ylabel(r"$\lambda_2$")
    ax0.set(xlim=(0, 1), ylim=(0, 1))

    # Create a twin axes for the top and right spines
    ax_top = ax0.twiny()  # Create twin axes for top x-axis
    ax_right = ax0.twinx()  # Create twin axes for right y-axis

    # ax_top.set_xlabel(r"$\lambda$ for $\mathcal{I}(\lambda,\mathbf{v}^2,\mathbf{1}_N$)")
    # ax_right.set_ylabel(r"$\lambda$ for $\mathcal{I}(\lambda,\mathbf{v}^1,\mathbf{1}_N$)")

    ax_top.grid(False)
    ax_right.grid(False)
    # ax1.axis("equal")
    neutral_point=np.full(len(data[0][4:]),0.5)
    distances = []
    for i in range(len(x)):
        distances.append(np.linalg.norm(data[i][4:]-neutral_point))
    nearest_point = data[np.argmin(distances)][:2]
    print(nearest_point) 
    plt.plot(nearest_point[0], nearest_point[1], 'X', markersize=10, c="red") 
    # plt.tight_layout()
    plt.show()
#%%
# Example usage
#%%
folder = "../../data/all_data_splited/trained_networks_fast/Fig_patterns_vector_field/sim_nb_0/"
file_name = "vector_field_two_patterns_"
pre_file  = folder +file_name+"pre_train_interpolate_plane.txt"
post_training_file = folder +file_name+"post_train_interpolate_plane.txt"
post_inhib_file = folder +file_name+ "post_inhib_interpolate_plane.txt"
post_inhib_2_file = folder +file_name+ "post_inhib_2_interpolate_plane.txt"
post_weight_sum_null_file = folder +file_name+ "post_null_w_wum_interpolate_plane.txt"
# Plot the 'pre-training' field

plot_dotproduct_interpolate_plane(pre_file, scale=30)

# # Plot the 'post-training' field
# plot_dotproduct_interpolate_plane(post_training_file, scale=30)

# # Plot the 'post-inhib' field
# plot_dotproduct_interpolate_plane(post_inhib_file, scale=30)

# # Plot the 'post-inhib_2' field
# plot_dotproduct_interpolate_plane(post_inhib_2_file, scale=30)

# plot_dotproduct_interpolate_plane(post_weight_sum_null_file, scale=30)
# %%
