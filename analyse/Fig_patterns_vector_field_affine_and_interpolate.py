#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid") 
sns.set_context("paper", font_scale=1.5)

def plot_dotproduct_affine_plane(filename, scale=10):
    """
    Plot a 2D field from the file produced by:
      compute_and_save_rate_vector_field_two_pattern_dotproduct(...)

    Each line in 'filename' has:
       x  y  dx  dy  v0 ... vN-1
    where:
      - (x,y) are integers in [0..N-1],
      - dx,dy = derivative in 'p1'/'p2' directions (via dot products),
      - v0..vN-1 = the actual 10D (or ND) rate vector.

    We then:
      1) Load the data
      2) Check it forms an NxN grid => total_points = N*N
      3) Rescale x,y to [0..1] by dividing by (N-1).
      4) Reshape for NxN and do quiver + stream plots.
    """

    data = np.loadtxt(filename)
    x  = data[:, 0]
    y  = data[:, 1]
    dx = data[:, 2]
    dy = data[:, 3]
    # If needed, rates = data[:, 4:]  # shape (N*N, size)

    total_points = len(x)
    N = int(np.sqrt(total_points))
    if N * N != total_points:
        raise ValueError(f"Data has {total_points} rows, not NxN. Check input file.")

    # 1) Rescale x,y from [0..N-1] => [0..1]
    x = x / (N - 1.0)
    y = y / (N - 1.0)

    # 2) Reshape for NxN
    X  = x.reshape(N, N)
    Y  = y.reshape(N, N)
    DX = dx.reshape(N, N)
    DY = dy.reshape(N, N)

    # 3) If Y is descending in rows, flip vertically so row 0 is bottom
    if not np.all(np.diff(Y[:,0])>0):
        X  = np.flipud(X)
        Y  = np.flipud(Y)
        DX = np.flipud(DX)
        DY = np.flipud(DY)

    # 4) Make subplots
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12,6))

    # --- Quiver Plot ---
    Q = ax0.quiver(X, Y, DX, DY, angles='xy', scale_units='xy',
                   scale=scale, color='red')
    ax0.set_title("Quiver: Dot-Product Projected Plane (p1,p2)")
    ax0.set_xlabel("Alpha in [0..1]")
    ax0.set_ylabel("Beta in [0..1]")
    ax0.axis("equal")
    ax0.set(xlim=(-0.1,1.1), ylim=(-0.1,1.1))

    # --- Stream Plot ---
    strm = ax1.streamplot(X, Y, DX, DY, color='blue', density=1.0)
    ax1.set_title("Stream: Dot-Product Projected Plane (p1,p2)")
    ax1.set_xlabel("Alpha in [0..1]")
    ax1.set_ylabel("Beta in [0..1]")
    ax1.axis("equal")

    plt.tight_layout()
    plt.show()

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

    # 3) Rescale x,y => [0..1]
    x /= x.max()
    y /= y.max()

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
    strm = ax0.streamplot(X, Y, DX, DY, density=1.0, color='tab:blue',arrowsize=1.5)
    neutral_point=np.full(len(data[0][4:]),0.5)
    distances = []
    for i in range(len(x)):
        distances.append(np.linalg.norm(data[i][4:]-neutral_point))
    nearest_point = data[np.argmin(distances)][:2]
    print(nearest_point) 
    plt.plot(nearest_point[0], nearest_point[1], 'X', markersize=10, c="red")
    ax0.set_xlabel(r"$\lambda$ (0 $\to$ p1)")
    ax0.set_ylabel(r"$\lambda$ (0 $\to$ p2)")
    ax0.set(xlim=(0, 1), ylim=(0, 1))
    # ax1.axis("equal")
    plt.tight_layout()
    plt.show()
#%%
# Example usage
#%%
folder = "../../data/all_data_splited/trained_networks_fast/Fig_patterns_vector_field/sim_nb_0/"
file_name = "vector_field_two_patterns_"
pre_file  = folder +file_name+"pre_train_interpolate_plane.txt"
post_training_file = folder +file_name+"post_train_interpolate_plane.txt"
post_inhib_file = folder +file_name+ "post_inhib_interpolate_plane.txt"
# Plot the 'pre-training' field
plot_dotproduct_interpolate_plane(pre_file, scale=10)

# Plot the 'post-training' field
plot_dotproduct_interpolate_plane(post_training_file, scale=30)

# Plot the 'post-inhib' field
plot_dotproduct_interpolate_plane(post_inhib_file, scale=30)

#%%
pre_file  = folder +file_name+ "pre_train_affine_plane.txt"
post_training_file = folder +file_name+ "post_train_affine_plane.txt"
post_inhib_file = folder +file_name+ "post_inhib_affine_plane.txt"
# Plot the 'pre-training' field
plot_dotproduct_interpolate_plane(pre_file, scale=10)

# Plot the 'post-training' field
plot_dotproduct_affine_plane(post_training_file, scale=30)

# Plot the 'post-inhib' field
plot_dotproduct_affine_plane(post_inhib_file, scale=30)

# %%
