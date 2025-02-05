#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")  # or "whitegrid", "dark", "white", "ticks"
sns.set_context("paper", font_scale=1.5)  # or "paper", "talk", "poster"
#%%
def plot_pattern_plane_vector_field(filename):
    """
    Expects each line of 'filename' to be:
      x  y  dx  dy  v0 ... v(size-1)

    Where x,y is a flattened (N x N) grid. Typically:
      - x,y in [0..N-1]
      - Then dx,dy are the projected derivatives in 2D
      - v0...vN-1 are the high-dimensional rates (optional for plotting).
    """

    # 1) Load data
    data = np.loadtxt(filename)
    x  = data[:, 0]
    y  = data[:, 1]
    dx = data[:, 2]
    dy = data[:, 3]
    # rates = data[:, 4:]  # If you need them: v0..v9 etc.

    # 2) Figure out N = sqrt(number_of_points)
    total_points = len(x)
    N = int(np.sqrt(total_points))
    if N * N != total_points:
        raise ValueError(f"Data has {total_points} rows, not a perfect square.")

    # 3) Rescale (x,y) to [0..1] by dividing by (N-1)
    x = x / (N - 1.0)
    y = y / (N - 1.0)

    # 4) Reshape into (N,N) for each of X,Y,DX,DY
    X  = x.reshape(N, N)
    Y  = y.reshape(N, N)
    DX = dx.reshape(N, N)
    DY = dy.reshape(N, N)

    # 5) Ensure that Y is strictly increasing in the streamplot sense.
    #    Matplotlib expects Y[0,:] < Y[1,:] < ... if you want a "normal" orientation.
    #    However, if your coordinate_square_mesh built Y from top to bottom,
    #    it could be reversed. Let's check and flip if needed:
    if not np.all(np.diff(Y[:, 0]) > 0):
        # Flip all arrays in the vertical (row) direction
        X  = np.flipud(X)
        Y  = np.flipud(Y)
        DX = np.flipud(DX)
        DY = np.flipud(DY)

    # Now we have a nice ascending Y from bottom to top.

    # 6) Build subplots
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12,6))

    # --- (a) Quiver Plot ---
    Q = ax0.quiver(X, Y, DX, DY, angles='xy', scale_units='xy',
                   scale=30, color='red')
    ax0.set_title("Quiver Plot (2D plane spanned by patterns)")
    ax0.set_xlabel("x in [0..1]")
    ax0.set_ylabel("y in [0..1]")
    ax0.axis("equal")
    ax0.set(xlim=(-0.2, 1.2), ylim=(-0.2, 1.2))

    # --- (b) Stream Plot ---
    strm = ax1.streamplot(X, Y, DX, DY, color='blue', density=1.0)
    ax1.set_title("Stream Plot (2D plane spanned by patterns)")
    ax1.set_xlabel("x in [0..1]")
    ax1.set_ylabel("y in [0..1]")
    ax1.axis("equal")

    plt.tight_layout()
    plt.show()


#%%
# Example usage with your file:
sim_foldername=  "../../data/all_data_splited/trained_networks_fast/"
sim_foldername_2 = "Fig_patterns_vector_field/sim_nb_0/"
filename_pre = "vector_field_two_patterns_pre_training.txt"
filename_post = "vector_field_two_patterns_post_training.txt"

filename = sim_foldername+sim_foldername_2+filename_pre

plot_pattern_plane_vector_field(filename)

filename = sim_foldername+sim_foldername_2+filename_post
plot_pattern_plane_vector_field(filename)


# %%
