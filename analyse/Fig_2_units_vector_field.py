#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")  # or "whitegrid", "dark", "white", "ticks"
sns.set_context("paper", font_scale=1.5)  # or "paper", "talk", "poster"
#%%
def load_vector_field(filename):
    """
    Expects lines of the form:
       x  y  dx  dy
    Returns arrays X, Y, U, V
    """
    data = np.loadtxt(filename)
    # data.shape = (N, 4): [x, y, dx, dy]
    X = data[:, 0]
    Y = data[:, 1]
    U = data[:, 2]
    V = data[:, 3]
    return X, Y, U, V

def plot_quiver(X, Y, U, V, x_label, y_label, title="", ax=None):
    if ax is None:
        ax = plt.gca()
    # ax.quiver(X, Y, U, V, color='black', angles='xy', scale_units='xy', scale=20)
    # magnitude = np.sqrt(U**2 + V**2)
    # U_normalized = U / magnitude
    # V_normalized = V / magnitude
    ax.quiver(X, Y, U, V, scale=None)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axis("equal")


def plot_trajectory(filename, ax=None, color='red', label='Trajectory'):
    """
    If you want to overlay your time evolution, e.g. results_evolution.data with lines like:
       time state_0 state_1
    you can read it and plot the (state_0, state_1) path.
    """
    if ax is None:
        ax = plt.gca()
    data = np.loadtxt(filename)
    # Suppose the data is columns: time, r0, r1 (or a0, a1) ...
    # e.g. your run_net_sim_save might store the rates or activities.
    # Adjust accordingly:
    s0 = data[:, 0]
    s1 = data[:, 1]
    ax.plot(s0, s1, '-o', color=color, label=label,markersize=1.5)
    # ax.plot(s0, s1)
    ax.legend()

#%%
# Example usage:
folder = "../../data/all_data_splited/trained_networks_fast/Fig_2_units_vector_field/sim_nb_0"  # Adjust to your actual output folder
# Load pre- and post-training vector fields (potential)
Xpa, Ypa, Upa, Vpa = load_vector_field(folder + "/vector_field_pre_pot.txt")
Xra, Yra, Ura, Vra = load_vector_field(folder + "/vector_field_post_pot.txt")
# Load pre- and post-training vector fields (rate)
Xpr, Ypr, Upr, Vpr = load_vector_field(folder + "/vector_field_pre_rate.txt")
Xrr, Yrr, Urr, Vrr = load_vector_field(folder + "/vector_field_post_rate.txt")
#%%
# Plot them side-by-side
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
plot_quiver(Xpa, Ypa, Upa, Vpa,"u1","u2", "Potential vector field (Pre-Training)", ax=ax[0])
plot_quiver(Xra, Yra, Ura, Vra,"u1","u2", "Potential Space (Post-Training)", ax=ax[1])
plt.tight_layout()
plt.show()
#%%

# Plot them side-by-side
fig2, ax2 = plt.subplots(1, 2, figsize=(10, 5))
plot_quiver(Xpr, Ypr, Upr, Vpr,"v1","v2", "Rate vector field (Pre-Training)", ax=ax2[0])
plot_quiver(Xrr, Yrr, Urr, Vrr,"v1","v2", "Rate vector field (Post-Training)", ax=ax2[1])
plt.tight_layout()
plt.show()
#%%
# Optionally, overlay your time evolution in rate space
# if your results_evolution.data logs (r0, r1).
fig3, ax3 = plt.subplots()
plot_quiver(Xrr, Yrr, Urr, Vrr,"v1","v2", "Rate vector field (Post-Training) + Trajectory", ax=ax3)
plot_trajectory(folder + "/results_evolution.data", ax3)
plt.show()
