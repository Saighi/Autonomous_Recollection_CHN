#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import griddata

# Match the style/formatting of Fig_pattern_vector_field_3plots.py
sns.set_theme(style="ticks")
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
    'font.weight': 'bold',
})


def plot_stream_field(ax, filename, display_pattern_numbers=True, enhance_weak=False, weak_threshold=1e-6):
    """Stream plot using the exact formatting used in 3plots."""
    data = np.loadtxt(filename)
    x, y, dx, dy = data[:, 0], data[:, 1], data[:, 2], data[:, 3]

    total_points = len(x)
    N = int(np.sqrt(total_points))
    if N * N != total_points:
        raise ValueError(f"Data has {total_points} rows, not an NxN grid.")

    X = x.reshape(N, N).T
    Y = y.reshape(N, N).T
    DX = dx.reshape(N, N).T
    DY = dy.reshape(N, N).T

    if not np.all(np.diff(Y[:, 0]) > 0):
        X = np.flipud(X)
        Y = np.flipud(Y)
        DX = np.flipud(DX)
        DY = np.flipud(DY)

    # Create evenly spaced grid for streamplot
    x_min, x_max = np.min(X), np.max(X)
    y_min, y_max = np.min(Y), np.max(Y)
    xi = np.linspace(x_min, x_max, N)
    yi = np.linspace(y_min, y_max, N)

    # Optionally enhance weak fields (useful for inhib-only cases)
    use_DX, use_DY = DX, DY
    if enhance_weak:
        speed = np.hypot(DX, DY)
        max_speed = float(np.nanmax(speed)) if speed.size else 0.0
        if max_speed == 0.0:
            # Nothing to stream; annotate and lightly shade background for clarity
            ax.set_facecolor('#f7f7f7')
            ax.text(0.5, 0.5, 'No inhibitory flow', transform=ax.transAxes,
                    ha='center', va='center', fontsize=21, color='black')
        elif max_speed < weak_threshold:
            # Normalize vectors to emphasize direction (direction-only view)
            with np.errstate(invalid='ignore', divide='ignore'):
                inv = 1.0 / np.where(speed == 0, 1.0, speed)
                use_DX = DX * inv
                use_DY = DY * inv

    # Use the same streamplot style as 3plots
    ax.streamplot(xi, yi, use_DX, use_DY, density=0.9, color='tab:blue', arrowsize=2.5, linewidth=2)

    if display_pattern_numbers:
        ax.plot(0, 0, 'o', markersize=10, c="red")
        ax.plot(1, 0, 'o', markersize=4.5, c="red")
        ax.plot(0, 1, 'o', markersize=4.5, c="red")

        # Use the same label positions/fonts as 3plots
        ax.text(0.92, +0.07, r" $\mathbf{2}$", c="red", fontsize=31, fontfamily="monospace")
        ax.text(+0.08, 0.95, r" $\mathbf{1}$", c="red", fontsize=31, fontfamily="monospace")


def plot_trajectory(ax, patterns, traj):
    coordinates = (((patterns * 2) - 1) @ ((traj.T * 2) - 1) / len(patterns[0]))
    x = coordinates[0]
    y = coordinates[1]
    return ax.plot(x, y, c='red', linewidth=3.0)[0]


def load_energy_field(filename):
    """Load energy field file and return X, Y, E ensured with ascending Y."""
    data = np.loadtxt(filename)
    x, y, energy = data[:, 0], data[:, 1], data[:, 2]

    total_points = len(x)
    N = int(np.sqrt(total_points))
    if N * N != total_points:
        raise ValueError(f"Data has {total_points} rows, not an NxN grid.")

    X = x.reshape(N, N).T
    Y = y.reshape(N, N).T
    E = energy.reshape(N, N).T

    if not np.all(np.diff(Y[:, 0]) > 0):
        X = np.flipud(X)
        Y = np.flipud(Y)
        E = np.flipud(E)

    return X, Y, E


def plot_energy_field(ax, X, Y, E, vmin=None, vmax=None, show_pattern_labels=True):
    cs = ax.contourf(X, Y, E, 60, cmap='viridis', vmin=vmin, vmax=vmax)
    lines = ax.contour(X, Y, E, 8, colors='white', alpha=0.5, linewidths=0.5)
    ax.clabel(lines, inline=True, fontsize=8, fmt='%.1f')

    if show_pattern_labels:
        ax.plot(0, 0, 'o', markersize=10, c="red")
        ax.plot(1, 0, 'o', markersize=4.5, c="red")
        ax.plot(0, 1, 'o', markersize=4.5, c="red")
        # Match 3plots labeling style
        ax.text(0.92, +0.07, r" $\mathbf{2}$", c="red", fontsize=31, fontfamily="monospace")
        ax.text(+0.08, 0.95, r" $\mathbf{1}$", c="red", fontsize=31, fontfamily="monospace")

    return cs


#%%
# Files and configuration
folder = "../../data/all_data_splited/trained_networks_fast/Fig_vector_fields_patterns_different_distances/sim_nb_0/"
vec_prefix = folder + "vector_field_two_patterns_"
eng_prefix = folder + "energy_field_two_patterns_"
inh_eng_prefix = folder + "energy_field_two_patterns_inhib_only_"
stages = ["pre_train", "post_train", "post_inhib"]

patterns_file = folder + "patterns.data"
trajectory_file = folder + "results_evolution_"

# Load shared data
patterns = np.loadtxt(patterns_file)
#%%
# 4-row plot: classic stream (row 1), energy (row 2), excit-only stream (row 3), inhib-only stream (row 4)
import os

excit_prefix = folder + "vector_field_two_patterns_excit_only_"
inhib_prefix = folder + "vector_field_two_patterns_inhib_only_"
# Use the full set of stages present for the new files
stages_4rows = ["pre_train", "post_train", "iter_1","iter_2"]
stage_titles = {
    "pre_train": "Pre-training",
    "post_train": "Post-training",
    "post_inhib": "Post-inhibition",
}

# Two-row energy-only figure (W+B+A on row 1; A-only on row 2)
import os

# Columns to display: post_train, iter_1, iter_2, iter_3 (no pre_train)
cols = ["post_train", "iter_1", "iter_2", "iter_3"]

# Prepare figure (GridSpec): 2 rows x 4 columns + 1 right-label column
fig2 = plt.figure(figsize=(26, 14))
gs2 = fig2.add_gridspec(nrows=2, ncols=5, width_ratios=[1, 1, 1, 1, 0.05], wspace=0.06, hspace=0.18)
axs2 = np.empty((2, 4), dtype=object)
for i in range(4):
    axs2[0, i] = fig2.add_subplot(gs2[0, i])
    axs2[1, i] = fig2.add_subplot(gs2[1, i])

# Right-side labels for both rows
right_axes2 = [fig2.add_subplot(gs2[i, 4]) for i in range(2)]
right_labels2 = [r"$\mathbf{W} + \mathbf{B} + \mathbf{A}$", r"$\mathbf{A}$"]
for ax_lab, txt in zip(right_axes2, right_labels2):
    ax_lab.set_axis_off()
    ax_lab.text(0.5, 0.5, txt, ha='center', va='center')

sns.despine(left=True, bottom=True, top=True, right=True)

# Load energy fields for both rows and compute global normalization
full_fields = []
inh_fields = []
for st in cols:
    full_fields.append(load_energy_field(eng_prefix + st + ".txt"))
    inh_fields.append(load_energy_field(inh_eng_prefix + st + ".txt"))

vmin_all = min(min(np.min(E) for (_, _, E) in full_fields),
               min(np.min(E) for (_, _, E) in inh_fields))
vmax_all = max(max(np.max(E) for (_, _, E) in full_fields),
               max(np.max(E) for (_, _, E) in inh_fields))

# Mapping for trajectory overlay: show next-iteration trajectory on the previous column
traj_map = {
    "post_train": "iter_1",
    "iter_1": "iter_2",
    "iter_2": "iter_3",
    # "iter_3": None  # final, no trajectory
}

last_cs = None
# Row 1: W+B+A energy
for j, st in enumerate(cols):
    X, Y, E = full_fields[j]
    ax = axs2[0, j]
    last_cs = plot_energy_field(ax, X, Y, E, vmin=vmin_all, vmax=vmax_all, show_pattern_labels=True)
    # Overlay offset trajectory if available
    next_stage = traj_map.get(st)
    if next_stage is not None:
        traj_path = trajectory_file + next_stage + ".data"
        if os.path.exists(traj_path):
            data_traj = np.loadtxt(traj_path)
            plot_trajectory(ax, patterns, data_traj)
    ax.set_xticks([0, 0.5, 1])
    if j == 0:
        ax.set_ylabel(r"$\lambda_2$", rotation=90)
    ax.set_title(st.replace('_', ' '))

# Row 2: A-only energy
for j, st in enumerate(cols):
    X, Y, E = inh_fields[j]
    ax = axs2[1, j]
    cs = plot_energy_field(ax, X, Y, E, vmin=vmin_all, vmax=vmax_all, show_pattern_labels=True)
    # Overlay offset trajectory if available (same mapping)
    next_stage = traj_map.get(st)
    if next_stage is not None:
        traj_path = trajectory_file + next_stage + ".data"
        if os.path.exists(traj_path):
            data_traj = np.loadtxt(traj_path)
            plot_trajectory(ax, patterns, data_traj)
    ax.set_xticks([0, 0.5, 1])
    if j == 0:
        ax.set_ylabel(r"$\lambda_2$", rotation=90)
    ax.set_xlabel(r"$\lambda_1$")

# Shared vertical colorbar to the left, spanning both rows
if last_cs is not None:
    pos_r1 = axs2[0, 0].get_position()
    pos_r2 = axs2[1, 0].get_position()
    total_h = pos_r1.y1 - pos_r2.y0
    shrink = 0.8
    cbar_h = total_h * shrink
    cbar_y0 = pos_r2.y0 + (total_h - cbar_h) / 2
    cbar_width = 0.02
    left_offset = 0.06
    cbar_x0 = max(0.01, pos_r1.x0 - (left_offset + cbar_width))
    cax = fig2.add_axes([cbar_x0, cbar_y0, cbar_width, cbar_h])
    cbar = fig2.colorbar(last_cs, cax=cax)
    cbar.set_label('Energy')
    cbar.ax.yaxis.set_ticks_position('left')
    cbar.ax.yaxis.set_label_position('left')
    cbar.ax.tick_params(labelleft=True, labelright=False)

# Consistent limits and aspect
for r in range(2):
    for c in range(len(cols)):
        ax = axs2[r, c]
        if ax.has_data():
            ax.set_xlim(-0.1, 1.1)
            ax.set_ylim(-0.1, 1.1)
            ax.set_aspect('equal')

plt.savefig('plots/Fig_pattern_vector_field_distant_many_iter_2rows_energy.png', dpi=300, bbox_inches='tight')
