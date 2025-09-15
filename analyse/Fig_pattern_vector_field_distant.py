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

# Prepare figure (GridSpec): 4 rows x 3 columns of plots + 1 dedicated colorbar column
fig4 = plt.figure(figsize=(20, 22))
gs4 = fig4.add_gridspec(nrows=4, ncols=4, width_ratios=[1, 1, 1, 0.05], wspace=0.05, hspace=0.18)
axs4 = np.empty((3, 3), dtype=object)
for i in range(3):
    axs4[0, i] = fig4.add_subplot(gs4[0, i])
    axs4[1, i] = fig4.add_subplot(gs4[1, i])
    axs4[2, i] = fig4.add_subplot(gs4[2, i])
    # axs4[3, i] = fig4.add_subplot(gs4[3, i])
# Dedicated colorbar axis for the energy row
cax4 = fig4.add_subplot(gs4[2, 3])

# Right-side labels for rows 0–1 only (leave [2,3] for the colorbar)
right_labels_axes = [fig4.add_subplot(gs4[i, 3]) for i in range(2)]
right_labels = [r"$\mathbf{W} + \mathbf{A}$", r"$\mathbf{A}$"]
for ax_lab, txt in zip(right_labels_axes, right_labels):
    ax_lab.set_axis_off()
    ax_lab.text(0.5, 0.5, txt, ha='center', va='center')

sns.despine(left=True, bottom=True, top=True, right=True)

# Row 1: classic stream plots
for j in range(len(stages_4rows)-1):
    ax = axs4[0, j]
    display_labels = (stages_4rows[j] != "pre_train")
    vec_file = vec_prefix + stages_4rows[j]  + ".txt"
    if not os.path.exists(vec_file):
        ax.text(0.5, 0.5, f"Missing: {os.path.basename(vec_file)}", ha='center', va='center')
        ax.set_axis_off()
        continue
    plot_stream_field(ax, vec_file, display_labels)
    if not j>=len(stages_4rows) and j!=0:
        data_traj = np.loadtxt(trajectory_file+stages_4rows[j+1]+".data")
        plot_trajectory(ax, patterns, data_traj)
    ax.set_title(stage_titles.get(stages_4rows[j] , stages_4rows[j] ))
    ax.set_xticks([0, 0.5, 1])
    if j == 0:
        ax.set_ylabel(r"$\lambda_2$", rotation=90)

# # Row 2: excit-only stream plots
# for j in range(len(stages_4rows)-1):
#     ax = axs4[1, j]
#     display_labels = (stages_4rows[j]  != "pre_train")
#     vec_file = excit_prefix + stages_4rows[j]  + ".txt"
#     if not os.path.exists(vec_file):
#         ax.text(0.5, 0.5, f"Missing: {os.path.basename(vec_file)}", ha='center', va='center')
#         ax.set_axis_off()
#         continue
#     plot_stream_field(ax, vec_file, display_labels)
#     ax.set_xticks([0, 0.5, 1])
#     if j == 0:
#         ax.set_ylabel(r"$\lambda_2$", rotation=90)


# Row 2 energy landscapes inhib only with shared color scale and single colorbar across row
energy_fields_inh = [load_energy_field(inh_eng_prefix +st + ".txt") for st in stages_4rows]
vmin4 = min(np.min(E) for (_, _, E) in energy_fields_inh)
vmax4 = max(np.max(E) for (_, _, E) in energy_fields_inh)

last_cs4 = None
for j in range(len(stages_4rows)-1):
    (X, Y, E) = energy_fields_inh[j]
    ax = axs4[1, j]
    last_cs4 = plot_energy_field(ax, X, Y, E, vmin=vmin4, vmax=vmax4, show_pattern_labels=True)
    # Overlay trajectories for consistency
    stage = stages_4rows[j]
    
    if not j>=len(stages_4rows) and j!=0:
        data_traj = np.loadtxt(trajectory_file+stages_4rows[j+1]+".data")
        plot_trajectory(ax, patterns, data_traj)
        
    ax.set_xticks([0, 0.5, 1])
    if j == 0:
        ax.set_ylabel(r"$\lambda_2$", rotation=90)
    ax.set_xlabel(r"$\lambda_1$")

if last_cs4 is not None:
    cbar = fig4.colorbar(last_cs4, cax=cax4)
    cbar.set_label('Energy')

# Consistent limits and aspect across all subplots
for r in range(3):
    for c in range(len(stages_4rows)-1):
        ax = axs4[r, c]
        if ax.has_data():
            ax.set_xlim(-0.1, 1.1)
            ax.set_ylim(-0.1, 1.1)
            ax.set_aspect('equal')


# # Row 3: inhib-only stream plots
# for j in range(len(stages_4rows)-1):
#     ax = axs4[1, j]
#     display_labels = (stages_4rows[j] != "pre_train")
#     vec_file = inhib_prefix + stages_4rows[j] + ".txt"
#     if not os.path.exists(vec_file):
#         ax.text(0.5, 0.5, f"Missing: {os.path.basename(vec_file)}", ha='center', va='center')
#         ax.set_axis_off()
#         continue
#     # Enhance weak inhibitory fields to avoid blank-looking plots
#     plot_stream_field(ax, vec_file, display_labels, enhance_weak=True, weak_threshold=1e-6)
#     ax.set_xticks([0, 0.5, 1])
#     if j == 0:
#         ax.set_ylabel(r"$\lambda_2$", rotation=90)
    
# Row 3 energy landscapes with shared color scale and single colorbar across row
energy_fields = [load_energy_field(eng_prefix + st + ".txt") for st in stages_4rows]
vmin4 = min(np.min(E) for (_, _, E) in energy_fields)
vmax4 = max(np.max(E) for (_, _, E) in energy_fields)

last_cs4 = None
for j in range(len(stages_4rows)-1):
    (X, Y, E) = energy_fields[j]
    ax = axs4[2, j]
    last_cs4 = plot_energy_field(ax, X, Y, E, vmin=vmin4, vmax=vmax4, show_pattern_labels=True)
    # Overlay trajectories for consistency
    stage = stages_4rows[j]
    if not j>=len(stages_4rows) and j!=0:
        data_traj = np.loadtxt(trajectory_file+stages_4rows[j+1]+".data")
        plot_trajectory(ax, patterns, data_traj)
    ax.set_xticks([0, 0.5, 1])
    if j == 0:
        ax.set_ylabel(r"$\lambda_2$", rotation=90)
    ax.set_xlabel(r"$\lambda_1$")

if last_cs4 is not None:
    cbar = fig4.colorbar(last_cs4, cax=cax4)
    cbar.set_label('Energy')

# Consistent limits and aspect across all subplots
for r in range(3):
    for c in range(len(stages_4rows)):
        ax = axs4[r, c]
        if ax.has_data():
            ax.set_xlim(-0.1, 1.1)
            ax.set_ylim(-0.1, 1.1)
            ax.set_aspect('equal')

# plt.savefig('plots/Fig_pattern_vector_field_4rows_stream_excit_inhib_energy.png', dpi=300, bbox_inches='tight')
# plt.show()

# if __name__ == "__main__":
#     print("Figure with 4 rows (classic, energy, excit-only, inhib-only) created and saved.")
# # %%

# %%
