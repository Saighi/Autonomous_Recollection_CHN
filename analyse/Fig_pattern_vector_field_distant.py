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
eng_prefix = folder + "energy_field_two_patterns_"            # W+B total energy
bias_eng_prefix = folder + "energy_field_two_patterns_bias_only_"  # B-only energy
stages = ["pre_train", "post_train", "post_inhib"]

patterns_file = folder + "patterns.data"
trajectory_file = folder + "results_evolution_"

# Load shared data
patterns = np.loadtxt(patterns_file)
#%%
# 4-row plot: classic stream (row 1), energy (row 2), excit-only stream (row 3), inhib-only stream (row 4)
import os

excit_prefix = folder + "vector_field_two_patterns_excit_only_"  # unused here
inhib_prefix = folder + "vector_field_two_patterns_inhib_only_"  # unused here
bias_prefix = folder + "vector_field_two_patterns_bias_only_"
# Only two columns: pre and post training
stages_2cols = ["pre_train", "post_train"]
stage_titles = {
    "pre_train": "Pre-training",
    "post_train": "Post-training",
    "post_inhib": "Post-inhibition",
}

# Prepare figure (GridSpec): 3 rows x 2 columns of plots + 1 right-label column
fig4 = plt.figure(figsize=(16, 18))
gs4 = fig4.add_gridspec(nrows=3, ncols=3, width_ratios=[1, 1, 0.05], wspace=0.08, hspace=0.18)
axs4 = np.empty((3, 2), dtype=object)
for i in range(2):
    axs4[0, i] = fig4.add_subplot(gs4[0, i])
    axs4[1, i] = fig4.add_subplot(gs4[1, i])
    axs4[2, i] = fig4.add_subplot(gs4[2, i])

# Right-side labels for all rows (no colorbar at right)
right_labels_axes = [fig4.add_subplot(gs4[i, 2]) for i in range(3)]
right_labels = [r"$\mathbf{W} + \mathbf{B}$", r"$\mathbf{W} + \mathbf{B}$", r"$\mathbf{B}$"]
for ax_lab, txt in zip(right_labels_axes, right_labels):
    ax_lab.set_axis_off()
    ax_lab.text(0.5, 0.5, txt, ha='center', va='center')

sns.despine(left=True, bottom=True, top=True, right=True)

# Row 1: classic stream plots (W+B)
for j, stage in enumerate(stages_2cols):
    ax = axs4[0, j]
    display_labels = (stage != "pre_train")
    vec_file = vec_prefix + stage + ".txt"
    if not os.path.exists(vec_file):
        ax.text(0.5, 0.5, f"Missing: {os.path.basename(vec_file)}", ha='center', va='center')
        ax.set_axis_off()
        continue
    plot_stream_field(ax, vec_file, display_labels)
    # Overlay trajectory for post_train if available
    if stage == "post_train":
        traj_path = trajectory_file + stage + ".data"
        if os.path.exists(traj_path):
            data_traj = np.loadtxt(traj_path)
            plot_trajectory(ax, patterns, data_traj)
    ax.set_title(stage_titles.get(stage, stage))
    ax.set_xticks([0, 0.5, 1])
    if j == 0:
        ax.set_ylabel(r"$\lambda_2$", rotation=90)

"""
Row 2: W+B energy landscapes
"""


"""
Row 3: W+B energy landscapes
"""
# Preload energy sets for displayed columns and compute a shared vmin/vmax across both rows
energy_fields_wb = [load_energy_field(eng_prefix + st + ".txt") for st in stages_2cols]
energy_fields_b = [load_energy_field(bias_eng_prefix + st + ".txt") for st in stages_2cols]
vmin_global = min(min(np.min(E) for (_, _, E) in energy_fields_wb),
                  min(np.min(E) for (_, _, E) in energy_fields_b))
vmax_global = max(max(np.max(E) for (_, _, E) in energy_fields_wb),
                  max(np.max(E) for (_, _, E) in energy_fields_b))

last_cs4 = None
for j, (X, Y, E) in enumerate(energy_fields_wb):
    ax = axs4[1, j]
    last_cs4 = plot_energy_field(ax, X, Y, E, vmin=vmin_global, vmax=vmax_global, show_pattern_labels=True)
    ax.set_xticks([0, 0.5, 1])
    if j == 0:
        ax.set_ylabel(r"$\lambda_2$", rotation=90)
    ax.set_xlabel(r"$\lambda_1$")

"""
Row 3: B-only energy landscapes (same global normalization)
"""
last_cs4 = None
for j, (X, Y, E) in enumerate(energy_fields_b):
    ax = axs4[2, j]
    last_cs4 = plot_energy_field(ax, X, Y, E, vmin=vmin_global, vmax=vmax_global, show_pattern_labels=True)
    ax.set_xticks([0, 0.5, 1])
    if j == 0:
        ax.set_ylabel(r"$\lambda_2$", rotation=90)
    ax.set_xlabel(r"$\lambda_1$")

# Add a single shared vertical colorbar to the left, spanning rows 2 and 3
if last_cs4 is not None:
    # Geometry of the leftmost axes in rows 2 and 3
    pos_r2 = axs4[1, 0].get_position()
    pos_r3 = axs4[2, 0].get_position()

    # Vertical span: cover most of rows 2 and 3, but not all
    total_h = pos_r2.y1 - pos_r3.y0
    shrink = 0.75  # show 75% of the combined height
    cbar_h = total_h * shrink
    cbar_y0 = pos_r3.y0 + (total_h - cbar_h) / 2  # vertically centered

    # Horizontal placement: shift left to avoid overlapping y labels
    cbar_width = 0.02
    left_offset = 0.06
    cbar_x0 = max(0.01, pos_r2.x0 - (left_offset + cbar_width))

    cax_left_both = fig4.add_axes([cbar_x0, cbar_y0, cbar_width, cbar_h])
    cbar = fig4.colorbar(last_cs4, cax=cax_left_both)  # vertical
    cbar.set_label('Energy')
    # Place ticks and label on the left side of the colorbar
    cbar.ax.yaxis.set_ticks_position('left')
    cbar.ax.yaxis.set_label_position('left')
    cbar.ax.tick_params(labelleft=True, labelright=False)

# Consistent limits and aspect across all subplots
for r in range(3):
    for c in range(len(stages_2cols)):
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

#%%
# Second figure: same base but with an extra energy row for W-only

 

fig6 = plt.figure(figsize=(16, 28))
gs6 = fig6.add_gridspec(nrows=4, ncols=3, width_ratios=[1, 1, 0.05], wspace=0.08, hspace=0.18)
axs6 = np.empty((4, 2), dtype=object)
for i in range(2):
    axs6[0, i] = fig6.add_subplot(gs6[0, i])
    axs6[1, i] = fig6.add_subplot(gs6[1, i])
    axs6[2, i] = fig6.add_subplot(gs6[2, i])
    axs6[3, i] = fig6.add_subplot(gs6[3, i])

right_axes6 = [fig6.add_subplot(gs6[i, 2]) for i in range(4)]
right_labels6 = [r"$\mathbf{W} + \mathbf{B}$", r"$\mathbf{W} + \mathbf{B}$", r"$\mathbf{B}$", r"$\mathbf{W}$"]
for ax_lab, txt in zip(right_axes6, right_labels6):
    ax_lab.set_axis_off()
    ax_lab.text(0.5, 0.5, txt, ha='center', va='center')

# Row 1: W+B stream
for j, stage in enumerate(stages_2cols):
    ax = axs6[0, j]
    display_labels = (stage != "pre_train")
    vec_file = vec_prefix + stage + ".txt"
    if not os.path.exists(vec_file):
        ax.text(0.5, 0.5, f"Missing: {os.path.basename(vec_file)}", ha='center', va='center')
        ax.set_axis_off()
        continue
    plot_stream_field(ax, vec_file, display_labels)
    if stage == "post_train":
        traj_path = trajectory_file + stage + ".data"
        if os.path.exists(traj_path):
            data_traj = np.loadtxt(traj_path)
            plot_trajectory(ax, patterns, data_traj)
    ax.set_title(stage_titles.get(stage, stage))
    ax.set_xticks([0, 0.5, 1])
    if j == 0:
        ax.set_ylabel(r"$\lambda_2$", rotation=90)

# Gather energy fields: W+B, B, W-only
wb_fields = [load_energy_field(eng_prefix + st + ".txt") for st in stages_2cols]
b_fields = [load_energy_field(bias_eng_prefix + st + ".txt") for st in stages_2cols]
w_fields = []
for st in stages_2cols:
    path = folder + f"energy_field_two_patterns_weights_only_{st}.txt"
    if os.path.exists(path):
        w_fields.append(load_energy_field(path))
    else:
        w_fields.append((None, None, None))

# Compute normalization across all available energy fields (rows 2–4)
vals6 = []
for arrs in (wb_fields, b_fields, w_fields):
    for _, _, E in arrs:
        if E is not None:
            vals6.append(E)
if vals6:
    vmin6 = min(np.min(E) for E in vals6)
    vmax6 = max(np.max(E) for E in vals6)
else:
    vmin6 = vmax6 = None

# Plot W+B energy (row 2)
last_cs6 = None
for j, (X, Y, E) in enumerate(wb_fields):
    ax = axs6[1, j]
    last_cs6 = plot_energy_field(ax, X, Y, E, vmin=vmin6, vmax=vmax6, show_pattern_labels=True)
    ax.set_xticks([0, 0.5, 1])
    if j == 0:
        ax.set_ylabel(r"$\lambda_2$", rotation=90)
    ax.set_xlabel(r"$\lambda_1$")

# Plot B energy (row 3)
for j, (X, Y, E) in enumerate(b_fields):
    ax = axs6[2, j]
    last_cs6 = plot_energy_field(ax, X, Y, E, vmin=vmin6, vmax=vmax6, show_pattern_labels=True)
    ax.set_xticks([0, 0.5, 1])
    if j == 0:
        ax.set_ylabel(r"$\lambda_2$", rotation=90)
    ax.set_xlabel(r"$\lambda_1$")

# Plot W-only energy or placeholder (row 4)
for j, (X, Y, E) in enumerate(w_fields):
    ax = axs6[3, j]
    if E is None:
        ax.text(0.5, 0.5, "Missing: weights-only", ha='center', va='center')
        ax.set_axis_off()
        continue
    last_cs6 = plot_energy_field(ax, X, Y, E, vmin=vmin6, vmax=vmax6, show_pattern_labels=True)
    ax.set_xticks([0, 0.5, 1])
    if j == 0:
        ax.set_ylabel(r"$\lambda_2$", rotation=90)
    ax.set_xlabel(r"$\lambda_1$")

# Shared vertical colorbar, centered across the three energy rows (rows 2–4)
if last_cs6 is not None:
    pos2 = axs6[1, 0].get_position()
    pos3 = axs6[2, 0].get_position()
    pos4 = axs6[3, 0].get_position()
    span_y0 = pos4.y0
    span_y1 = pos2.y1
    total_h = span_y1 - span_y0
    shrink = 0.75
    cbar_h = total_h * shrink
    cbar_y0 = span_y0 + (total_h - cbar_h) / 2
    cbar_width = 0.02
    left_offset = 0.06
    cbar_x0 = max(0.01, pos2.x0 - (left_offset + cbar_width))
    cax6 = fig6.add_axes([cbar_x0, cbar_y0, cbar_width, cbar_h])
    cbar6 = fig6.colorbar(last_cs6, cax=cax6)
    cbar6.set_label('Energy')
    cbar6.ax.yaxis.set_ticks_position('left')
    cbar6.ax.yaxis.set_label_position('left')
    cbar6.ax.tick_params(labelleft=True, labelright=False)

for r in range(4):
    for c in range(len(stages_2cols)):
        ax = axs6[r, c]
        if ax.has_data():
            ax.set_xlim(-0.1, 1.1)
            ax.set_ylim(-0.1, 1.1)
            ax.set_aspect('equal')
