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
    'axes.labelsize': 35,
    'axes.titlesize': 30,
    'xtick.labelsize': 30,
    'ytick.labelsize': 30,
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
        ax.plot(0, 0, 'o', markersize=8, c="red")
        ax.plot(1, 0, 'o', markersize=6, c="red")
        ax.plot(0, 1, 'o', markersize=6, c="red")
        # Match 3plots labeling style
        ax.text(0.92, +0.07, r" $\mathbf{2}$", c="red", fontsize=35, fontfamily="monospace")
        ax.text(+0.08, 0.95, r" $\mathbf{1}$", c="red", fontsize=35, fontfamily="monospace")


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
    # Use consistent levels when vmin/vmax are provided to keep colorbar full
    if (vmin is not None) and (vmax is not None):
        levels = np.linspace(vmin, vmax, 60)
        line_levels = np.linspace(vmin, vmax, 8)
        cs = ax.contourf(X, Y, E, levels=levels, cmap='viridis')
        lines = ax.contour(X, Y, E, levels=line_levels, colors='white', alpha=0.5, linewidths=0.5)
    else:
        cs = ax.contourf(X, Y, E, 60, cmap='viridis')
        lines = ax.contour(X, Y, E, 8, colors='white', alpha=0.5, linewidths=0.5)
    ax.clabel(lines, inline=True, fontsize=8, fmt='%.1f')

    if show_pattern_labels:
        ax.plot(0, 0, 'o', markersize=8, c="red")
        ax.plot(1, 0, 'o', markersize=6, c="red")
        ax.plot(0, 1, 'o', markersize=6, c="red")
        # Match 3plots labeling style
        ax.text(0.92, +0.07, r" $\mathbf{2}$", c="red", fontsize=35, fontfamily="monospace")
        ax.text(+0.08, 0.95, r" $\mathbf{1}$", c="red", fontsize=35, fontfamily="monospace")

    return cs
#%%
# Files and configuration
folder = "../../data/all_data_splited/trained_networks_fast/Fig_vector_fields_patterns_same_distances/sim_nb_0/"
vec_prefix = folder + "vector_field_two_patterns_"
eng_prefix = folder + "energy_field_two_patterns_"
stages = ["pre_train", "post_train", "post_inhib"]

patterns_file = folder + "patterns.data"
first_trajectory_file = folder + "results_evolution_iter_1.data"
second_trajectory_file = folder + "results_evolution_iter_2.data"

# Load shared data
patterns = np.loadtxt(patterns_file)
first_trajectory = np.loadtxt(first_trajectory_file)
second_trajectory = np.loadtxt(second_trajectory_file)
#%%
# 4-row plot: classic stream (row 1), energy (row 2), excit-only stream (row 3), inhib-only stream (row 4)
import os

excit_prefix = folder + "vector_field_two_patterns_excit_only_"
inhib_prefix = folder + "vector_field_two_patterns_inhib_only_"
# Use the full set of stages present for the new files
stages_4rows = ["pre_train", "post_train", "iter_1"]
stage_titles = {
    "pre_train": "Pre-training",
    "post_train": "Post-training",
    "post_inhib": "Post-inhibition",
}

# Prepare figure (GridSpec): 4 rows x 3 columns of plots + 1 dedicated colorbar column
fig4 = plt.figure(figsize=(17, 22))
gs4 = fig4.add_gridspec(nrows=4, ncols=4, width_ratios=[1, 1, 1, 0.05], wspace=0.05, hspace=0.18)
axs4 = np.empty((3, 3), dtype=object)
for i in range(3):
    axs4[0, i] = fig4.add_subplot(gs4[0, i])
    axs4[1, i] = fig4.add_subplot(gs4[1, i])
    axs4[2, i] = fig4.add_subplot(gs4[2, i])
    # axs4[3, i] = fig4.add_subplot(gs4[3, i])
# Right-side labels for all 3 rows (no colorbar at right)
right_labels_axes = [fig4.add_subplot(gs4[i, 3]) for i in range(3)]
right_labels = [r"$\mathbf{W}+\mathbf{A}$", r"$\mathbf{A}$", r"$\mathbf{W}+\mathbf{A}$"]
for ax_lab, txt in zip(right_labels_axes, right_labels):
    ax_lab.set_axis_off()
    ax_lab.text(1.5, 0.5, txt, ha='center', va='center')

sns.despine(left=True, bottom=True, top=True, right=True)

# Row 1: classic stream plots
for j, stage in enumerate(stages_4rows):
    ax = axs4[0, j]
    display_labels = (stage != "pre_train")
    vec_file = vec_prefix + stage + ".txt"
    if not os.path.exists(vec_file):
        ax.text(0.5, 0.5, f"Missing: {os.path.basename(vec_file)}", ha='center', va='center')
        ax.set_axis_off()
        continue
    plot_stream_field(ax, vec_file, display_labels)
    if stage == "post_train":
        plot_trajectory(ax, patterns, first_trajectory)
    elif stage == "iter_1":
        plot_trajectory(ax, patterns, second_trajectory)
    ax.set_title(stage_titles.get(stage, stage))
    ax.set_xticks([0, 0.5, 1])
    if j == 0:
        ax.set_ylabel(r"$\lambda_2$", rotation=90)

# # Row 2: excit-only stream plots
# for j, stage in enumerate(stages_4rows):
#     ax = axs4[1, j]
#     display_labels = (stage != "pre_train")
#     vec_file = excit_prefix + stage + ".txt"
#     if not os.path.exists(vec_file):
#         ax.text(0.5, 0.5, f"Missing: {os.path.basename(vec_file)}", ha='center', va='center')
#         ax.set_axis_off()
#         continue
#     plot_stream_field(ax, vec_file, display_labels)
#     ax.set_xticks([0, 0.5, 1])
#     if j == 0:
#         ax.set_ylabel(r"$\lambda_2$", rotation=90)

# Row 3: inhib-only stream plots
for j, stage in enumerate(stages_4rows):
    ax = axs4[1, j]
    display_labels = (stage != "pre_train")
    vec_file = inhib_prefix + stage + ".txt"
    if not os.path.exists(vec_file):
        ax.text(0.5, 0.5, f"Missing: {os.path.basename(vec_file)}", ha='center', va='center')
        ax.set_axis_off()
        continue
    # Enhance weak inhibitory fields to avoid blank-looking plots
    plot_stream_field(ax, vec_file, display_labels, enhance_weak=True, weak_threshold=1e-6)
    ax.set_xticks([0, 0.5, 1])
    if j == 0:
        ax.set_ylabel(r"$\lambda_2$", rotation=90)
    
# Row 3 energy landscapes with shared color scale and single colorbar across row
# Remove normalization: use original energy values with a shared vmin/vmax
energy_fields_4 = [load_energy_field(eng_prefix + st + ".txt") for st in stages_4rows]
vmin4 = min(np.min(E) for (_, _, E) in energy_fields_4)
vmax4 = max(np.max(E) for (_, _, E) in energy_fields_4)

last_cs4 = None
for j, (X, Y, E) in enumerate(energy_fields_4):
    ax = axs4[2, j]
    last_cs4 = plot_energy_field(ax, X, Y, E, vmin=vmin4, vmax=vmax4, show_pattern_labels=True)
    # Overlay trajectories for consistency
    stage = stages_4rows[j]
    if stage == "post_train":
        plot_trajectory(ax, patterns, first_trajectory)
    elif stage == "iter_1":
        plot_trajectory(ax, patterns, second_trajectory)
    ax.set_xticks([0, 0.5, 1])
    if j == 0:
        ax.set_ylabel(r"$\lambda_2$", rotation=90)
    ax.set_xlabel(r"$\lambda_1$")

if last_cs4 is not None:
    # Place a smaller colorbar to the left of the energy row to avoid overlapping right labels
    row_ax = axs4[2, 0]
    pos = row_ax.get_position()
    cbar_width = 0.015
    cbar_h = pos.height * 0.6
    cbar_y0 = pos.y0 + (pos.height - cbar_h) / 2
    left_offset = 0.09
    cbar_x0 = max(0.01, pos.x0 - (left_offset + cbar_width))
    cax_small = fig4.add_axes([cbar_x0, cbar_y0, cbar_width, cbar_h])
    cbar = fig4.colorbar(last_cs4, cax=cax_small)
    cbar.set_label('Energy')
    # Four readable integer ticks spanning [vmin, vmax]
    try:
        ticks4 = np.linspace(vmin4, vmax4, 4)
        cbar.set_ticks(ticks4)
        cbar.set_ticklabels([f"{int(round(t))}" for t in ticks4])
    except Exception:
        pass
    cbar.ax.yaxis.set_ticks_position('left')
    cbar.ax.yaxis.set_label_position('left')
    cbar.ax.tick_params(labelleft=True, labelright=False)

# Consistent limits and aspect across all subplots
for r in range(3):
    for c in range(len(stages_4rows)):
        ax = axs4[r, c]
        if ax.has_data():
            ax.set_xlim(-0.1, 1.1)
            ax.set_ylim(-0.1, 1.1)
            ax.set_aspect('equal')

# Show x tick labels only on the last row and y tick labels only on the first column
nrows, ncols = axs4.shape
for r in range(nrows):
    for c in range(ncols):
        ax = axs4[r, c]
        if r != nrows - 1:
            ax.tick_params(labelbottom=False)
        if c != 0:
            ax.tick_params(labelleft=False)

plt.savefig('plots/Fig_pattern_vector_field_4rows_stream_excit_inhib_energy.png', dpi=300, bbox_inches='tight')
plt.show()

if __name__ == "__main__":
    print("Figure with 4 rows (classic, energy, excit-only, inhib-only) created and saved.")

#%%
# Extended figure: add inhibitory-only energy row as third row (label 'A' on the right)

# Prepare extended figure: 4 rows x 4 columns (3 plot columns + right label/cbar column)
fig_ext = plt.figure(figsize=(20, 26))
gs_ext = fig_ext.add_gridspec(nrows=4, ncols=4, width_ratios=[1, 1, 1, 0.05], wspace=0.05, hspace=0.18)

axs_ext = np.empty((4, 3), dtype=object)
for i in range(3):
    axs_ext[0, i] = fig_ext.add_subplot(gs_ext[0, i])
    axs_ext[1, i] = fig_ext.add_subplot(gs_ext[1, i])
    axs_ext[2, i] = fig_ext.add_subplot(gs_ext[2, i])
    axs_ext[3, i] = fig_ext.add_subplot(gs_ext[3, i])

# Right-side labels for all 4 rows (stream W+A, stream A, energy A, energy W+A)
right_axes_ext = [fig_ext.add_subplot(gs_ext[i, 3]) for i in range(4)]
right_labels_ext = [r"$\mathbf{W}+\mathbf{A}$", r"$\mathbf{A}$", r"$\mathbf{A}$", r"$\mathbf{W}+\mathbf{A}$"]
for ax_lab, txt in zip(right_axes_ext, right_labels_ext):
    ax_lab.set_axis_off()
    ax_lab.text(1.5, 0.5, txt, ha='center', va='center')

sns.despine(left=True, bottom=True, top=True, right=True)

# Row 1: classic stream plots (same as before)
for j, stage in enumerate(stages_4rows):
    ax = axs_ext[0, j]
    display_labels = (stage != "pre_train")
    vec_file = vec_prefix + stage + ".txt"
    if not os.path.exists(vec_file):
        ax.text(0.5, 0.5, f"Missing: {os.path.basename(vec_file)}", ha='center', va='center')
        ax.set_axis_off()
        continue
    plot_stream_field(ax, vec_file, display_labels)
    if stage == "post_train":
        plot_trajectory(ax, patterns, first_trajectory)
    elif stage == "iter_1":
        plot_trajectory(ax, patterns, second_trajectory)
    ax.set_title(stage_titles.get(stage, stage))
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    if j == 0:
        ax.set_ylabel(r"$\lambda_2$", rotation=90)

# Row 2: inhib-only stream plots (same as before)
for j, stage in enumerate(stages_4rows):
    ax = axs_ext[1, j]
    display_labels = (stage != "pre_train")
    vec_file = inhib_prefix + stage + ".txt"
    if not os.path.exists(vec_file):
        ax.text(0.5, 0.5, f"Missing: {os.path.basename(vec_file)}", ha='center', va='center')
        ax.set_axis_off()
        continue
    plot_stream_field(ax, vec_file, display_labels, enhance_weak=True, weak_threshold=1e-6)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    if j == 0:
        ax.set_ylabel(r"$\lambda_2$", rotation=90)

# Row 3: inhibitory-only energy landscapes (A)
inh_eng_prefix = folder + "energy_field_two_patterns_inhib_only_"
inh_energy_fields = [load_energy_field(inh_eng_prefix + st + ".txt") for st in stages_4rows]

# Row 4: combined energy landscapes (W+A)
comb_energy_fields = [load_energy_field(eng_prefix + st + ".txt") for st in stages_4rows]

# Remove normalization: compute shared vmin/vmax across both energy rows
all_E_ext = [E for (_, _, E) in inh_energy_fields + comb_energy_fields]
vmin_ext = min(np.min(E) for E in all_E_ext)
vmax_ext = max(np.max(E) for E in all_E_ext)

last_cs_ext = None
for j, (X, Y, E) in enumerate(inh_energy_fields):
    ax = axs_ext[2, j]
    last_cs_ext = plot_energy_field(ax, X, Y, E, vmin=vmin_ext, vmax=vmax_ext, show_pattern_labels=True)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    if j == 0:
        ax.set_ylabel(r"$\lambda_2$", rotation=90)

for j, (X, Y, E) in enumerate(comb_energy_fields):
    ax = axs_ext[3, j]
    last_cs_ext = plot_energy_field(ax, X, Y, E, vmin=vmin_ext, vmax=vmax_ext, show_pattern_labels=True)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    if j == 0:
        ax.set_ylabel(r"$\lambda_2$", rotation=90)
    ax.set_xlabel(r"$\lambda_1$")

# Small shared colorbar to the left of the two energy rows (original scale)
if last_cs_ext is not None:
    pos_r3 = axs_ext[2, 0].get_position()
    pos_r4 = axs_ext[3, 0].get_position()
    total_h = pos_r3.y1 - pos_r4.y0
    cbar_h = total_h * 0.55
    cbar_y0 = pos_r4.y0 + (total_h - cbar_h) / 2
    cbar_width = 0.015
    left_offset = 0.09
    cbar_x0 = max(0.01, pos_r3.x0 - (left_offset + cbar_width))
    cax_ext = fig_ext.add_axes([cbar_x0, cbar_y0, cbar_width, cbar_h])
    cbar_ext = fig_ext.colorbar(last_cs_ext, cax=cax_ext)
    cbar_ext.set_label('Energy')
    # Four readable integer ticks spanning [vmin_ext, vmax_ext]
    try:
        ticks_ext = np.linspace(vmin_ext, vmax_ext, 4)
        cbar_ext.set_ticks(ticks_ext)
        cbar_ext.set_ticklabels([f"{int(round(t))}" for t in ticks_ext])
    except Exception:
        pass
    cbar_ext.ax.yaxis.set_ticks_position('left')
    cbar_ext.ax.yaxis.set_label_position('left')
    cbar_ext.ax.tick_params(labelleft=True, labelright=False)

# Apply shared tick label visibility: only bottom row shows x labels; only first column shows y labels
rows_ext, cols_ext = axs_ext.shape
for r in range(rows_ext):
    for c in range(cols_ext):
        ax = axs_ext[r, c]
        if r != rows_ext - 1:
            ax.tick_params(labelbottom=False)
        if c != 0:
            ax.tick_params(labelleft=False)

for r in range(4):
    for c in range(len(stages_4rows)):
        ax = axs_ext[r, c]
        if ax.has_data():
            ax.set_xlim(-0.1, 1.1)
            ax.set_ylim(-0.1, 1.1)
            ax.set_aspect('equal')

plt.savefig('plots/Fig_pattern_vector_field_with_inhib_energy_row.png', dpi=300, bbox_inches='tight')

# %%
