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

def plot_stream_field(ax, filename, show_pattern_labels=True, enhance_weak=False, weak_threshold=1e-6):
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
    ax.streamplot(xi, yi, use_DX, use_DY, density=1, color='tab:blue', arrowsize=3, linewidth=2.5)

    if show_pattern_labels:
        ax.plot(0, 0, 'o', markersize=13, c="red")
        ax.plot(1, 0, 'o', markersize=7, c="red")
        ax.plot(0, 1, 'o', markersize=7, c="red")

        # Use the same label positions/fonts as 3plots
        ax.text(0.92, +0.07, r" $\mathbf{2}$", c="red", fontsize=39, fontfamily="monospace")
        ax.text(+0.08, 0.95, r" $\mathbf{1}$", c="red", fontsize=39, fontfamily="monospace")


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
    # Use consistent levels if a range is provided to avoid partial/empty colorbars
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
        ax.plot(0, 0, 'o', markersize=13, c="red")
        ax.plot(1, 0, 'o', markersize=7, c="red")
        ax.plot(0, 1, 'o', markersize=7, c="red")

        # Use the same label positions/fonts as 3plots
        ax.text(0.92, +0.07, r" $\mathbf{2}$", c="red", fontsize=39, fontfamily="monospace")
        ax.text(+0.08, 0.95, r" $\mathbf{1}$", c="red", fontsize=39, fontfamily="monospace")

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
fig2 = plt.figure(figsize=(29, 14))
gs2 = fig2.add_gridspec(nrows=2, ncols=5, width_ratios=[1, 1, 1, 1, 0.05], wspace=0.06, hspace=0.18)
axs2 = np.empty((2, 4), dtype=object)
for i in range(4):
    axs2[0, i] = fig2.add_subplot(gs2[0, i])
    axs2[1, i] = fig2.add_subplot(gs2[1, i])

# Right-side labels for both rows
right_axes2 = [fig2.add_subplot(gs2[i, 4]) for i in range(2)]
right_labels2 = ["$\\mathbf{W}$\n$+\\mathbf{B}$\n$+\\mathbf{A}$", r"$\mathbf{A}$"]
for ax_lab, txt in zip(right_axes2, right_labels2):
    ax_lab.set_axis_off()
    ax_lab.text(0.5, 0.5, txt, ha='center', va='center')

sns.despine(left=True, bottom=True, top=True, right=True)

# Load energy fields for both rows (original scale, no normalization)
full_fields = []
inh_fields = []
for st in cols:
    full_fields.append(load_energy_field(eng_prefix + st + ".txt"))
    inh_fields.append(load_energy_field(inh_eng_prefix + st + ".txt"))

# Compute shared vmin/vmax across both rows
all_E = [E for (_, _, E) in full_fields] + [E for (_, _, E) in inh_fields]
vmin_all = min(np.min(E) for E in all_E)
vmax_all = max(np.max(E) for E in all_E)

# Mapping for trajectory overlay: show next-iteration trajectory on the previous column
traj_map = {
    "post_train": "iter_1",
    "iter_1": "iter_2",
    "iter_2": "iter_3",
    "iter_3": "iter_4",
}

last_cs = None
# Row 1: W+B+A energy (original scale)
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

# Row 2: A-only energy (original scale)
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
    # Four readable integer ticks spanning [vmin_all, vmax_all]
    try:
        ticks2 = np.linspace(vmin_all, vmax_all, 4)
        cbar.set_ticks(ticks2)
        cbar.set_ticklabels([f"{int(round(t))}" for t in ticks2])
    except Exception:
        pass
    cbar.ax.yaxis.set_ticks_position('left')
    cbar.ax.yaxis.set_label_position('left')
    cbar.ax.tick_params(labelleft=True, labelright=False)

# Consistent limits and aspect
for r in range(2):
    for c in range(len(cols)):
        ax = axs2[r, c]
        if ax.has_data():
            ax.set_xlim(-0.2, 1.1)
            ax.set_ylim(-0.2, 1.1)
            ax.set_aspect('equal')

# Only bottom row shows x tick labels; only first column shows y tick labels (2-row fig)
rows2, cols2 = axs2.shape
for r in range(rows2):
    for c in range(cols2):
        ax = axs2[r, c]
        if r != rows2 - 1:
            ax.tick_params(labelbottom=False)
        if c != 0:
            ax.tick_params(labelleft=False)
plt.savefig('plots/Fig_pattern_vector_field_distant_many_iter_2rows_energy.png', dpi=300, bbox_inches='tight')

#%%
# Three-row figure: Row1 streams (W+B+A), Row2 energy (W+B+A), Row3 energy (A)
# Stream and energy columns (no pre_train)
cols_stream = ["post_train", "iter_1", "iter_2", "iter_3"]
cols_energy = ["post_train", "iter_1", "iter_2", "iter_3"]

fig3 = plt.figure(figsize=(28, 18))
gs3 = fig3.add_gridspec(nrows=3, ncols=5, width_ratios=[1, 1, 1, 1, 0.05], wspace=0.06, hspace=0.18)
axs3 = np.empty((3, 4), dtype=object)
for i in range(4):
    axs3[0, i] = fig3.add_subplot(gs3[0, i])
    axs3[1, i] = fig3.add_subplot(gs3[1, i])
    axs3[2, i] = fig3.add_subplot(gs3[2, i])

# Right labels
right_axes3 = [fig3.add_subplot(gs3[i, 4]) for i in range(3)]
right_labels3 = ["$\\mathbf{W}+\\mathbf{B}$\n$+\\mathbf{A}$", "$\\mathbf{W}+\\mathbf{B}$\n$+\\mathbf{A}$", r"$\mathbf{A}$"]
for ax_lab, txt in zip(right_axes3, right_labels3):
    ax_lab.set_axis_off()
    ax_lab.text(0.5, 0.5, txt, ha='center', va='center')
    
sns.despine(left=True, bottom=True, top=True, right=True)
# Row 1 streams
for j, st in enumerate(cols_stream):
    ax = axs3[0, j]
    vec_file = vec_prefix + st + ".txt"
    if not os.path.exists(vec_file):
        ax.text(0.5, 0.5, f"Missing: {os.path.basename(vec_file)}", ha='center', va='center')
        ax.set_axis_off()
        continue
    display_labels = (st != "pre_train")
    plot_stream_field(ax, vec_file, display_labels)
    ax.set_xticks([0, 0.5, 1])
    nxt = traj_map.get(st)
    if nxt is not None:
        tpath = trajectory_file + nxt + ".data"
        if os.path.exists(tpath):
            plot_trajectory(ax, patterns, np.loadtxt(tpath))
    if j == 0:
        ax.set_ylabel(r"$\lambda_2$", rotation=90)
    ax.set_title(st.replace('_', ' '))

# Load energies (original scale) and compute shared vmin/vmax
full_fields3 = [load_energy_field(eng_prefix + st + ".txt") for st in cols_energy]
inh_fields3  = [load_energy_field(inh_eng_prefix + st + ".txt") for st in cols_energy]
all_E3 = [E for (_, _, E) in full_fields3] + [E for (_, _, E) in inh_fields3]
vmin3 = min(np.min(E) for E in all_E3)
vmax3 = max(np.max(E) for E in all_E3)

# Trajectory overlay mapping
traj_map = {
    "post_train": "iter_1",
    "iter_1": "iter_2",
    "iter_2": "iter_3",
    "iter_3": "iter_4",
}

last_cs3 = None
# Row 2 energies (W+B+A)
for j, st in enumerate(cols_energy):
    X, Y, E = full_fields3[j]
    ax = axs3[1, j]
    last_cs3 = plot_energy_field(ax, X, Y, E, vmin=vmin3, vmax=vmax3, show_pattern_labels=True)
    nxt = traj_map.get(st)
    if nxt is not None:
        tpath = trajectory_file + nxt + ".data"
        if os.path.exists(tpath):
            plot_trajectory(ax, patterns, np.loadtxt(tpath))
    ax.set_xticks([0, 0.5, 1])
    if j == 0:
        ax.set_ylabel(r"$\lambda_2$", rotation=90)

# Row 3 energies (A-only)
for j, st in enumerate(cols_energy):
    X, Y, E = inh_fields3[j]
    ax = axs3[2, j]
    cs = plot_energy_field(ax, X, Y, E, vmin=vmin3, vmax=vmax3, show_pattern_labels=True)
    nxt = traj_map.get(st)
    if nxt is not None:
        tpath = trajectory_file + nxt + ".data"
        if os.path.exists(tpath):
            plot_trajectory(ax, patterns, np.loadtxt(tpath))
    ax.set_xticks([0, 0.5, 1])
    if j == 0:
        ax.set_ylabel(r"$\lambda_2$", rotation=90)
    ax.set_xlabel(r"$\lambda_1$")

# Colorbar to the left of energy rows only
if last_cs3 is not None:
    pos_r2 = axs3[1, 0].get_position()
    pos_r3 = axs3[2, 0].get_position()
    total_h = pos_r2.y1 - pos_r3.y0
    cbar_h = total_h * 0.5
    cbar_y0 = pos_r3.y0 + (total_h - cbar_h) / 2
    cbar_width = 0.015
    left_offset = 0.04
    cbar_x0 = max(0.01, pos_r2.x0 - (left_offset + cbar_width))
    cax3 = fig3.add_axes([cbar_x0, cbar_y0, cbar_width, cbar_h])
    cbar3 = fig3.colorbar(last_cs3, cax=cax3)
    cbar3.set_label('Energy')
    # Four readable integer ticks spanning [vmin3, vmax3]
    try:
        ticks3 = np.linspace(vmin3, vmax3, 4)
        cbar3.set_ticks(ticks3)
        cbar3.set_ticklabels([f"{int(round(t))}" for t in ticks3])
    except Exception:
        pass
    cbar3.ax.yaxis.set_ticks_position('left')
    cbar3.ax.yaxis.set_label_position('left')
    cbar3.ax.tick_params(labelleft=True, labelright=False)

for r in range(3):
    for c in range(4):
        ax = axs3[r, c]
        if ax is not None and ax.has_data():
            ax.set_xlim(-0.1, 1.1)
            ax.set_ylim(-0.1, 1.1)
            ax.set_aspect('equal')

# Only bottom row shows x tick labels; only first column shows y tick labels (3-row fig)
rows3, cols3 = axs3.shape
for r in range(rows3):
    for c in range(cols3):
        ax = axs3[r, c]
        if ax is None:
            continue
        if r != rows3 - 1:
            ax.tick_params(labelbottom=False)
        if c != 0:
            ax.tick_params(labelleft=False)

plt.savefig('plots/Fig_pattern_vector_field_distant_many_iter_3rows_stream_and_energy.png', dpi=300, bbox_inches='tight')

# %%
