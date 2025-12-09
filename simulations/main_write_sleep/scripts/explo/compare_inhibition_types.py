"""
Compare recovery of 3 patterns in a 50-unit network with symmetric transfer
across three inhibition conditions:
1. No inhibition plasticity
2. Diagonal inhibition only (default)
3. Full matrix inhibition (new)

Use the NEW pattern generator (parent + redraw) with rho = 0.5.
Run this as a VS Code notebook with #%% cells.
"""

# %% Imports and configuration

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add scripts directory to path (parent.parent = scripts/)
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import (
    generate_patterns_new,
    setup_write_experiment,
    setup_sleep_experiment,
    run_cpp,
    build,
    list_simulations,
    load_trajectories,
    read_patterns,
    compute_correlations,
    DATA_DIR,
)

NETWORK_SIZE = 50
NUM_PATTERNS = 3
RHO = 0.5
SPARSITY = 0.5  # P(x_i = 0) for NEW generator
MAX_QUERIES = 5
NOISE_DYNAMICS = 0
STDDEV = 0.1

np.random.seed(0)

# Build binaries
build()


# %% Generate patterns with NEW generator

patterns = generate_patterns_new(
    k=NUM_PATTERNS,
    n=NETWORK_SIZE,
    sparsity=SPARSITY,
    rho=RHO,
)

print("Patterns (0/1):")
print(patterns.astype(int))
print("Ones per pattern:", patterns.sum(axis=1))


# %% Write phase: symmetric transfer

write_name = "compare_inhibition_sym"
write_config = setup_write_experiment(
    name=write_name,
    patterns=patterns,
    params={
        "leak": 1.0,
        "drive_target": 6.0,
        "learning_rate": 0.0001,
        "distance_noise_level": 0.0,
        "momentum_coef": 0.9,
        "max_iter": 100000,
        "symmetric_transfer": 1.0,  # Use symmetric transfer
    },
    native_pattern_generation=False,
)

print("\n[SYM] Running write phase...")
run_cpp("write", write_config)


# %% Sleep phase: NO inhibition plasticity

sleep_name_no_inhib = "compare_inhibition_no_plasticity"

sleep_config_no_inhib = setup_sleep_experiment(
    name=sleep_name_no_inhib,
    trained_networks_dir=DATA_DIR / "trained_networks" / write_name,
    params={
        "beta": 0.5,
        "delta": 0.01,
        "noise_dynamics": NOISE_DYNAMICS,
        "stddev_dynamics": STDDEV,
        "max_queries": MAX_QUERIES,
        "stop_on_spurious": 0,
        "stop_on_all_found": 0,
        "save_trajectories": 1,
        "use_inhibition_plasticity": 0,  # NO inhibition plasticity
    },
)

print("[NO INHIB] Running sleep phase...")
run_cpp("sleep", sleep_config_no_inhib)

sleep_results_dir_no_inhib = DATA_DIR / "sleep_results" / sleep_name_no_inhib
sims_no_inhib = list_simulations(sleep_results_dir_no_inhib)
sim_dir_no_inhib = sims_no_inhib[0]
cpp_patterns_no_inhib = read_patterns(sim_dir_no_inhib / "patterns.data").astype(float)
traj_no_inhib = load_trajectories(sim_dir_no_inhib)

print("Loaded NO INHIB sleep simulation from:", sim_dir_no_inhib)


# %% Sleep phase: DIAGONAL inhibition

sleep_name_diag = "compare_inhibition_diagonal"

sleep_config_diag = setup_sleep_experiment(
    name=sleep_name_diag,
    trained_networks_dir=DATA_DIR / "trained_networks" / write_name,
    params={
        "beta": 0.5,
        "delta": 0.01,
        "noise_dynamics": NOISE_DYNAMICS,
        "stddev_dynamics": STDDEV,
        "max_queries": MAX_QUERIES,
        "stop_on_spurious": 0,
        "stop_on_all_found": 0,
        "save_trajectories": 1,
        "use_inhibition_plasticity": 1,  # Enable inhibition plasticity
        "use_full_inhibition": 0,  # Diagonal only
    },
)

print("[DIAGONAL INHIB] Running sleep phase...")
run_cpp("sleep", sleep_config_diag)

sleep_results_dir_diag = DATA_DIR / "sleep_results" / sleep_name_diag
sims_diag = list_simulations(sleep_results_dir_diag)
sim_dir_diag = sims_diag[0]
cpp_patterns_diag = read_patterns(sim_dir_diag / "patterns.data").astype(float)
traj_diag = load_trajectories(sim_dir_diag)

print("Loaded DIAGONAL INHIB sleep simulation from:", sim_dir_diag)


# %% Sleep phase: FULL MATRIX inhibition

sleep_name_full = "compare_inhibition_full_matrix"

sleep_config_full = setup_sleep_experiment(
    name=sleep_name_full,
    trained_networks_dir=DATA_DIR / "trained_networks" / write_name,
    params={
        "beta": 0.1,
        "delta": 0.01,
        "noise_dynamics": NOISE_DYNAMICS,
        "stddev_dynamics": STDDEV,
        "max_queries": MAX_QUERIES,
        "stop_on_spurious": 0,
        "stop_on_all_found": 0,
        "save_trajectories": 1,
        "use_inhibition_plasticity": 1,  # Enable inhibition plasticity
        "use_full_inhibition": 1,  # Full matrix inhibition
    },
)

print("[FULL MATRIX INHIB] Running sleep phase...")
run_cpp("sleep", sleep_config_full)

sleep_results_dir_full = DATA_DIR / "sleep_results" / sleep_name_full
sims_full = list_simulations(sleep_results_dir_full)
sim_dir_full = sims_full[0]
cpp_patterns_full = read_patterns(sim_dir_full / "patterns.data").astype(float)
traj_full = load_trajectories(sim_dir_full)

print("Loaded FULL MATRIX INHIB sleep simulation from:", sim_dir_full)


# %% Compute correlations over time for all conditions

# Use centralized compute_correlations from utils with symmetric transfer
corr_no_inhib, lengths_no_inhib = compute_correlations(traj_no_inhib, cpp_patterns_no_inhib, symmetric_transfer=True)
corr_diag, lengths_diag = compute_correlations(traj_diag, cpp_patterns_diag, symmetric_transfer=True)
corr_full, lengths_full = compute_correlations(traj_full, cpp_patterns_full, symmetric_transfer=True)


# %% Plot comparison - No inhibition plasticity

plots_dir = Path(__file__).parent.parent / "plots"  # scripts/plots/
plots_dir.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

for p in range(corr_no_inhib.shape[1]):
    ax.plot(corr_no_inhib[:, p], label=f"Pattern {p+1}")

cumulative = 0
for L in lengths_no_inhib[:-1]:
    cumulative += L
    ax.axvline(cumulative, color="k", linestyle="--", alpha=0.3)

ax.set_ylabel("Correlation")
ax.set_xlabel("Sleep time (across queries)")
ax.set_title("Symmetric transfer - NO inhibition plasticity")
ax.set_ylim(-1.0, 1.0)
ax.legend(loc="upper right")

plt.tight_layout()
out_path = plots_dir / "compare_inhibition_no_plasticity.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print("Saved NO INHIB recovery plot to", out_path)
plt.show()


# %% Plot comparison - Diagonal inhibition

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

for p in range(corr_diag.shape[1]):
    ax.plot(corr_diag[:, p], label=f"Pattern {p+1}")

cumulative = 0
for L in lengths_diag[:-1]:
    cumulative += L
    ax.axvline(cumulative, color="k", linestyle="--", alpha=0.3)

ax.set_ylabel("Correlation")
ax.set_xlabel("Sleep time (across queries)")
ax.set_title("Symmetric transfer - Diagonal inhibition plasticity")
ax.set_ylim(-1.0, 1.0)
ax.legend(loc="upper right")

plt.tight_layout()
out_path = plots_dir / "compare_inhibition_diagonal.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print("Saved DIAGONAL INHIB recovery plot to", out_path)
plt.show()


# %% Plot comparison - Full matrix inhibition

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

for p in range(corr_full.shape[1]):
    ax.plot(corr_full[:, p], label=f"Pattern {p+1}")

cumulative = 0
for L in lengths_full[:-1]:
    cumulative += L
    ax.axvline(cumulative, color="k", linestyle="--", alpha=0.3)

ax.set_ylabel("Correlation")
ax.set_xlabel("Sleep time (across queries)")
ax.set_title("Symmetric transfer - Full matrix inhibition plasticity")
ax.set_ylim(-1.0, 1.0)
ax.legend(loc="upper right")

plt.tight_layout()
out_path = plots_dir / "compare_inhibition_full_matrix.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print("Saved FULL MATRIX INHIB recovery plot to", out_path)
plt.show()


# %% Visualize neuron states - No inhibition

if traj_no_inhib:
    traj0_no_inhib = traj_no_inhib[0]
    plt.figure(figsize=(8, 4))
    plt.imshow(traj0_no_inhib.T, aspect="auto", cmap="viridis")
    plt.xlabel("Time step")
    plt.ylabel("Neuron")
    plt.colorbar(label="Rate")
    plt.title("No inhibition: trajectory of first query")
    plt.tight_layout()
    out_path_traj = plots_dir / "compare_inhibition_no_plasticity_trajectory.png"
    plt.savefig(out_path_traj, dpi=300, bbox_inches="tight")
    print("Saved NO INHIB trajectory plot to", out_path_traj)
    plt.show()


# %% Visualize neuron states - Diagonal inhibition

if traj_diag:
    traj0_diag = traj_diag[0]
    plt.figure(figsize=(8, 4))
    plt.imshow(traj0_diag.T, aspect="auto", cmap="viridis")
    plt.xlabel("Time step")
    plt.ylabel("Neuron")
    plt.colorbar(label="Rate")
    plt.title("Diagonal inhibition: trajectory of first query")
    plt.tight_layout()
    out_path_traj = plots_dir / "compare_inhibition_diagonal_trajectory.png"
    plt.savefig(out_path_traj, dpi=300, bbox_inches="tight")
    print("Saved DIAGONAL INHIB trajectory plot to", out_path_traj)
    plt.show()


# %% Visualize neuron states - Full matrix inhibition

if traj_full:
    traj0_full = traj_full[0]
    plt.figure(figsize=(8, 4))
    plt.imshow(traj0_full.T, aspect="auto", cmap="viridis")
    plt.xlabel("Time step")
    plt.ylabel("Neuron")
    plt.colorbar(label="Rate")
    plt.title("Full matrix inhibition: trajectory of first query")
    plt.tight_layout()
    out_path_traj = plots_dir / "compare_inhibition_full_matrix_trajectory.png"
    plt.savefig(out_path_traj, dpi=300, bbox_inches="tight")
    print("Saved FULL MATRIX INHIB trajectory plot to", out_path_traj)
    plt.show()

# %%
