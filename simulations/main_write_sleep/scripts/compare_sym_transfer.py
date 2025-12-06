"""
Compare recovery of 3 patterns in a 50-unit network
with standard vs symmetrized transfer function.

Use the NEW pattern generator (parent + redraw) with rho = 0.5.
Run this as a VS Code notebook with #%% cells.
"""

# %% Imports and configuration

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from utils import (
    generate_patterns_new,
    setup_write_experiment,
    setup_sleep_experiment,
    run_cpp,
    build,
    list_simulations,
    load_trajectories,
    read_patterns,
    DATA_DIR,
)

NETWORK_SIZE = 50
NUM_PATTERNS = 3
RHO = 0.5
SPARSITY = 0.5  # P(x_i = 0) for NEW generator
MAX_QUERIES = 5

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


# %% Write phase: standard transfer (symmetric_transfer = 0)

write_name_std = "compare_sym_std"
sleep_name_std = "compare_sym_std_sleep"

write_config_std = setup_write_experiment(
    name=write_name_std,
    patterns=patterns,
    params={
        "leak": 1.0,
        "drive_target": 6.0,
        "learning_rate": 0.0001,
        "distance_noise_level": 0.0,
        "momentum_coef": 0.9,
        "max_iter": 100000,
        "symmetric_transfer": 0.0,  # standard sigmoid in [0,1]
    },
    native_pattern_generation=False,
)

print("\n[STD] Running write phase...")
run_cpp("write", write_config_std)


# %% Sleep phase: standard transfer

sleep_config_std = setup_sleep_experiment(
    name=sleep_name_std,
    trained_networks_dir=DATA_DIR / "trained_networks" / write_name_std,
    params={
        "beta": 0.1,
        "delta": 0.01,
        "noise_dynamics": 1,
        "stddev_dynamics": 0.01,
        "max_queries": MAX_QUERIES,
        "stop_on_spurious": 0,
        "stop_on_all_found": 0,
        "save_trajectories": 1,
    },
)

print("[STD] Running sleep phase...")
run_cpp("sleep", sleep_config_std)

sleep_results_dir_std = DATA_DIR / "sleep_results" / sleep_name_std
sims_std = list_simulations(sleep_results_dir_std)
sim_dir_std = sims_std[0]
cpp_patterns_std = read_patterns(sim_dir_std / "patterns.data").astype(float)
traj_std = load_trajectories(sim_dir_std)

print("Loaded STD sleep simulation from:", sim_dir_std)


# %% Write phase: symmetrized transfer (output shifted by -0.5)

write_name_sym = "compare_sym_sym"
sleep_name_sym = "compare_sym_sym_sleep"

write_config_sym = setup_write_experiment(
    name=write_name_sym,
    patterns=patterns,
    params={
        "leak": 1.0,
        "drive_target": 6.0,
        "learning_rate": 0.0001,
        "distance_noise_level": 0.0,
        "momentum_coef": 0.9,
        "max_iter": 100000,
        "symmetric_transfer": 1.0,  # sigmoid(x) - 0.5, centered around 0
    },
    native_pattern_generation=False,
)

print("\n[SYM] Running write phase...")
run_cpp("write", write_config_sym)


# %% Sleep phase: symmetrized transfer

sleep_config_sym = setup_sleep_experiment(
    name=sleep_name_sym,
    trained_networks_dir=DATA_DIR / "trained_networks" / write_name_sym,
    params={
        "beta": 0.1,
        "delta": 0.01,
        "noise_dynamics": 1,
        "stddev_dynamics": 0.01,
        "max_queries": MAX_QUERIES,
        "stop_on_spurious": 0,
        "stop_on_all_found": 0,
        "save_trajectories": 1,
    },
)

print("[SYM] Running sleep phase...")
run_cpp("sleep", sleep_config_sym)

sleep_results_dir_sym = DATA_DIR / "sleep_results" / sleep_name_sym
sims_sym = list_simulations(sleep_results_dir_sym)
sim_dir_sym = sims_sym[0]
cpp_patterns_sym = read_patterns(sim_dir_sym / "patterns.data").astype(float)
traj_sym = load_trajectories(sim_dir_sym)

print("Loaded SYM sleep simulation from:", sim_dir_sym)


# %% Compute correlations over time for STD and SYM

def compute_corr(traj_list, patterns_arr):
    all_corr = []
    lengths = []
    for traj in traj_list:
        corr_this_query = []
        for t in range(traj.shape[0]):
            state = traj[t].astype(float)
            pattern_corrs = []
            for p in range(patterns_arr.shape[0]):
                a = state
                b = patterns_arr[p]
                if np.allclose(a, 0) or np.allclose(b, 0):
                    c = 0.0
                else:
                    c = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
                pattern_corrs.append(c)
            corr_this_query.append(pattern_corrs)
        all_corr.extend(corr_this_query)
        lengths.append(len(corr_this_query))
    return np.array(all_corr), lengths


corr_std, lengths_std = compute_corr(traj_std, cpp_patterns_std)
corr_sym, lengths_sym = compute_corr(traj_sym, cpp_patterns_sym)


# %% Plot comparison of recovery (STD vs SYM)

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, sharey=True)

for ax, title, corr, lens in zip(
    axes,
    ["Standard transfer", "Symmetrized transfer (shift -0.5)"],
    [corr_std, corr_sym],
    [lengths_std, lengths_sym],
):
    for p in range(corr.shape[1]):
        ax.plot(corr[:, p], label=f"Pattern {p+1}")

    cumulative = 0
    for L in lens[:-1]:
        cumulative += L
        ax.axvline(cumulative, color="k", linestyle="--", alpha=0.3)

    ax.set_ylabel("Correlation")
    ax.set_title(title)
    ax.set_ylim(-1.0, 1.0)

axes[-1].set_xlabel("Sleep time (across queries)")
axes[0].legend(loc="upper right")

plt.tight_layout()
plots_dir = Path("plots")
plots_dir.mkdir(exist_ok=True)
out_path = plots_dir / "compare_sym_transfer.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print("Saved sym vs std transfer plot to", out_path)
plt.show()


# %% Visualize neuron states over time for first query (STD vs SYM)

if traj_std:
    traj0_std = traj_std[0]
    plt.figure(figsize=(8, 4))
    plt.imshow(traj0_std.T, aspect="auto", cmap="viridis")
    plt.xlabel("Time step")
    plt.ylabel("Neuron")
    plt.colorbar(label="Rate")
    plt.title("STD transfer: trajectory of first query")
    plt.tight_layout()
    out_path_std_traj = plots_dir / "compare_sym_std_trajectory.png"
    plt.savefig(out_path_std_traj, dpi=300, bbox_inches="tight")
    print("Saved STD trajectory plot to", out_path_std_traj)
    plt.show()

if traj_sym:
    traj0_sym = traj_sym[0]
    plt.figure(figsize=(8, 4))
    plt.imshow(traj0_sym.T, aspect="auto", cmap="viridis")
    plt.xlabel("Time step")
    plt.ylabel("Neuron")
    plt.colorbar(label="Rate")
    plt.title("SYM transfer: trajectory of first query")
    plt.tight_layout()
    out_path_sym_traj = plots_dir / "compare_sym_sym_trajectory.png"
    plt.savefig(out_path_sym_traj, dpi=300, bbox_inches="tight")
    print("Saved SYM trajectory plot to", out_path_sym_traj)
    plt.show()
