"""
Interactive, didactic comparison between OLD and NEW pattern generators
on write + sleep dynamics for a single network.

Use this as a VS Code notebook (run cell by cell with #%%).
"""

# %% Imports and global configuration

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from utils import (
    generate_patterns_old,
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

NETWORK_SIZE = 100
NUM_PATTERNS = 3
RHO = 0.5
SPARSITY = 0.5  # For OLD: fraction of active units; for NEW: P(0)=0.5

np.random.seed(0)

# Build C++ binaries once
build()


# %% Generate OLD patterns (balanced flips)

old_patterns = generate_patterns_old(
    k=NUM_PATTERNS,
    n=NETWORK_SIZE,
    sparsity=SPARSITY,
    rho=RHO,
)

print("=== OLD PATTERNS (balanced flips) ===")
print("Shape:", old_patterns.shape)
print(old_patterns.astype(int))

old_ones_per_pattern = old_patterns.sum(axis=1)
print("Ones per pattern:", old_ones_per_pattern)
print("Mean ones:", old_ones_per_pattern.mean(), "Std:", old_ones_per_pattern.std())

# Pairwise Hamming distances
old_dists = []
for i in range(NUM_PATTERNS):
    for j in range(i + 1, NUM_PATTERNS):
        old_dists.append(np.sum(old_patterns[i] != old_patterns[j]))
old_dists = np.array(old_dists)
print("Mean pairwise Hamming distance (OLD):", old_dists.mean(), "Std:", old_dists.std())


# %% Generate NEW patterns (parent + redraw)

new_patterns = generate_patterns_new(
    k=NUM_PATTERNS,
    n=NETWORK_SIZE,
    sparsity=SPARSITY,
    rho=RHO,
)

print("\n=== NEW PATTERNS (parent + redraw) ===")
print("Shape:", new_patterns.shape)
print(new_patterns.astype(int))

new_ones_per_pattern = new_patterns.sum(axis=1)
print("Ones per pattern:", new_ones_per_pattern)
print("Mean ones:", new_ones_per_pattern.mean(), "Std:", new_ones_per_pattern.std())

new_dists = []
for i in range(NUM_PATTERNS):
    for j in range(i + 1, NUM_PATTERNS):
        new_dists.append(np.sum(new_patterns[i] != new_patterns[j]))
new_dists = np.array(new_dists)
print("Mean pairwise Hamming distance (NEW):", new_dists.mean(), "Std:", new_dists.std())


# %% Train network with OLD patterns (write phase)

write_name_old = "compare_patterns_old"
sleep_name_old = "compare_patterns_old_sleep"

write_config_old = setup_write_experiment(
    name=write_name_old,
    patterns=old_patterns,
    params={
        "leak": 1.0,
        "drive_target": 6.0,
        "learning_rate": 0.0001,
        "distance_noise_level": 0.0,
        "momentum_coef": 0.9,
        "max_iter": 100000,
    },
    native_pattern_generation=False,
)

print("\n[OLD] Running write phase...")
run_cpp("write", write_config_old)


# %% Sleep with OLD patterns (3 queries, save trajectories)

sleep_config_old = setup_sleep_experiment(
    name=sleep_name_old,
    trained_networks_dir=DATA_DIR / "trained_networks" / write_name_old,
    params={
        "beta": 0.1,
        "delta": 0.01,
        "noise_dynamics": 1,
        "stddev_dynamics": 0.01,
        "max_queries": 3,
        "stop_on_spurious": 0,
        "stop_on_all_found": 0,
        "save_trajectories": 1,
    },
)

print("[OLD] Running sleep phase...")
run_cpp("sleep", sleep_config_old)

sleep_results_dir_old = DATA_DIR / "sleep_results" / sleep_name_old
sims_old = list_simulations(sleep_results_dir_old)
sim_dir_old = sims_old[0]
cpp_old_patterns = read_patterns(sim_dir_old / "patterns.data").astype(float)
traj_old = load_trajectories(sim_dir_old)

print("Loaded OLD sleep simulation from:", sim_dir_old)


# %% Train network with NEW patterns (write phase)

write_name_new = "compare_patterns_new"
sleep_name_new = "compare_patterns_new_sleep"

write_config_new = setup_write_experiment(
    name=write_name_new,
    patterns=new_patterns,
    params={
        "leak": 1.0,
        "drive_target": 6.0,
        "learning_rate": 0.0001,
        "distance_noise_level": 0.0,
        "momentum_coef": 0.9,
        "max_iter": 100000,
    },
    native_pattern_generation=False,
)

print("\n[NEW] Running write phase...")
run_cpp("write", write_config_new)


# %% Sleep with NEW patterns (3 queries, save trajectories)

sleep_config_new = setup_sleep_experiment(
    name=sleep_name_new,
    trained_networks_dir=DATA_DIR / "trained_networks" / write_name_new,
    params={
        "beta": 0.1,
        "delta": 0.01,
        "noise_dynamics": 1,
        "stddev_dynamics": 0.01,
        "max_queries": 3,
        "stop_on_spurious": 0,
        "stop_on_all_found": 0,
        "save_trajectories": 1,
    },
)

print("[NEW] Running sleep phase...")
run_cpp("sleep", sleep_config_new)

sleep_results_dir_new = DATA_DIR / "sleep_results" / sleep_name_new
sims_new = list_simulations(sleep_results_dir_new)
sim_dir_new = sims_new[0]
cpp_new_patterns = read_patterns(sim_dir_new / "patterns.data").astype(float)
traj_new = load_trajectories(sim_dir_new)

print("Loaded NEW sleep simulation from:", sim_dir_new)


# %% Compute correlations between patterns and trajectories (OLD)

all_corr_old = []
lengths_old = []
for traj in traj_old:
    # traj: T x N
    corr_this_query = []
    for t in range(traj.shape[0]):
        state = traj[t]
        pattern_corrs = []
        for p in range(cpp_old_patterns.shape[0]):
            a = state.astype(float)
            b = cpp_old_patterns[p]
            if np.allclose(a, 0) or np.allclose(b, 0):
                corr = 0.0
            else:
                corr = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
            pattern_corrs.append(corr)
        corr_this_query.append(pattern_corrs)
    all_corr_old.extend(corr_this_query)
    lengths_old.append(len(corr_this_query))

all_corr_old = np.array(all_corr_old)


# %% Compute correlations between patterns and trajectories (NEW)

all_corr_new = []
lengths_new = []
for traj in traj_new:
    corr_this_query = []
    for t in range(traj.shape[0]):
        state = traj[t]
        pattern_corrs = []
        for p in range(cpp_new_patterns.shape[0]):
            a = state.astype(float)
            b = cpp_new_patterns[p]
            if np.allclose(a, 0) or np.allclose(b, 0):
                corr = 0.0
            else:
                corr = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
            pattern_corrs.append(corr)
        corr_this_query.append(pattern_corrs)
    all_corr_new.extend(corr_this_query)
    lengths_new.append(len(corr_this_query))

all_corr_new = np.array(all_corr_new)


# %% Plot correlation trajectories for OLD vs NEW

num_patterns = cpp_old_patterns.shape[0]

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, sharey=True)

titles = ["Old generator (balanced flips)", "New generator (parent + redraw)"]
all_corr = [all_corr_old, all_corr_new]
lengths = [lengths_old, lengths_new]

for ax, title, corr, lens in zip(axes, titles, all_corr, lengths):
    for p in range(num_patterns):
        ax.plot(corr[:, p], label=f"Pattern {p+1}")

    cumulative = 0
    for L in lens[:-1]:
        cumulative += L
        ax.axvline(cumulative, color="k", linestyle="--", alpha=0.3)

    ax.set_ylabel("Correlation")
    ax.set_title(title)

axes[-1].set_xlabel("Sleep time (across 3 queries)")
axes[0].legend(loc="upper right")

plt.tight_layout()
plots_dir = Path("plots")
plots_dir.mkdir(exist_ok=True)
out_path = plots_dir / "compare_old_new_patterns.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print("Saved plot to", out_path)
plt.show()


# %% Multi-sparsity example: s = 0.1, 0.5, 0.9 (NEW generator)

SPARSITY_LEVELS = [0.1, 0.5, 0.9]  # P(x_i = 0)
MULTI_NUM_PATTERNS = 3
MULTI_MAX_QUERIES = 5

multi_corr = {}
multi_lengths = {}

for s in SPARSITY_LEVELS:
    print(f"\n=== NEW PATTERNS for sparsity s={s} (P(0)={s}) ===")

    patterns_s = generate_patterns_new(
        k=MULTI_NUM_PATTERNS,
        n=NETWORK_SIZE,
        sparsity=s,
        rho=RHO,
    )

    print("Patterns (0/1):")
    print(patterns_s.astype(int))
    ones_per_pattern = patterns_s.sum(axis=1)
    print("Ones per pattern:", ones_per_pattern)
    print("Mean ones:", ones_per_pattern.mean(), "Std:", ones_per_pattern.std())

    # Write phase for this sparsity
    write_name_s = f"compare_multi_s_{int(s * 10)}"
    sleep_name_s = f"compare_multi_s_{int(s * 10)}_sleep"

    write_config_s = setup_write_experiment(
        name=write_name_s,
        patterns=patterns_s,
        params={
            "leak": 1.0,
            "drive_target": 6.0,
            "learning_rate": 0.0001,
            "distance_noise_level": 0.0,
            "momentum_coef": 0.9,
            "max_iter": 100000,
        },
        native_pattern_generation=False,
    )

    print(f"[s={s}] Running write phase...")
    run_cpp("write", write_config_s)

    # Sleep phase with multiple queries
    sleep_config_s = setup_sleep_experiment(
        name=sleep_name_s,
        trained_networks_dir=DATA_DIR / "trained_networks" / write_name_s,
        params={
            "beta": 0.1,
            "delta": 0.01,
            "noise_dynamics": 1,
            "stddev_dynamics": 0.01,
            "init_drive": 0.5,
            "max_queries": MULTI_MAX_QUERIES,
            "stop_on_spurious": 0,
            "stop_on_all_found": 0,
            "save_trajectories": 1,
        },
    )

    print(f"[s={s}] Running sleep phase...")
    run_cpp("sleep", sleep_config_s)

    sleep_results_dir_s = DATA_DIR / "sleep_results" / sleep_name_s
    sims_s = list_simulations(sleep_results_dir_s)
    sim_dir_s = sims_s[0]
    cpp_patterns_s = read_patterns(sim_dir_s / "patterns.data").astype(float)
    traj_s = load_trajectories(sim_dir_s)

    print(f"Loaded sleep simulation for s={s} from:", sim_dir_s)

    # Correlations for all queries at this sparsity
    all_corr_s = []
    lengths_s = []
    for traj in traj_s:
        corr_this_query = []
        for t in range(traj.shape[0]):
            state = traj[t].astype(float)
            pattern_corrs = []
            for p in range(cpp_patterns_s.shape[0]):
                a = state
                b = cpp_patterns_s[p]
                if np.allclose(a, 0) or np.allclose(b, 0):
                    c = 0.0
                else:
                    c = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
                pattern_corrs.append(c)
            corr_this_query.append(pattern_corrs)
        all_corr_s.extend(corr_this_query)
        lengths_s.append(len(corr_this_query))

    multi_corr[s] = np.array(all_corr_s)
    multi_lengths[s] = lengths_s


# %% Plot correlation trajectories for different sparsities (NEW generator)

fig, axes = plt.subplots(len(SPARSITY_LEVELS), 1, figsize=(10, 8), sharex=True, sharey=True)

if len(SPARSITY_LEVELS) == 1:
    axes = [axes]

for ax, s in zip(axes, SPARSITY_LEVELS):
    corr = multi_corr[s]
    lens = multi_lengths[s]

    for p in range(corr.shape[1]):
        ax.plot(corr[:, p], label=f"Pattern {p+1}")

    cumulative = 0
    for L in lens[:-1]:
        cumulative += L
        ax.axvline(cumulative, color="k", linestyle="--", alpha=0.3)

    ax.set_ylabel("Corr")
    ax.set_title(f"New generator, sparsity s={s}")
    ax.set_ylim(-1.0, 1.0)

axes[-1].set_xlabel("Sleep time (across queries)")
axes[0].legend(loc="upper right")

plt.tight_layout()
out_path_multi = plots_dir / "multi_sparsity_correlation.png"
plt.savefig(out_path_multi, dpi=300, bbox_inches="tight")
print("Saved multi-sparsity correlation plot to", out_path_multi)
plt.show()
