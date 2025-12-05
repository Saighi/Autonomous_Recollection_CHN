# %% [markdown]
# # Neural Network Write/Sleep Workflow
# Generic workflow for training and sleep simulations.

# %% Imports
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from utils import (
    generate_patterns,
    setup_write_experiment,
    setup_sleep_experiment,
    run_cpp,
    build,
    load_results,
    load_simulation,
    list_simulations,
    load_trajectories,
    DATA_DIR
)

# %% Build C++ (run once or after code changes)
build()

# %% ==========================================================================
# EXAMPLE 1: Simple single experiment
# =============================================================================

# %% Generate patterns
network_size = 100
sparsity = 0.5  # Fraction of active units
num_patterns = 10

patterns = generate_patterns(
    k=num_patterns,
    n=network_size,
    sparsity=sparsity,
    rho=0.5  # Pattern correlation (1=identical, 0=uncorrelated)
)

print(f"Generated {len(patterns)} patterns of size {network_size}")
print(f"Each pattern has {patterns[0].sum()} active units")

# %% Setup write experiment
write_config = setup_write_experiment(
    name="example_simple",
    patterns=patterns,
    params={
        "leak": 1.0,
        "drive_target": 6.0,
        "learning_rate": 0.0001,
        "distance_noise_level": 0.0,
        "momentum_coef": 0.9,
    }
)
print(f"Config saved to: {write_config}")

# %% Run training
run_cpp("write", write_config)

# %% Setup sleep experiment on trained network
sleep_config = setup_sleep_experiment(
    name="example_simple_sleep",
    trained_networks_dir=DATA_DIR / "trained_networks" / "example_simple",
    params={
        "beta": 0.1,              # Inhibitory plasticity rate
        "delta": 0.01,            # Integration timestep
        "noise_dynamics": 1,      # Enable noise in dynamics
        "stddev_dynamics": 0.01,  # Noise standard deviation
        "init_drive": 0.5,        # Initial state
        "max_queries": 200,       # Max retrieval attempts
        "stop_on_spurious": 1,    # Stop when spurious pattern encountered
        "stop_on_all_found": 0,   # Don't stop when all patterns found
        "save_trajectories": 0,
    }
)

# %% Run sleep simulation
run_cpp("sleep", sleep_config)

# %% Load and plot results
results = load_results(DATA_DIR / "sleep_results" / "example_simple_sleep")
print(results.head())

# %% ==========================================================================
# EXAMPLE 2: Parameter sweep
# =============================================================================

# %% Generate patterns for sweep
patterns = generate_patterns(k=15, n=100, sparsity=0.5, rho=0.5)

# %% Setup write experiment with varying parameters
write_config = setup_write_experiment(
    name="sweep_leak",
    patterns=patterns,
    params={
        "drive_target": 6.0,
        "learning_rate": 0.0001,
        "distance_noise_level": 0.0,
        "momentum_coef": 0.9,
    },
    varying_params={
        "leak": [0.25, 0.5, 1.0, 1.5, 2.0],  # Sweep over leak values
    }
)

# %% Run training (C++ parallelizes internally)
run_cpp("write", write_config)

# %% Setup sleep with parameter sweep
sleep_config = setup_sleep_experiment(
    name="sweep_leak_sleep",
    trained_networks_dir=DATA_DIR / "trained_networks" / "sweep_leak",
    params={
        "delta": 0.01,
        "noise_dynamics": 1,
        "stddev_dynamics": 0.01,
        "init_drive": 0.5,
        "max_queries": 200,
        "stop_on_spurious": 1,
        "stop_on_all_found": 0,
        "save_trajectories": 0,
    },
    varying_params={
        "beta": [0.05, 0.1, 0.2],  # Sweep inhibitory plasticity
    }
)

# %% Run sleep
run_cpp("sleep", sleep_config)

# %% Load and analyze
results = load_results(DATA_DIR / "sleep_results" / "sweep_leak_sleep")

# %% Plot: retrieval vs leak for different beta values
fig, ax = plt.subplots(figsize=(8, 5))

for beta in results['beta'].unique():
    subset = results[results['beta'] == beta]
    # Get final retrieval count per simulation
    final_results = subset.groupby('leak').agg({'nb_fnd_pat': 'max'}).reset_index()
    ax.plot(final_results['leak'], final_results['nb_fnd_pat'], 'o-', label=f'beta={beta}')

ax.set_xlabel('Leak')
ax.set_ylabel('Patterns Retrieved')
ax.legend()
ax.set_title('Memory Retrieval vs Network Leak')
plt.tight_layout()
plt.show()

# %% ==========================================================================
# EXAMPLE 3: Full parameter grid
# =============================================================================

# %% Varying patterns AND network parameters
# Use a single experiment name and per-run subfolders via run_name
grid_experiment = "grid_sweep"
for num_pat in [5, 10, 15, 20]:
    for net_size in [64, 100, 144]:
        patterns = generate_patterns(k=num_pat, n=net_size, sparsity=0.5, rho=0.5)

        config = setup_write_experiment(
            name=grid_experiment,
            run_name=f"n{net_size}_p{num_pat}",
            patterns=patterns,
            params={
                "leak": 1.0,
                "drive_target": 6.0,
                "learning_rate": 0.0001,
                "distance_noise_level": 0.0,
                "momentum_coef": 0.9,
            }
        )

        run_cpp("write", config, verbose=False)
        print(f"Trained: n={net_size}, patterns={num_pat}")

# %% ==========================================================================
# EXAMPLE 4: Inspecting individual simulations
# =============================================================================

# %% List all simulations in a results folder
sims = list_simulations(DATA_DIR / "trained_networks" / "example_simple")
print(f"Found {len(sims)} simulations")

# %% Load a specific simulation
sim_data = load_simulation(sims[0])
print(f"Parameters: {sim_data['parameters']}")
print(f"Weight matrix shape: {sim_data['weights'].shape}")
print(f"Patterns shape: {sim_data['patterns'].shape}")

# %% Visualize weight matrix
plt.figure(figsize=(8, 8))
plt.imshow(sim_data['weights'], cmap='RdBu_r', vmin=-np.abs(sim_data['weights']).max(),
           vmax=np.abs(sim_data['weights']).max())
plt.colorbar(label='Weight')
plt.title('Trained Weight Matrix')
plt.show()

# %% Visualize patterns
n_show = min(5, len(sim_data['patterns']))
fig, axes = plt.subplots(1, n_show, figsize=(12, 3))
side = int(np.sqrt(sim_data['patterns'].shape[1]))

for i, ax in enumerate(axes):
    ax.imshow(sim_data['patterns'][i].reshape(side, side) if side**2 == sim_data['patterns'].shape[1]
              else sim_data['patterns'][i].reshape(1, -1), cmap='binary')
    ax.set_title(f'Pattern {i}')
    ax.axis('off')

plt.tight_layout()
plt.show()

# %% ==========================================================================
# EXAMPLE 5: Load trajectories (if saved)
# =============================================================================

# %% Setup sleep with trajectory saving
sleep_config = setup_sleep_experiment(
    name="with_trajectories",
    trained_networks_dir=DATA_DIR / "trained_networks" / "example_simple",
    params={
        "beta": 0.1,
        "delta": 0.01,
        "noise_dynamics": 1,
        "stddev_dynamics": 0.01,
        "init_drive": 0.5,
        "max_queries": 10,        # Few queries for demo
        "stop_on_spurious": 0,    # Continue even after spurious
        "stop_on_all_found": 0,   # Continue even after all found
        "save_trajectories": 1,   # Enable saving
    }
)

run_cpp("sleep", sleep_config)

# %% Load and plot trajectory
sims = list_simulations(DATA_DIR / "sleep_results" / "with_trajectories")
if sims:
    trajectories = load_trajectories(sims[0])
    if trajectories:
        print(f"Loaded {len(trajectories)} trajectories")

        # Plot first trajectory
        traj = trajectories[0]
        plt.figure(figsize=(10, 4))
        plt.imshow(traj.T, aspect='auto', cmap='viridis')
        plt.xlabel('Time step')
        plt.ylabel('Neuron')
        plt.colorbar(label='Rate')
        plt.title('Network Trajectory During Retrieval')
        plt.show()
