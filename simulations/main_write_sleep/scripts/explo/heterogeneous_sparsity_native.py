# %% [markdown]
# # Heterogeneous Sparsity Exploration (C++ Native Generation)
#
# This script uses C++ native pattern generation with heterogeneous sparsity:
# 1. Trains a 250-unit network on 10 patterns with varying sparsities (generated in C++)
# 2. Runs sleep simulation
# 3. Plots recovery frequency vs pattern sparsity
#
# This demonstrates that patterns with different sparsities can have different
# recovery probabilities during autonomous retrieval.
#
# Unlike the Python version, this uses C++ native pattern generation for
# better parallelization and performance.

# %% Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import sys

# Add scripts directory to path (parent.parent = scripts/)
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import (
    read_pattern_metadata,
    setup_write_experiment,
    setup_sleep_experiment,
    run_cpp,
    build,
    load_results,
    DATA_DIR
)

# %%=========================================================================
# CONFIGURATION SECTION
# ==========================================================================

# Network and pattern parameters
NETWORK_SIZE = 250
NUM_PATTERNS = 10
MEAN_SPARSITY = 0.5     # Center of sparsity distribution (P(0) convention)
SPARSITY_WIDTH = 0.4    # Full width: sparsities in [0.3, 0.7]
RHO = 0.3               # Pattern correlation

# Training parameters
LEAK = 1.0
DRIVE_TARGET = 6.0
LEARNING_RATE = 0.0001
MAX_ITER = 100000
MOMENTUM_COEF = 0.9
SYMMETRIC_TRANSFER = 0.0  # Use standard transfer (default)

# Sleep parameters (matching SR_sparsity_sim)
BETA = 0.1               # Inhibitory plasticity rate
DELTA = 0.01             # Integration timestep
MAX_QUERIES = 200        # Number of retrieval attempts
NOISE_DYNAMICS = 1       # Enable stochastic noise
STDDEV_DYNAMICS = 0.01   # Noise standard deviation (matching SR_sparsity_sim)
USE_FULL_INHIBITION = 0  # Diagonal inhibition only (default)

# Experiment name
EXPERIMENT_NAME = "heterogeneous_sparsity_native"

# %% [markdown]
# ## Phase 1: Build C++ Executables

# %% Build
print("="*70)
print("BUILDING C++ EXECUTABLES")
print("="*70)
build()
print("Build complete!\n")

# %% [markdown]
# ## Phase 2: Training Phase (Write) with C++ Native Pattern Generation

# %% Setup and run training
print("="*70)
print("TRAINING PHASE (C++ NATIVE PATTERN GENERATION)")
print("="*70)
print(f"Network size: {NETWORK_SIZE}")
print(f"Number of patterns: {NUM_PATTERNS}")
print(f"Mean sparsity (P(0)): {MEAN_SPARSITY}")
print(f"Sparsity width: {SPARSITY_WIDTH}")
print(f"Expected sparsity range: [{MEAN_SPARSITY - SPARSITY_WIDTH/2:.2f}, {MEAN_SPARSITY + SPARSITY_WIDTH/2:.2f}]")
print(f"Pattern correlation (rho): {RHO}")
print("="*70 + "\n")

# Use native pattern generation (C++ generates patterns with metadata)
write_config = setup_write_experiment(
    name=EXPERIMENT_NAME,
    patterns=None,  # No patterns from Python - C++ will generate them
    pattern_metadata=None,  # No metadata from Python - C++ will generate it
    params={
        # Training parameters
        "leak": LEAK,
        "drive_target": DRIVE_TARGET,
        "learning_rate": LEARNING_RATE,
        "max_iter": MAX_ITER,
        "momentum_coef": MOMENTUM_COEF,
        "symmetric_transfer": SYMMETRIC_TRANSFER,
        # Native pattern generation parameters
        "network_size": NETWORK_SIZE,
        "num_patterns": NUM_PATTERNS,
        "use_heterogeneous_sparsity": 1,  # Enable heterogeneous mode
        "mean_sparsity": MEAN_SPARSITY,
        "sparsity_width": SPARSITY_WIDTH,
        "rho": RHO,
    },
    varying_params={},
    native_pattern_generation=True,  # Enable C++ native generation
)

print(f"Configuration saved to: {write_config}\n")
print("Starting training with C++ native pattern generation...")
run_cpp("write", write_config)
print("\nTraining complete!")

# %% [markdown]
# ## Phase 3: Load and Display Generated Patterns

# %% Load generated metadata
print("\n" + "="*70)
print("GENERATED PATTERNS")
print("="*70)

# Load metadata from the trained network output
metadata_path = DATA_DIR / "trained_networks" / EXPERIMENT_NAME / "sim_nb_0" / "pattern_metadata.json"
loaded_metadata = read_pattern_metadata(metadata_path)

print(f"\nC++ generated {len(loaded_metadata['patterns'])} patterns with heterogeneous sparsities:\n")
print(f"{'Pattern':<10} {'Sparsity (P(0))':<20} {'Active Units':<15} {'Density':<15}")
print("-" * 60)
for p in loaded_metadata["patterns"]:
    density = 1 - p['sparsity']
    print(f"{p['index']:<10} {p['sparsity']:<20.4f} {p['nb_active']:<15} {density:<15.4f}")

# %% [markdown]
# ## Phase 4: Sleep Phase

# %% Setup and run sleep
print("\n" + "="*70)
print("SLEEP PHASE")
print("="*70)

sleep_config = setup_sleep_experiment(
    name=f"{EXPERIMENT_NAME}_sleep",
    trained_networks_dir=DATA_DIR / "trained_networks" / EXPERIMENT_NAME,
    params={
        "beta": BETA,
        "delta": DELTA,
        "max_queries": MAX_QUERIES,
        "noise_dynamics": NOISE_DYNAMICS,
        "stddev_dynamics": STDDEV_DYNAMICS,
        "stop_on_spurious": 1,      # Matching SR_sparsity_sim
        "stop_on_all_found": 1,     # Matching SR_sparsity_sim
        "use_full_inhibition": USE_FULL_INHIBITION,
    }
)

print(f"Configuration saved to: {sleep_config}\n")
print("Starting sleep simulation...")
run_cpp("sleep", sleep_config)
print("\nSleep simulation complete!")

# %% [markdown]
# ## Phase 5: Analysis and Visualization

# %% Load results
print("\n" + "="*70)
print("ANALYSIS")
print("="*70)

results_dir = DATA_DIR / "sleep_results" / f"{EXPERIMENT_NAME}_sleep"
results = load_results(results_dir)

# Load metadata from the sleep output
metadata_path = results_dir / "sim_nb_0" / "pattern_metadata.json"
loaded_metadata = read_pattern_metadata(metadata_path)

print(f"Loaded {len(results)} query results")
print(f"Columns: {list(results.columns)}")

# %% Compute recovery statistics per pattern
# Filter out spurious patterns (recovered_pattern_idx == -1)
valid_recoveries = results[results['recovered_pattern_idx'] >= 0]

# Count recoveries per pattern
recovery_counts = valid_recoveries.groupby('recovered_pattern_idx').size()

# Build DataFrame with pattern metadata and recovery counts
pattern_stats = []
for p in loaded_metadata["patterns"]:
    idx = p["index"]
    count = recovery_counts.get(idx, 0)
    density = 1 - p["sparsity"]  # Fraction active
    pattern_stats.append({
        "pattern_idx": idx,
        "sparsity": p["sparsity"],
        "density": density,
        "nb_active": p["nb_active"],
        "recovery_count": count
    })

stats_df = pd.DataFrame(pattern_stats)

print("\nRecovery Statistics per Pattern:")
print(stats_df.to_string(index=False))

# Total stats
total_recoveries = valid_recoveries.shape[0]
spurious_count = (results['recovered_pattern_idx'] == -1).sum()
print(f"\nTotal recoveries: {total_recoveries}")
print(f"Spurious patterns: {spurious_count}")

# %% Create visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Recovery count vs Sparsity (P(0))
ax1 = axes[0]
scatter1 = ax1.scatter(stats_df["sparsity"], stats_df["recovery_count"],
                       s=100, alpha=0.7, c=stats_df["pattern_idx"], cmap='viridis')
ax1.set_xlabel("Pattern Sparsity (P(0) = fraction inactive)", fontsize=12)
ax1.set_ylabel("Recovery Count", fontsize=12)
ax1.set_title(f"Pattern Recovery vs Sparsity\n(N={NETWORK_SIZE}, K={NUM_PATTERNS}, C++ Native Gen)", fontsize=14)

# Add trend line
if len(stats_df) > 1:
    z = np.polyfit(stats_df["sparsity"], stats_df["recovery_count"], 1)
    p_fit = np.poly1d(z)
    x_line = np.linspace(stats_df["sparsity"].min(), stats_df["sparsity"].max(), 100)
    ax1.plot(x_line, p_fit(x_line), "r--", alpha=0.5, label=f"Linear fit (slope={z[0]:.1f})")
    ax1.legend()

# Plot 2: Recovery count vs Density (fraction active)
ax2 = axes[1]
scatter2 = ax2.scatter(stats_df["density"], stats_df["recovery_count"],
                       s=100, alpha=0.7, c=stats_df["pattern_idx"], cmap='viridis')
ax2.set_xlabel("Pattern Density (fraction active)", fontsize=12)
ax2.set_ylabel("Recovery Count", fontsize=12)
ax2.set_title(f"Pattern Recovery vs Density\n(N={NETWORK_SIZE}, K={NUM_PATTERNS}, C++ Native Gen)", fontsize=14)

# Add trend line
if len(stats_df) > 1:
    z2 = np.polyfit(stats_df["density"], stats_df["recovery_count"], 1)
    p_fit2 = np.poly1d(z2)
    x_line2 = np.linspace(stats_df["density"].min(), stats_df["density"].max(), 100)
    ax2.plot(x_line2, p_fit2(x_line2), "r--", alpha=0.5, label=f"Linear fit (slope={z2[0]:.1f})")
    ax2.legend()

# Add colorbar
plt.colorbar(scatter2, ax=ax2, label="Pattern Index")

plt.tight_layout()

# Save figure
output_path = Path(__file__).parent.parent / "plots" / "heterogeneous_sparsity_native.png"
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nFigure saved to: {output_path}")

plt.show()

# %% Summary
print("\n" + "="*70)
print("EXPERIMENT COMPLETE!")
print("="*70)
print(f"\nTrained network: {DATA_DIR / 'trained_networks' / EXPERIMENT_NAME}")
print(f"Sleep results: {results_dir}")
print(f"Visualization: {output_path}")
print("\nNote: Patterns were generated natively in C++ for better performance!")
print("="*70 + "\n")

# %%
