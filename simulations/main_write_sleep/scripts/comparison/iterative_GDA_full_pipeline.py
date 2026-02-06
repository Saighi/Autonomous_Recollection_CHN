# %% [markdown]
# # Iterative GDA CHN Full Pipeline - Training and Query
#
# This notebook demonstrates **catastrophic forgetting** in Continuous Hopfield Networks
# when patterns are trained ONE AT A TIME (iterative GDA) instead of simultaneously (batch GDA).
#
# ## What is Iterative GDA?
#
# ### Batch GDA (Standard - write.cc):
# ```cpp
# while (max_error > epsilon) {
#     for ALL patterns p:
#         gradient_descent(pattern_p)  // All patterns get updates each iteration
#     max_error = compute_error()
# }
# ```
# All patterns are trained **simultaneously** - weights converge to store all patterns in a balanced way.
#
# ### Iterative GDA (This experiment - write_iterative.cc):
# ```cpp
# for EACH pattern p:
#     while (max_error > epsilon):
#         gradient_descent(pattern_p)  // ONLY this pattern until convergence
#     // Move to next pattern - previous patterns may be corrupted!
# ```
# Patterns are trained **sequentially** - pattern 1 is trained until perfect recall, then pattern 2, etc.
#
# ## Why Iterative GDA Performs Poorly
#
# 1. **Catastrophic forgetting**: Training pattern N disrupts memories of patterns 1..N-1
# 2. **No balance**: Unlike batch training which balances all patterns, iterative optimizes greedily
# 3. **Expected result**: High recovery for last few patterns, poor for early patterns
# 4. **This demonstrates why batch/Hopfield-style learning is important**
#
# ## Comparison with DHN
#
# This uses the SAME parameters as the DHN comparison for a fair comparison:
# - Same network sizes, pattern counts, correlations
# - Same informed fractions for query
# - Expected: Iterative GDA << Batch GDA ~ DHN Storkey

# %% [markdown]
# ## Imports
# %%
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import (
    setup_write_iterative_experiment,
    setup_query_experiment,
    run_cpp,
    build,
    DATA_DIR
)

# %% [markdown]
# ## Configuration - Write Phase
#
# These parameters MATCH the DHN comparison for fair evaluation.

# %%
# Repetitions
NB_REPETITION = 2
REPETITIONS = list(range(NB_REPETITION))

# Network and pattern parameters (SAME as DHN)
NETWORK_SIZES = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
NUM_PATTERNS = list(range(1, 100, 2))  # 1, 3, 5, ..., 99
CORRELATIONS = [0.6, 0.4,0.5, 0.2, 0.0]
SPARSITY = 0.5

# CHN-specific training parameters
LEAK = 1.0
DRIVE_TARGET = 6.0
LEARNING_RATE = 0.001
MOMENTUM_COEF = 0.9

# Experiment names
ITERATIVE_NAME = "comparison_chn_iterative_small"
ITERATIVE_QUERY_NAME = "comparison_chn_iterative_query_small"

# %% [markdown]
# ## Configuration - Query Phase
#
# Test partial cue retrieval with various informed fractions (SAME as DHN).

# %%
# Informed fractions for partial cue queries
INFORMED_FRACTIONS = [0.5]

# Query parameters
DELTA = 0.01           # Integration timestep
NOISE_DYNAMICS = 1     # Enable noise during dynamics
STDDEV_DYNAMICS = 0.01 # Noise level

# %% [markdown]
# ## Configuration Summary

# %%
# Calculate total simulations
sims_iterative = len(NETWORK_SIZES) * len(NUM_PATTERNS) * len(CORRELATIONS) * NB_REPETITION
queries_per_network = len(INFORMED_FRACTIONS)

print("=" * 70)
print("ITERATIVE GDA CHN FULL PIPELINE - Configuration Summary")
print("=" * 70)
print(f"\nWRITE PHASE (Iterative Training):")
print(f"  Training mode:   ITERATIVE (patterns trained one-at-a-time)")
print(f"  Repetitions:     {NB_REPETITION}")
print(f"  Network sizes:   {NETWORK_SIZES[0]} to {NETWORK_SIZES[-1]} ({len(NETWORK_SIZES)} values)")
print(f"  Pattern counts:  {NUM_PATTERNS[0]} to {NUM_PATTERNS[-1]} ({len(NUM_PATTERNS)} values)")
print(f"  Correlations:    {CORRELATIONS}")
print(f"  Sparsity:        {SPARSITY}")
print(f"  Leak:            {LEAK}")
print(f"  Drive target:    {DRIVE_TARGET}")
print(f"  Learning rate:   {LEARNING_RATE}")
print(f"  Momentum:        {MOMENTUM_COEF}")
print("-" * 70)
print(f"  Total networks:  {sims_iterative:,}")
print(f"\nQUERY PHASE:")
print(f"  Informed fractions:  {INFORMED_FRACTIONS}")
print(f"  Delta:               {DELTA}")
print(f"  Noise:               {STDDEV_DYNAMICS if NOISE_DYNAMICS else 'None'}")
print(f"  Queries per network: {queries_per_network}")
print(f"  Total queries:       {sims_iterative * queries_per_network:,}")
print("=" * 70)
print("\nEXPECTED BEHAVIOR:")
print("  - Early patterns (index 0,1,2...) should have LOW recovery")
print("  - Last patterns should have HIGH recovery (most recently trained)")
print("  - Overall recovery << DHN (especially Storkey)")
print("=" * 70)

# %% [markdown]
# ## Build C++ Executables
#
# Compile the C++ simulation binaries including the new write_iterative.

# %%
print("Building C++ executables...")
build_success = build()
if build_success:
    print("Build complete!")
else:
    print("Build failed! Check compilation errors.")

# %% [markdown]
# ---
# # PHASE 1: WRITE (Iterative Training)
# ---

# %% [markdown]
# ## Train CHN with Iterative GDA
#
# This demonstrates catastrophic forgetting by training patterns sequentially.
# Each pattern is trained until convergence before moving to the next.
# Training pattern N will corrupt the weights storing patterns 1..N-1.

# %%
print("\n" + "=" * 70)
print("TRAINING CHN NETWORKS WITH ITERATIVE GDA")
print("=" * 70)
print("\nNOTE: This demonstrates CATASTROPHIC FORGETTING")
print("      Each pattern is trained until convergence, then the next pattern.")
print("      Expect: Last patterns = high recovery, Early patterns = low recovery")

iterative_config = setup_write_iterative_experiment(
    name=ITERATIVE_NAME,
    params={
        "sparsity": SPARSITY,
        "leak": LEAK,
        "drive_target": DRIVE_TARGET,
        "learning_rate": LEARNING_RATE,
        "momentum_coef": MOMENTUM_COEF,
    },
    varying_params={
        "network_size": NETWORK_SIZES,
        "num_patterns": NUM_PATTERNS,
        "rho": CORRELATIONS,
        "seed": REPETITIONS,
    },
    native_pattern_generation=True
)

print(f"\nConfiguration saved to: {iterative_config}")
print("\nStarting ITERATIVE training (patterns one-at-a-time)...")
run_cpp("write_iterative", iterative_config)
print("Iterative training complete!")

# %% [markdown]
# ## Write Phase Summary

# %%
print("\n" + "=" * 70)
print("WRITE PHASE COMPLETE")
print("=" * 70)
print(f"\nIterative CHN networks: {DATA_DIR / 'trained_networks' / ITERATIVE_NAME}")

# %% [markdown]
# ---
# # PHASE 2: QUERY (Partial Cue Retrieval)
# ---

# %% [markdown]
# ## Check Prerequisites
#
# Verify that trained networks exist before running queries.

# %%
iterative_dir = DATA_DIR / "trained_networks" / ITERATIVE_NAME

# Check prerequisites
prereqs_ok = True

if not iterative_dir.exists():
    print(f"ERROR: Iterative CHN networks not found at {iterative_dir}")
    print("Please run the training cells first.")
    prereqs_ok = False
else:
    iterative_sims = len([d for d in iterative_dir.iterdir()
                          if d.is_dir() and d.name.startswith("sim_nb_")])
    print(f"Found {iterative_sims} iterative CHN networks")

if prereqs_ok:
    print("\nPrerequisites OK - ready for query phase")

# %% [markdown]
# ## Query Iterative CHN Networks
#
# Test partial cue retrieval on networks trained with iterative GDA.
# Expect overall poor performance due to catastrophic forgetting.

# %%
if prereqs_ok:
    print("\n" + "=" * 70)
    print("QUERYING ITERATIVE CHN NETWORKS")
    print("=" * 70)

    iterative_query_config = setup_query_experiment(
        name=ITERATIVE_QUERY_NAME,
        trained_networks_dir=iterative_dir,
        params={
            "delta": DELTA,
            "noise_dynamics": NOISE_DYNAMICS,
            "stddev_dynamics": STDDEV_DYNAMICS,
        },
        varying_params={
            "informed_fraction": INFORMED_FRACTIONS,
        }
    )

    print(f"Configuration saved to: {iterative_query_config}")
    print("\nStarting iterative CHN queries...")
    run_cpp("query", iterative_query_config)
    print("Iterative CHN queries complete!")
else:
    print("Skipping queries - prerequisites not met")

# %% [markdown]
# ## Final Summary

# %%
if prereqs_ok:
    print("\n" + "=" * 70)
    print("FULL PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\nTrained Networks:")
    print(f"  Iterative CHN: {DATA_DIR / 'trained_networks' / ITERATIVE_NAME}")
    print(f"\nQuery Results:")
    print(f"  Iterative CHN: {DATA_DIR / 'query_results' / ITERATIVE_QUERY_NAME}")
    print(f"\nNext Steps:")
    print("  1. Run viz script to generate comparison figures")
    print("  2. Compare with DHN results in data/query_results/comparison_dhn_*")
    print("  3. Compare with batch CHN results (if available)")
    print("=" * 70)
    print("\nEXPECTED RESULTS:")
    print("  - Iterative GDA should show catastrophic forgetting")
    print("  - Overall success rate << DHN Storkey (~42%)")
    print("  - Overall success rate << DHN Hebbian (~16%)")
    print("  - Last trained patterns have higher recovery than early ones")
    print("=" * 70)

# %%
