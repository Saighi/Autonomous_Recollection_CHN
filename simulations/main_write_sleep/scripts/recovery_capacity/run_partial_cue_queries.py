# %% [markdown]
# # Run Partial Cue Query Simulations
#
# ## Overview
#
# This script tests the **partial cue retrieval** capability of trained Continuous
# Hopfield Networks (CHN). After training networks to store patterns and validating
# their autonomous recovery (AR) capacity during sleep, we test whether these networks
# can retrieve stored patterns when given only a **partial cue** (incomplete input).
#
# ## Methodology
#
# ### Partial Cue Construction
# For each stored pattern, we construct a partial cue by:
# 1. Starting with the full pattern encoded as firing rates (high ~0.997 for active,
#    low ~0.003 for inactive units)
# 2. Randomly selecting `(1 - informed_fraction)` of the units and setting them to
#    the neutral state (0.5), effectively removing their information
# 3. Only `informed_fraction` of units retain their pattern-specific values
#
# For example, with `informed_fraction = 0.2`:
# - 20% of units keep their pattern values (the "cue")
# - 80% of units are set to neutral 0.5 (uninformed)
#
# ### Query Process
# The network is initialized with this partial cue and allowed to evolve according
# to its trained dynamics (with noise, without inhibitory plasticity). We then check
# if the network converges to the correct pattern using winner-take-all comparison.
#
# ## Tested Configurations
#
# We only test configurations that achieved **>= 90% successful autonomous recovery**
# during sleep simulations. This ensures we are probing the retrieval capacity of
# networks that have demonstrably good attractor basins.
#
# ## Key Findings
#
# In the load regime that allows 90% successful AR, we tested queries with as little
# as **10% informed units** (90% of the cue masked). The results show that as long as
# the partial cue correlates more strongly with the queried pattern than with other
# stored patterns (i.e., has a smaller Hamming distance to the target), the network
# reliably recovers the correct pattern. This demonstrates the robustness of the
# trained attractor basins.
#
# ## Data Flow
#
# 1. **Input**: Trained networks from `data/trained_networks/Fig_load_SR_many_correlation_diag_inh/`
#    - These networks were trained with varying (network_size, num_patterns, correlation)
# 2. **Processing**: C++ simulation (`bin/query`) tests each pattern with each informed_fraction
# 3. **Output**: Results saved to `data/query_results/partial_cue_correlations/`
#    - Per-simulation results with pattern-level success/failure
#    - Aggregated CSV for analysis in `recovery_capacity_analysis.py`

# %% Imports
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import setup_query_experiment, run_cpp, DATA_DIR

# %% Configuration
EXPERIMENT_NAME = "partial_cue_correlations"
TRAINED_NETWORKS_DIR = DATA_DIR / "trained_networks" / "Fig_load_SR_many_correlation_diag_inh"
INFORMED_FRACTIONS = [0.9, 0.5, 0.3, 0.2]  # Fraction of units that keep pattern info

# %% Setup and run query experiment
print("=" * 70)
print("PARTIAL CUE QUERY SIMULATION")
print("=" * 70)
print(f"Trained networks: {TRAINED_NETWORKS_DIR}")
print(f"Informed fractions: {INFORMED_FRACTIONS}")
print("=" * 70)

config_path = setup_query_experiment(
    name=EXPERIMENT_NAME,
    trained_networks_dir=TRAINED_NETWORKS_DIR,
    params={
        "delta": 0.01,
        "noise_dynamics": 1,
        "stddev_dynamics": 0.01
    },
    varying_params={
        "informed_fraction": INFORMED_FRACTIONS
    }
)

print(f"\nConfig saved to: {config_path}")
print("\nRunning C++ query simulation...")

run_cpp("query", config_path, verbose=True)

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
print(f"Results saved to: {DATA_DIR / 'query_results' / EXPERIMENT_NAME}")

# %%
