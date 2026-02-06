# %% [markdown]
# # AR (Continuous Incorporation) Simulation
#
# This notebook launches AR incremental simulations for the thesis comparison.
# The algorithm uses:
# - Continuous Hopfield Network (CHN) with sigmoid activations
# - Batch GDA learning with momentum
# - Sleep consolidation via autonomous retrieval (AR)
#
# **Key difference from McCallum**: Spurious states during sleep cause FAILURE,
# whereas in McCallum they become useful pseudoitems.

# %% [markdown]
# ## Configuration

# %%
import sys
from pathlib import Path

# Get absolute path to parent scripts directory
_THIS_DIR = Path(__file__).resolve().parent if '__file__' in dir() else Path.cwd()
_SCRIPTS_DIR = _THIS_DIR.parent
sys.path.insert(0, str(_SCRIPTS_DIR))

from utils import *
import json

# Experimental grid from spec
NETWORK_SIZES = [50,75,100,125, 150,175, 200,225,250]
RHO_VALUES = [0.0, 0.2, 0.4, 0.5, 0.6]
NUM_SEEDS = 10  # Simulations per (N, rho)
MAX_PATTERNS = 50
THETA = 0.9  # Success threshold for M*

EXPERIMENT_NAME = "ar_comparison"
OUTPUT_DIR = DATA_DIR / "mccallum_results" / "ar"

# AR-specific parameters
AR_PARAMS = {
    "leak": 1.0,
    "drive_target": 6.0,
    "learning_rate": 0.0001,
    "momentum_coef": 0.9,
    "delta": 0.01,
    "beta": 0.1,
    "stddev_dynamics": 0.01,
    "noise_dynamics": 1.0,  # Enable noise
    "max_sleep_queries": 100,
    "max_iter": 100000
}

# %% [markdown]
# ## Setup Experiment

# %%
def setup_ar_incremental_experiment(
    name: str,
    network_sizes: list,
    rho_values: list,
    num_seeds: int,
    max_patterns: int = 50,
    ar_params: dict = None,
    output_dir: Path = None
) -> Path:
    """
    Setup AR incremental experiment.

    Creates JSON config for the C++ ar_incremental simulation.
    """
    if output_dir is None:
        output_dir = DATA_DIR / "mccallum_results" / "ar"
    if ar_params is None:
        ar_params = {}

    config_dir = DATA_DIR / "configs" / name
    config_dir.mkdir(parents=True, exist_ok=True)

    base_params = {
        "max_patterns": max_patterns,
        **ar_params
    }

    config = {
        "type": "ar_incremental",
        "output_dir": str(output_dir),
        "base_params": base_params,
        "varying_params": {
            "network_size": network_sizes,
            "rho": rho_values,
            "seed": list(range(num_seeds))
        }
    }

    config_path = config_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    return config_path

# %%
# Create configuration
config_path = setup_ar_incremental_experiment(
    name=EXPERIMENT_NAME,
    network_sizes=NETWORK_SIZES,
    rho_values=RHO_VALUES,
    num_seeds=NUM_SEEDS,
    max_patterns=MAX_PATTERNS,
    ar_params=AR_PARAMS,
    output_dir=OUTPUT_DIR
)

print(f"Config saved to: {config_path}")
print(f"Output will be in: {OUTPUT_DIR}")

# Calculate total simulations
total_sims = len(NETWORK_SIZES) * len(RHO_VALUES) * NUM_SEEDS
print(f"Total simulations: {total_sims}")

# %% [markdown]
# ## Build and Run

# %%
# Build if needed
print("Building C++ simulations...")
build_result = build()
if not build_result:
    print("Build failed!")
else:
    print("Build successful!")

# %%
# Run simulations
print(f"\nRunning {total_sims} AR incremental simulations...")
print("This may take a while for larger networks...\n")

run_cpp("ar_incremental", config_path, verbose=True)

# %% [markdown]
# ## Collect Results

# %%
# Load aggregated results
results_csv = OUTPUT_DIR / "all_simulation_data.csv"
if results_csv.exists():
    df = pd.read_csv(results_csv)
    print(f"Loaded {len(df)} simulation records")
    print(df.head())
else:
    print(f"Results not found at {results_csv}")
    print("Run the simulations first!")

# %% [markdown]
# ## Summary Statistics

# %%
def compute_M_star(M_star_values, theta=0.9):
    """
    Compute M* from list of individual M*_s values.

    M* = max M such that >= theta fraction achieved M*_s >= M
    """
    if len(M_star_values) == 0:
        return 0

    max_M = max(M_star_values)
    for M in range(max_M, -1, -1):
        fraction = sum(1 for m in M_star_values if m >= M) / len(M_star_values)
        if fraction >= theta:
            return M
    return 0

# %%
if 'df' in dir() and len(df) > 0:
    # Compute M* for each (N, rho) configuration
    summary = []

    for N in NETWORK_SIZES:
        for rho in RHO_VALUES:
            subset = df[(df['network_size'] == N) & (df['rho'] == rho)]
            if 'M_star' in subset.columns:
                M_star_list = subset['M_star'].values
                M_star = compute_M_star(M_star_list, THETA)
                mean_M = M_star_list.mean()
                std_M = M_star_list.std()

                summary.append({
                    'N': N,
                    'rho': rho,
                    'M_star': M_star,
                    'mean_M_star': mean_M,
                    'std_M_star': std_M,
                    'num_sims': len(M_star_list)
                })

    summary_df = pd.DataFrame(summary)
    print("\nAR M* Summary:")
    print("=" * 60)
    print(summary_df.to_string(index=False))

    # Save summary
    summary_path = OUTPUT_DIR / "M_star_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")

# %% [markdown]
# ## Analysis: Sleep Failures vs Query Failures
#
# For AR, failures can occur during:
# 1. Sleep phase (spurious state encountered)
# 2. Query phase (50% partial cue didn't retrieve correct pattern)

# %%
if 'df' in dir() and len(df) > 0:
    # Load detailed results for analysis
    print("\nAnalyzing failure modes...")

    sleep_failures = 0
    query_failures = 0
    total_runs = len(df)

    # This requires looking at individual sim results
    # For now, just show overall statistics
    print(f"Total simulation runs: {total_runs}")
    if 'M_star' in df.columns:
        print(f"Mean M* across all runs: {df['M_star'].mean():.2f}")
        print(f"Max M* achieved: {df['M_star'].max()}")

# %% [markdown]
# ## Comparison with Expected Results
#
# AR's capacity depends heavily on:
# - Sleep effectiveness (avoiding spurious states)
# - GDA convergence quality
#
# The theoretical capacity should be higher than McCallum due to continuous
# activations and better optimization, but the strict spurious-as-failure
# criterion may limit practical capacity.

# %%
if 'df' in dir() and len(df) > 0:
    print("\nAR capacity summary by network size:")
    for N in NETWORK_SIZES:
        subset = df[df['network_size'] == N]
        if len(subset) > 0 and 'M_star' in subset.columns:
            print(f"  N={N}: mean M*={subset['M_star'].mean():.1f}, "
                  f"capacity ratio = {subset['M_star'].mean()/N:.3f}")
