# %% [markdown]
# # McCallum Pseudorehearsal Simulation
#
# This notebook launches McCallum's pseudorehearsal simulations for the
# thesis comparison. The algorithm uses:
# - Delta learning rule with asymmetric weights
# - Probing phase to discover stable states (pseudoitems)
# - Noise applied only to new patterns
#
# **Key difference from AR**: Spurious states during probing become pseudoitems
# (useful information), whereas in AR they cause failure.

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

EXPERIMENT_NAME = "mccallum_comparison"
OUTPUT_DIR = DATA_DIR / "mccallum_results" / "mccallum"

# %% [markdown]
# ## Setup Experiment

# %%
def setup_mccallum_experiment(
    name: str,
    network_sizes: list,
    rho_values: list,
    num_seeds: int,
    max_patterns: int = 50,
    output_dir: Path = None
) -> Path:
    """
    Setup McCallum pseudorehearsal experiment.

    Creates JSON config for the C++ mccallum simulation.
    """
    if output_dir is None:
        output_dir = DATA_DIR / "mccallum_results" / "mccallum"

    config_dir = DATA_DIR / "configs" / name
    config_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "type": "mccallum",
        "output_dir": str(output_dir),
        "base_params": {
            "max_patterns": max_patterns
        },
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
config_path = setup_mccallum_experiment(
    name=EXPERIMENT_NAME,
    network_sizes=NETWORK_SIZES,
    rho_values=RHO_VALUES,
    num_seeds=NUM_SEEDS,
    max_patterns=MAX_PATTERNS,
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
print(f"\nRunning {total_sims} McCallum simulations...")
print("This may take a while for larger networks...\n")

run_cpp("mccallum", config_path, verbose=True)

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
    print("\nMcCallum M* Summary:")
    print("=" * 60)
    print(summary_df.to_string(index=False))

    # Save summary
    summary_path = OUTPUT_DIR / "M_star_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")

# %% [markdown]
# ## Quick Sanity Check
#
# For N=100, rho=0.0, McCallum's thesis shows M* ~ 10-15

# %%
if 'df' in dir() and len(df) > 0:
    sanity = df[(df['network_size'] == 100) & (df['rho'] == 0.0)]
    if len(sanity) > 0 and 'M_star' in sanity.columns:
        print(f"\nSanity check (N=100, rho=0.0):")
        print(f"  M* values: {sanity['M_star'].values}")
        print(f"  Mean: {sanity['M_star'].mean():.1f}")
        print(f"  Expected (from McCallum Fig 4.23): ~10-15")
