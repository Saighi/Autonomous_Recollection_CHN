# %% [markdown]
# # DHN (Hebbian + Storkey) Simulations
#
# This notebook launches DHN simulations using existing infrastructure.
# Two learning rules are compared:
#
# 1. **Hebbian**: W_ij += (1/N) * xi_i * xi_j
#    - Simple outer product rule
#    - Theoretical capacity: ~0.138*N patterns
#
# 2. **Storkey**: W_ij += (1/N) * [xi_i*xi_j - xi_i*h_j - h_i*xi_j]
#    - Local field correction reduces crosstalk
#    - Theoretical capacity: ~0.42*N patterns

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

# Experimental grid from spec (matching McCallum/AR)
NETWORK_SIZES = [50,75,100,125, 150,175, 200,225,250]
RHO_VALUES = [0.0, 0.2, 0.4, 0.5, 0.6]
NUM_SEEDS = 10  # Simulations per (N, rho)
MAX_PATTERNS = 50
THETA = 0.9  # Success threshold for M*
INFORMED_FRACTION = 0.5  # 50% partial cues

# Output directories
HEBBIAN_TRAIN_DIR = DATA_DIR / "mccallum_results" / "hebbian_trained"
HEBBIAN_QUERY_DIR = DATA_DIR / "mccallum_results" / "hebbian"
STORKEY_TRAIN_DIR = DATA_DIR / "mccallum_results" / "storkey_trained"
STORKEY_QUERY_DIR = DATA_DIR / "mccallum_results" / "storkey"

# %% [markdown]
# ## Setup and Run Hebbian

# %%
# Setup Hebbian training
hebbian_train_config = setup_dhn_train_experiment(
    name="mccallum_hebbian_train",
    params={
        "learning_rule": 0,  # 0 = Hebbian
        "sparsity": 0.5
    },
    varying_params={
        "network_size": NETWORK_SIZES,
        "num_patterns": list(range(1, MAX_PATTERNS + 1)),
        "rho": RHO_VALUES,
        "seed": list(range(NUM_SEEDS))
    },
    output_dir=HEBBIAN_TRAIN_DIR
)

print(f"Hebbian training config: {hebbian_train_config}")

# Calculate total simulations
total_train = len(NETWORK_SIZES) * MAX_PATTERNS * len(RHO_VALUES) * NUM_SEEDS
print(f"Total training simulations: {total_train}")

# %% [markdown]
# **Note**: The grid above trains networks for ALL values of num_patterns
# (1 to 50). This is different from McCallum/AR which train incrementally.
# We'll compute M* by finding the maximum num_patterns where queries succeed.

# %%
# Build if needed
print("Building C++ simulations...")
build_result = build()
if not build_result:
    print("Build failed!")
else:
    print("Build successful!")

# %%
# Run Hebbian training
print(f"\nRunning Hebbian training ({total_train} simulations)...")
print("This will take a while...\n")

run_cpp("dhn_train", hebbian_train_config, verbose=True)

# %%
# Setup Hebbian query
hebbian_query_config = setup_dhn_query_experiment(
    name="mccallum_hebbian_query",
    trained_networks_dir=HEBBIAN_TRAIN_DIR,
    params={
        "nb_dynamics_steps": 10
    },
    varying_params={
        "informed_fraction": [INFORMED_FRACTION]
    },
    output_dir=HEBBIAN_QUERY_DIR
)

print(f"Hebbian query config: {hebbian_query_config}")

# %%
# Run Hebbian query
print("\nRunning Hebbian queries...")
run_cpp("dhn_query", hebbian_query_config, verbose=True)

# %% [markdown]
# ## Setup and Run Storkey

# %%
# Setup Storkey training
storkey_train_config = setup_dhn_train_experiment(
    name="mccallum_storkey_train",
    params={
        "learning_rule": 1,  # 1 = Storkey
        "sparsity": 0.5
    },
    varying_params={
        "network_size": NETWORK_SIZES,
        "num_patterns": list(range(1, MAX_PATTERNS + 1)),
        "rho": RHO_VALUES,
        "seed": list(range(NUM_SEEDS))
    },
    output_dir=STORKEY_TRAIN_DIR
)

print(f"Storkey training config: {storkey_train_config}")

# %%
# Run Storkey training
print(f"\nRunning Storkey training ({total_train} simulations)...")
run_cpp("dhn_train", storkey_train_config, verbose=True)

# %%
# Setup Storkey query
storkey_query_config = setup_dhn_query_experiment(
    name="mccallum_storkey_query",
    trained_networks_dir=STORKEY_TRAIN_DIR,
    params={
        "nb_dynamics_steps": 10
    },
    varying_params={
        "informed_fraction": [INFORMED_FRACTION]
    },
    output_dir=STORKEY_QUERY_DIR
)

print(f"Storkey query config: {storkey_query_config}")

# %%
# Run Storkey query
print("\nRunning Storkey queries...")
run_cpp("dhn_query", storkey_query_config, verbose=True)

# %% [markdown]
# ## Collect and Process Results

# %%
def compute_M_star_from_queries(df, theta=0.9):
    """
    Compute M* from DHN query results.

    For each (N, rho, seed) combination:
    1. Find max num_patterns where all queries succeeded
    2. Then aggregate across seeds using theta threshold

    The DHN query data has per-pattern rows with 'recovered' column (0 or 1).
    We need to aggregate by (N, rho, seed, num_patterns) to check if ALL patterns
    were recovered for that configuration.
    """
    results = []

    # First, aggregate per-pattern results to per-(N, rho, seed, num_patterns) level
    # Check if 'recovered' column exists (per-pattern data)
    if 'recovered' in df.columns:
        # Aggregate: for each (N, rho, seed, num_patterns), count recovered patterns
        agg = df.groupby(['network_size', 'rho', 'seed', 'num_patterns']).agg({
            'recovered': ['sum', 'count']
        }).reset_index()
        agg.columns = ['network_size', 'rho', 'seed', 'num_patterns', 'recovered_sum', 'total']
        # All patterns recovered if sum == count
        agg['all_recovered'] = (agg['recovered_sum'] == agg['total'])
    else:
        # Fallback: use query_success_rate if available
        agg = df.copy()
        if 'query_success_rate' in agg.columns:
            agg['all_recovered'] = agg['query_success_rate'] >= 1.0
        else:
            print("Warning: No 'recovered' or 'query_success_rate' column found!")
            return pd.DataFrame()

    for N in agg['network_size'].unique():
        for rho in agg['rho'].unique():
            M_star_per_seed = []

            for seed in agg['seed'].unique():
                subset = agg[
                    (agg['network_size'] == N) &
                    (agg['rho'] == rho) &
                    (agg['seed'] == seed)
                ].sort_values('num_patterns')

                # Find max num_patterns where ALL patterns were recovered
                M_star_s = 0
                for _, row in subset.iterrows():
                    if row['all_recovered']:
                        M_star_s = int(row['num_patterns'])
                    # Note: we don't break on failure because we want max M where success

                M_star_per_seed.append(M_star_s)

            if len(M_star_per_seed) > 0:
                # Compute M* using theta threshold
                max_M = max(M_star_per_seed)
                M_star = 0
                for M in range(max_M, -1, -1):
                    fraction = sum(1 for m in M_star_per_seed if m >= M) / len(M_star_per_seed)
                    if fraction >= theta:
                        M_star = M
                        break

                results.append({
                    'N': N,
                    'rho': rho,
                    'M_star': M_star,
                    'mean_M_star': np.mean(M_star_per_seed),
                    'std_M_star': np.std(M_star_per_seed),
                    'num_seeds': len(M_star_per_seed)
                })

    return pd.DataFrame(results)

# %%
# Load Hebbian results
hebbian_csv = HEBBIAN_QUERY_DIR / "all_simulation_data.csv"
if hebbian_csv.exists():
    hebbian_df = pd.read_csv(hebbian_csv)
    print(f"Loaded {len(hebbian_df)} Hebbian query records")

    # Compute M*
    hebbian_summary = compute_M_star_from_queries(hebbian_df, THETA)
    print("\nHebbian M* Summary:")
    print("=" * 60)
    print(hebbian_summary.to_string(index=False))

    # Save
    hebbian_summary.to_csv(HEBBIAN_QUERY_DIR / "M_star_summary.csv", index=False)
else:
    print(f"Hebbian results not found at {hebbian_csv}")

# %%
# Load Storkey results
storkey_csv = STORKEY_QUERY_DIR / "all_simulation_data.csv"
if storkey_csv.exists():
    storkey_df = pd.read_csv(storkey_csv)
    print(f"Loaded {len(storkey_df)} Storkey query records")

    # Compute M*
    storkey_summary = compute_M_star_from_queries(storkey_df, THETA)
    print("\nStorkey M* Summary:")
    print("=" * 60)
    print(storkey_summary.to_string(index=False))

    # Save
    storkey_summary.to_csv(STORKEY_QUERY_DIR / "M_star_summary.csv", index=False)
else:
    print(f"Storkey results not found at {storkey_csv}")

# %% [markdown]
# ## Comparison Summary

# %%
if 'hebbian_summary' in dir() and 'storkey_summary' in dir():
    print("\nDHN Comparison (Hebbian vs Storkey):")
    print("=" * 60)

    comparison = hebbian_summary[['N', 'rho', 'M_star']].copy()
    comparison.columns = ['N', 'rho', 'Hebbian_M_star']
    comparison = comparison.merge(
        storkey_summary[['N', 'rho', 'M_star']],
        on=['N', 'rho']
    )
    comparison.columns = ['N', 'rho', 'Hebbian', 'Storkey']

    print(comparison.to_string(index=False))

    # Theoretical comparison
    print("\n\nTheoretical capacity ratios:")
    print("  Hebbian: 0.138 * N")
    print("  Storkey: 0.42 * N")
    print("\nActual capacity ratios (averaged):")
    if len(comparison) > 0:
        hebbian_ratio = comparison.groupby('N').apply(
            lambda x: x['Hebbian'].mean() / x['N'].iloc[0]
        ).mean()
        storkey_ratio = comparison.groupby('N').apply(
            lambda x: x['Storkey'].mean() / x['N'].iloc[0]
        ).mean()
        print(f"  Hebbian: {hebbian_ratio:.3f}")
        print(f"  Storkey: {storkey_ratio:.3f}")

# %% [markdown]
# ## Alternative: Simplified DHN M* Computation
#
# If the full grid is too expensive, we can use a more efficient approach
# by finding M* via binary search for each (N, rho, seed).

# %%
def setup_dhn_mstar_experiment_efficient(
    name: str,
    learning_rule: int,  # 0=Hebbian, 1=Storkey
    network_sizes: list,
    rho_values: list,
    num_seeds: int,
    max_patterns: int = 50,
    output_dir: Path = None
) -> dict:
    """
    Setup a more efficient M* experiment by testing fewer num_patterns values.

    Instead of testing all 1..50, test: [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    Then do binary search between values where success transitions to failure.
    """
    pattern_values = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    pattern_values = [p for p in pattern_values if p <= max_patterns]

    config_dir = DATA_DIR / "configs" / name
    config_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "type": "dhn_train",
        "native_pattern_generation": True,
        "output_dir": str(output_dir),
        "base_params": {
            "learning_rule": learning_rule,
            "sparsity": 0.5
        },
        "varying_params": {
            "network_size": network_sizes,
            "num_patterns": pattern_values,
            "rho": rho_values,
            "seed": list(range(num_seeds))
        }
    }

    config_path = config_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    return config_path

# Example: run efficient version instead of full grid
# efficient_config = setup_dhn_mstar_experiment_efficient(
#     name="efficient_hebbian",
#     learning_rule=0,
#     network_sizes=NETWORK_SIZES,
#     rho_values=RHO_VALUES,
#     num_seeds=NUM_SEEDS,
#     output_dir=DATA_DIR / "mccallum_results" / "hebbian_efficient"
# )
