# %% [markdown]
# # CI (Continuous Incorporation) M* Capacity with Multi-Cue Evaluation
#
# Runs AR incremental simulations across a grid of network sizes and rho values,
# evaluating at 4 cue levels (100%, 95%, 80%, 50%) per incorporation step.
#
# **Protocol:**
# 1. Incremental incorporation with sleep consolidation (AR)
# 2. After each step, test all M patterns at 4 cue levels
# 3. Early stopping: failed cue levels are frozen; stability failure aborts the run
# 4. M*_s(cue) = max M where all M patterns recovered at that cue level
#
# **Output:**
# - `data/mccallum_results/ci_vs_mccallum_capacity/all_simulation_data.csv`
# - `data/mccallum_results/ci_vs_mccallum_capacity/M_star_summary.csv`

# %% Imports
import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

# %% Paths
_THIS_DIR = Path(__file__).resolve().parent if '__file__' in dir() else Path.cwd()
_SCRIPTS_DIR = _THIS_DIR.parent.parent
sys.path.insert(0, str(_SCRIPTS_DIR))

from utils import DATA_DIR, build, run_cpp

OUTPUT_DIR = DATA_DIR / "mccallum_results" / "ci_vs_mccallum_capacity"

# %% Configuration (matching McCallum capacity grid)
NETWORK_SIZES = [50, 100, 150, 200, 250]
RHO_VALUES    = [0.0, 0.2, 0.4, 0.6, 0.8]
NUM_SEEDS     = 30
MAX_PATTERNS  = 50
THETA         = 0.9

EXPERIMENT_NAME = "ci_vs_mccallum_capacity"

# AR/CI parameters
CI_PARAMS = {
    "leak": 1.0,
    "drive_target": 6.0,
    "learning_rate": 0.0001,
    "momentum_coef": 0.9,
    "delta": 0.01,
    "beta": 0.1,
    "stddev_dynamics": 0.01,
    "noise_dynamics": 1.0,
    "max_sleep_queries": 100,
    "max_iter": 100000,
    "max_patterns": MAX_PATTERNS,
}

# %% Setup experiment
def setup_experiment():
    """Create JSON config for the C++ ar_incremental simulation."""
    config_dir = DATA_DIR / "configs" / EXPERIMENT_NAME
    config_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "type": "ar_incremental",
        "output_dir": str(OUTPUT_DIR),
        "base_params": CI_PARAMS,
        "varying_params": {
            "network_size": NETWORK_SIZES,
            "rho": RHO_VALUES,
            "seed": list(range(NUM_SEEDS)),
        }
    }

    config_path = config_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    return config_path

# %% M* computation
def compute_M_star(M_star_values, theta=0.9):
    """M* = max M such that >= theta fraction achieved M*_s >= M."""
    if len(M_star_values) == 0:
        return 0
    max_M = max(M_star_values)
    for M in range(max_M, -1, -1):
        fraction = sum(1 for m in M_star_values if m >= M) / len(M_star_values)
        if fraction >= theta:
            return M
    return 0

# %% Run
total_sims = len(NETWORK_SIZES) * len(RHO_VALUES) * NUM_SEEDS
print(f"\n{'='*60}")
print(f"CI M* Capacity — Multi-Cue Evaluation")
print(f"  Network sizes: {NETWORK_SIZES}")
print(f"  Rho values:    {RHO_VALUES}")
print(f"  Seeds:         {NUM_SEEDS}")
print(f"  Max patterns:  {MAX_PATTERNS}")
print(f"  Cue levels:    [1.00, 0.95, 0.80, 0.50]")
print(f"  Total runs:    {total_sims}")
print(f"{'='*60}\n")

# Build
print("Building C++ simulations...")
if not build():
    print("Build failed!")
    sys.exit(1)
print("Build successful!\n")

# Setup config
config_path = setup_experiment()
print(f"Config saved to: {config_path}")
print(f"Output will be in: {OUTPUT_DIR}\n")

# Run C++ simulations
t0 = time.time()
print(f"Running {total_sims} CI incremental simulations...")
run_cpp("ar_incremental", config_path, verbose=True)
elapsed = time.time() - t0
print(f"\nC++ simulations done in {elapsed:.0f}s ({elapsed/60:.1f}min)")

# %% Load results
results_csv = OUTPUT_DIR / "all_simulation_data.csv"
if not results_csv.exists():
    print(f"ERROR: Results not found at {results_csv}")
    sys.exit(1)

df = pd.read_csv(results_csv)
print(f"\nLoaded {len(df)} simulation records")

# %% Compute M* summary
# The C++ code writes M_star_stable, M_star_rec95, M_star_rec80, M_star_rec50
# into parameters.data, which gets aggregated into all_simulation_data.csv

CUE_MAP = {
    "stable": "M_star_stable",
    "rec95":  "M_star_rec95",
    "rec80":  "M_star_rec80",
    "rec50":  "M_star_rec50",
}

summary_rows = []
for N in NETWORK_SIZES:
    for rho in RHO_VALUES:
        subset = df[(df['network_size'] == N) & (df['rho'] == rho)]

        for cue_label, col_name in CUE_MAP.items():
            if col_name not in subset.columns:
                print(f"WARNING: Column {col_name} not found in data")
                continue

            M_star_s_list = subset[col_name].values.tolist()
            M_star = compute_M_star(M_star_s_list, THETA)
            mean_ms = np.mean(M_star_s_list) if M_star_s_list else 0
            std_ms = np.std(M_star_s_list) if M_star_s_list else 0

            summary_rows.append({
                "N": N,
                "rho": rho,
                "cue_level": cue_label,
                "M_star": M_star,
                "mean_M_star": round(float(mean_ms), 2),
                "std_M_star": round(float(std_ms), 2),
                "num_sims": len(M_star_s_list),
            })

df_summary = pd.DataFrame(summary_rows)
summary_csv = OUTPUT_DIR / "M_star_summary.csv"
df_summary.to_csv(summary_csv, index=False)
print(f"\nM* summary saved: {summary_csv}")

# %% Print summary table
print(f"\n{'='*60}")
print("CI M* Summary")
print(f"{'='*60}")

for rho in RHO_VALUES:
    print(f"\n  rho = {rho:.1f}:")
    print(f"  {'N':>5s}  {'stable':>7s}  {'rec95':>7s}  {'rec80':>7s}  {'rec50':>7s}")
    print(f"  {'-'*5}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}")
    for N in NETWORK_SIZES:
        vals = {}
        for _, row in df_summary[(df_summary["N"] == N) & (df_summary["rho"] == rho)].iterrows():
            vals[row["cue_level"]] = int(row["M_star"])
        print(f"  {N:5d}  {vals.get('stable',0):7d}  {vals.get('rec95',0):7d}  "
              f"{vals.get('rec80',0):7d}  {vals.get('rec50',0):7d}")

print(f"\nTotal elapsed: {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f}min)")
