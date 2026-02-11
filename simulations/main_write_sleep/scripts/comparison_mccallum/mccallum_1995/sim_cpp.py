# %% [markdown]
# # McCallum 1995 — C++ Backend Simulation
#
# Runs the same 5 conditions as `sim.py` but delegates all
# heavy computation to the shared `bin/mccallum` C++ backend.
#
# **Conditions:**
# - Pr100 / Pr256 / Pr512 — pseudorehearsal with varying pseudoitem caps
# - Delta (hetero) — pure delta learning, heteroassociative noise
# - Delta (Gaussian) — pure delta learning, Gaussian input noise
#
# **Protocol:**
# 1. Base population BP=5 trained first (no noise)
# 2. Pp = 2000 probes, r = 4N relaxation, inverse rejected
# 3. Asymmetric weight updates (McCallum Eq. 2.7)
# 4. Stability tracking at every step (no early stop)
#
# **Output:** `data/mccallum_results/mccallum_1995/{condition}.csv`
# with columns: `condition, trial, M, stable, pseudo`
# (same format as the Numba version — compatible with `viz.py`)

# %% Imports
import sys
import json
import time
from pathlib import Path

import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent if '__file__' in dir() else Path.cwd()
_SCRIPTS_DIR = _THIS_DIR.parent.parent
sys.path.insert(0, str(_SCRIPTS_DIR))

from utils import DATA_DIR, BIN_DIR, build, run_cpp

# %% [markdown]
# ## Configuration

# %% Editable knobs
NETWORK_SIZE = 100
BASE_POP     = 5
MAX_PATTERNS = 100   # base_pop + max_new = 5 + 95
N_TRIALS     = 1
SEED_OFFSET  = 0     # first seed value

# McCallum algorithm parameters (shared across all conditions)
SHARED_PARAMS = {
    "eta":             0.1,
    "max_epochs":      500,
    "error_criterion": 0.001,
    "nu_h":            0.05,
    "sigma_input":     0.5,
    "n_probes":        2000,
}

# Per-condition overrides: mode (0=PR, 1=delta_hetero, 2=delta_gaussian)
# and max_pseudoitems where applicable
CONDITIONS = {
    "pr100":          {"mode": 0, "max_pseudoitems": 100},
    "pr256":          {"mode": 0, "max_pseudoitems": 256},
    "pr512":          {"mode": 0, "max_pseudoitems": 512},
    "delta_hetero":   {"mode": 1},
    "delta_gaussian": {"mode": 2},
}

# Which conditions to run (edit to toggle)
RUN = ["pr100", "pr256", "pr512", "delta_hetero", "delta_gaussian"]

# Output paths
RAW_DIR = DATA_DIR / "mccallum_results" / "mccallum_1995_raw"
CSV_DIR = DATA_DIR / "mccallum_results" / "mccallum_1995"

# %% [markdown]
# ## Build C++

# %% Build
print("Building C++ simulations...")
if not build():
    raise RuntimeError("Build failed!")
print("Build OK")

# %% [markdown]
# ## Run conditions

# %% Helper: run one condition through C++ backend
def run_condition(name, cond_params):
    """Generate config, run C++ binary, parse results, save CSV."""
    t0 = time.time()

    # --- Create JSON config ---
    config_dir = DATA_DIR / "configs" / "mccallum_1995" / name
    config_dir.mkdir(parents=True, exist_ok=True)

    base_params = {
        "network_size":    NETWORK_SIZE,
        "max_patterns":    MAX_PATTERNS,
        "rho":             0.0,
        "base_pop":        BASE_POP,
        "stop_on_failure": 0,   # Don't stop — track stable count through all M
        **SHARED_PARAMS,
        **cond_params,
    }

    config = {
        "type": "mccallum",
        "output_dir": str(RAW_DIR / name),
        "base_params": base_params,
        "varying_params": {
            "seed": [SEED_OFFSET + s for s in range(N_TRIALS)]
        },
    }

    config_path = config_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # --- Run C++ ---
    print(f"\n{'='*60}")
    print(f"  {name} — {N_TRIALS} trials, N={NETWORK_SIZE}")
    print(f"{'='*60}")
    run_cpp("mccallum", config_path, verbose=True)

    # --- Parse results.data from each sim and build CSV ---
    rows = []
    for trial in range(N_TRIALS):
        results_path = RAW_DIR / name / f"sim_nb_{trial}" / "results.data"
        if not results_path.exists():
            print(f"  WARNING: missing {results_path}")
            continue
        df = pd.read_csv(results_path)
        for _, row in df.iterrows():
            rows.append({
                "condition": name,
                "trial":     trial,
                "M":         int(row["M"]),
                "stable":    int(row["num_stable"]),
                "pseudo":    int(row["num_pseudoitems"]),
            })

    # --- Save CSV (same format as Numba version) ---
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(rows)
    csv_path = CSV_DIR / f"{name}.csv"
    out_df.to_csv(csv_path, index=False)

    elapsed = time.time() - t0
    n_steps = out_df.groupby("trial").size().iloc[0] if len(out_df) > 0 else 0
    print(f"  Saved {csv_path.name} ({N_TRIALS} trials, {n_steps} steps, {elapsed:.1f}s)")

    return out_df


# %% Run all conditions
all_results = {}
for cond_name in RUN:
    if cond_name in CONDITIONS:
        all_results[cond_name] = run_condition(cond_name, CONDITIONS[cond_name])

# %% Done
print(f"\n{'='*60}")
print(f"All conditions saved to: {CSV_DIR}")
print(f"Raw C++ output in:       {RAW_DIR}")
print(f"{'='*60}")

# %% [markdown]
# ## Quick summary

# %% Summary
for name, df in all_results.items():
    if len(df) == 0:
        continue
    last_M = df.groupby("trial").last()
    mean_stable = last_M["stable"].mean()
    print(f"  {name:20s}: final stable = {mean_stable:.1f}/{MAX_PATTERNS} "
          f"(avg over {N_TRIALS} trials)")
