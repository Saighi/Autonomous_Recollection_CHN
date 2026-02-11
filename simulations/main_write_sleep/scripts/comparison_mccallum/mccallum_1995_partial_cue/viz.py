# %% [markdown]
# # McCallum 1995 — Partial Cue Visualization
#
# Loads CSV data produced by `sim.py` and generates a publication-ready
# figure showing pattern recovery as a function of patterns learned (M)
# at different partial-cue levels.
#
# **Expected data:** `data/mccallum_results/mccallum_1995_partial_cue/pr256_partial_cue.csv`

# %% Imports
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_THIS_DIR = Path(__file__).resolve().parent if '__file__' in dir() else Path.cwd()
_SCRIPTS_DIR = _THIS_DIR.parent.parent
sys.path.insert(0, str(_SCRIPTS_DIR))

try:
    from utils import DATA_DIR, SCRIPT_DIR
except ImportError:
    DATA_DIR   = _SCRIPTS_DIR.parent / "data"
    SCRIPT_DIR = _SCRIPTS_DIR

DATA_PATH = DATA_DIR / "mccallum_results" / "mccallum_1995_partial_cue"
PLOT_DIR  = SCRIPT_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)

# %% [markdown]
# ## Publication rcParams

# %% Style
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'font.family': 'sans-serif',
    'axes.linewidth': 1.2,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
})

# %% [markdown]
# ## Load data

# %% Load CSV
csv_path = DATA_PATH / "pr256_partial_cue.csv"
assert csv_path.exists(), f"Data not found: {csv_path}\nRun sim.py first."

df = pd.read_csv(csv_path)
n_trials = df["trial"].nunique()
print(f"Loaded {len(df)} rows, {n_trials} trials")

# %% Aggregate mean/std across trials
agg = df.groupby("M").agg(
    stable_mean=("stable", "mean"),
    stable_std=("stable", "std"),
    rec95_mean=("recovered_95", "mean"),
    rec95_std=("recovered_95", "std"),
    rec90_mean=("recovered_90", "mean"),
    rec90_std=("recovered_90", "std"),
    rec80_mean=("recovered_80", "mean"),
    rec80_std=("recovered_80", "std"),
    rec50_mean=("recovered_50", "mean"),
    rec50_std=("recovered_50", "std"),
    pseudo_mean=("pseudo", "mean"),
).reset_index()

# Fill NaN std (single-trial case)
for col in agg.columns:
    if col.endswith("_std"):
        agg[col] = agg[col].fillna(0.0)

M = agg["M"].values

# %% [markdown]
# ## Main figure: Recovery vs M at different cue levels

# %% Define line styles (gradient from easy to hard)
LINES = [
    ("stable_mean", "stable_std", "#444444", "-",  2.5, "Stability (100%)"),
    ("rec95_mean",  "rec95_std",  "#1B4F72", "-",  2.2, "95% cue"),
    ("rec90_mean",  "rec90_std",  "#2980B9", "-",  2.2, "90% cue"),
    ("rec80_mean",  "rec80_std",  "#17A589", "-",  2.2, "80% cue"),
    ("rec50_mean",  "rec50_std",  "#E74C3C", "-",  2.2, "50% cue"),
]

# %% Plot
fig, ax = plt.subplots(figsize=(8, 6))

for mean_col, std_col, color, ls, lw, label in LINES:
    y_mean = agg[mean_col].values
    y_std  = agg[std_col].values

    ax.plot(M, y_mean, color=color, ls=ls, lw=lw, label=label, zorder=3)
    ax.fill_between(M, y_mean - y_std, y_mean + y_std,
                    color=color, alpha=0.15, zorder=2)

ax.set_xlabel("Patterns learned ($M$)", fontsize=20)
ax.set_ylabel("Recovered patterns", fontsize=20)
ax.set_xlim(M[0] - 1, M[-1] + 2)
ax.set_ylim(0, None)
ax.tick_params(labelsize=16, width=1.2, length=5)

ax.legend(loc="upper left", fontsize=14, frameon=False)

for spine in ax.spines.values():
    spine.set_linewidth(1.2)

plt.tight_layout()
save_path = PLOT_DIR / "mccallum_1995_partial_cue.png"
fig.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()
print(f"Saved: {save_path}")

# %% [markdown]
# ## Summary statistics

# %% Summary
print("\n" + "=" * 60)
print("SUMMARY — Pr256 Partial Cue Recovery")
print(f"  Trials: {n_trials}")
print("=" * 60)

for mean_col, _, _, _, _, label in LINES:
    y = agg[mean_col].values
    peak_idx = np.argmax(y)
    print(f"\n  {label}:")
    print(f"    At M={M[-1]:3d}: {y[-1]:.1f}")
    print(f"    Peak:  {y[peak_idx]:.1f} at M={M[peak_idx]}")

    for target_M in [15, 30, 50]:
        hits = [i for i, x in enumerate(M) if x >= target_M]
        if hits:
            print(f"    At M={target_M:3d}: {y[hits[0]]:.1f}")
