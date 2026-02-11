# %% [markdown]
# # McCallum 1995 — Visualization
#
# Loads CSV data produced by `sim.py` and generates
# publication-ready figures.
#
# **Expected data:** `data/mccallum_results/mccallum_1995/{condition}.csv`
# with columns: `condition, trial, M, stable, pseudo`

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

DATA_PATH = DATA_DIR / "mccallum_results" / "mccallum_1995"
PLOT_DIR  = SCRIPT_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)

# %% [markdown]
# ## Load data

# %% Load all conditions
CONDITIONS = ["pr100", "pr256", "pr512", "delta_hetero", "delta_gaussian"]

frames = {}
for cond in CONDITIONS:
    path = DATA_PATH / f"{cond}.csv"
    if path.exists():
        frames[cond] = pd.read_csv(path)
        n_trials = frames[cond]["trial"].nunique()
        n_steps  = frames[cond].groupby("trial").size().iloc[0]
        print(f"  {cond:20s}: {n_trials} trials, {n_steps} steps")
    else:
        print(f"  {cond:20s}: NOT FOUND ({path})")

assert len(frames) > 0, "No data found! Run sim.py first."

# %% Compute per-step mean/std for each condition
def summarise(df: pd.DataFrame):
    """Return M, mean_stable, std_stable, mean_pseudo, std_pseudo."""
    grouped = df.groupby("M").agg(
        stable_mean=("stable", "mean"),
        stable_std=("stable", "std"),
        pseudo_mean=("pseudo", "mean"),
        pseudo_std=("pseudo", "std"),
    ).reset_index()
    return grouped

stats = {cond: summarise(df) for cond, df in frames.items()}

# %% [markdown]
# ## Main comparison figure

# %% Style definitions
STYLE = {
    "delta_hetero":   {"color": "#c0392b", "ls": "-",  "lw": 2.5, "label": "Iterative GDA"},
    "delta_gaussian": {"color": "#e74c3c", "ls": "--", "lw": 2.2, "label": r"Delta (Gaussian $\sigma$=0.5)"},
    "pr100":          {"color": "#8e44ad", "ls": "-",  "lw": 2.2, "label": "Pr100"},
    "pr256":          {"color": "#2980b9", "ls": "-",  "lw": 2.2, "label": "Pr256"},
    "pr512":          {"color": "#16a085", "ls": "-",  "lw": 2.2, "label": "Pr512"},
}

# Which conditions to show (edit to toggle)
SHOW = ["delta_hetero", "pr100", "pr256", "pr512"]

# %% Comparison plot — publication figure
fig, ax = plt.subplots(figsize=(8, 6))

# Perfect recall
M_ref = stats[next(iter(stats))]["M"].values
# ax.plot(M_ref, M_ref, color="0.45", ls="-", lw=1.8,
#         label="Perfect recall", zorder=1)

# Data curves
for cond in SHOW:
    if cond not in stats:
        continue
    s = stats[cond]
    st = STYLE[cond]
    ax.plot(s["M"], s["stable_mean"],
            color=st["color"], ls=st["ls"], lw=st["lw"],
            label=st["label"], zorder=2 if "delta" in cond else 3)

# Axes
pr_maxes = [stats[c]["stable_mean"].max() for c in SHOW if c in stats]
ymax = max(25, int(max(pr_maxes)) + 5)
ax.set_xlabel("Patterns learned ($M$)", fontsize=20)
ax.set_ylabel("Stable patterns", fontsize=20)
ax.set_xlim(0, M_ref[-1] + 2)
ax.set_ylim(0, ymax)
ax.tick_params(labelsize=16, width=1.2, length=5)

ax.legend(loc="upper left", fontsize=15, frameon=False)

for spine in ax.spines.values():
    spine.set_linewidth(1.2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
save_path = PLOT_DIR / "mccallum_1995_comparison.png"
fig.savefig(save_path, dpi=200, bbox_inches="tight")
plt.show()
print(f"Saved: {save_path}")

# %% [markdown]
# ## Pseudoitems plot

# %% Pseudoitems per step (Pr conditions only)
PR_CONDS = [c for c in ["pr100", "pr256", "pr512"] if c in stats]

if PR_CONDS:
    fig, ax = plt.subplots(figsize=(8, 5))

    for cond in PR_CONDS:
        s  = stats[cond]
        st = STYLE[cond]
        ax.plot(s["M"], s["pseudo_mean"],
                color=st["color"], ls="-", lw=2.2, label=st["label"])

    # Cap reference lines
    caps = {"pr100": 100, "pr256": 256, "pr512": 512}
    for cond in PR_CONDS:
        ax.axhline(caps[cond], ls="--", color=STYLE[cond]["color"],
                    alpha=0.35, lw=1)

    ax.set_xlabel("Patterns learned ($M$)", fontsize=18)
    ax.set_ylabel("Pseudoitems found", fontsize=18)
    ax.set_xlim(0, M_ref[-1] + 2)
    ax.tick_params(labelsize=15, width=1.2, length=5)
    ax.legend(fontsize=14, frameon=False)

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    save_path = PLOT_DIR / "mccallum_1995_pseudoitems.png"
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Saved: {save_path}")

# %% [markdown]
# ## Summary statistics

# %% Summary table
print("=" * 60)
print("SUMMARY")
print("=" * 60)

for cond in CONDITIONS:
    if cond not in stats:
        continue
    s = stats[cond]
    m = s["stable_mean"].values
    label = STYLE[cond]["label"]
    M_vals = s["M"].values

    peak_idx = np.argmax(m)
    print(f"\n  {label}:")
    print(f"    Stable at M={M_vals[-1]:3d}: {m[-1]:.1f}")
    print(f"    Peak: {m[peak_idx]:.1f} at M={M_vals[peak_idx]}")

    for target in [15, 30, 50]:
        hits = [i for i, x in enumerate(M_vals) if x >= target]
        if hits:
            print(f"    Stable at M={target:3d}: {m[hits[0]]:.1f}")
