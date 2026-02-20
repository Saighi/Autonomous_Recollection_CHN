# %% [markdown]
# # McCallum M* Capacity — Partial Cue Visualization
#
# Single row of subplots (one per rho), with lines for each cue level.
# X = network size, Y = M*.
#
# **Expected data:** `data/mccallum_results/mccallum_capacity_partial_cue/M_star_summary.csv`

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

DATA_PATH = DATA_DIR / "mccallum_results" / "mccallum_capacity_partial_cue"
PLOT_DIR  = SCRIPT_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)

# %% Style
plt.rcParams.update({
    'font.size': 18,
    'axes.labelsize': 30,
    'axes.titlesize': 30,
    'xtick.labelsize': 30,
    'ytick.labelsize': 30,
    'legend.fontsize': 16,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'font.family': 'sans-serif',
    'axes.linewidth': 1.4,
    'xtick.major.width': 1.4,
    'ytick.major.width': 1.4,
})

# %% Load data
summary_csv = DATA_PATH / "M_star_summary.csv"
raw_csv = DATA_PATH / "raw_results.csv"

assert summary_csv.exists(), f"Data not found: {summary_csv}\nRun sim.py first."
df = pd.read_csv(summary_csv)
print(f"Loaded M* summary: {len(df)} rows")

# Also load raw results for std bands
has_raw = raw_csv.exists()
if has_raw:
    df_raw = pd.read_csv(raw_csv)
    print(f"Loaded raw results: {len(df_raw)} rows")

rho_values = sorted(df["rho"].unique())
n_rho = len(rho_values)
N_values = sorted(df["N"].unique())

print(f"  Rho values:    {rho_values}")
print(f"  Network sizes: {N_values}")

# %% Line styles
LINES = [
    ("stable", "#444444", "o", 2.5, "Stability (100%)"),
    ("rec95",  "#1B4F72", "s", 2.2, "95% cue"),
    ("rec80",  "#17A589", "^", 2.2, "80% cue"),
    ("rec50",  "#E74C3C", "D", 2.2, "50% cue"),
]

# %% Plot
fig, axes = plt.subplots(1, n_rho, figsize=(4 * n_rho, 5), sharey=True)
if n_rho == 1:
    axes = [axes]

for ax_idx, rho in enumerate(rho_values):
    ax = axes[ax_idx]
    sub = df[df["rho"] == rho]

    for cue_label, color, marker, lw, label in LINES:
        cue_sub = sub[sub["cue_level"] == cue_label].sort_values("N")
        if len(cue_sub) == 0:
            continue

        x = cue_sub["N"].values
        y = cue_sub["M_star"].values
        y_mean = cue_sub["mean_M_star"].values
        y_std = cue_sub["std_M_star"].values

        ax.plot(x, y, color=color, lw=lw, marker=marker, markersize=8,
                label=label if ax_idx == 0 else None, zorder=3)

    ax.set_title(r"$\rho = " + f"{rho:.1f}" + r"$", fontsize=30)
    # ax.set_xlabel("$N$", fontsize=24)
    ax.tick_params(labelsize=25, width=1.4, length=6)
    ax.set_xticks(N_values[::2])
    ax.set_ylim(0, df["M_star"].max()+1)

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        
fig.text(0.5, -0.1, 'Network size $N$', ha='center', va='bottom', fontsize=30)

axes[0].set_ylabel("$M^*$", fontsize=30)

fig.legend(*axes[0].get_legend_handles_labels(),
           loc="upper center", ncol=4, fontsize=25, frameon=False,
           bbox_to_anchor=(0.5, 1.2))

plt.tight_layout()
save_path = PLOT_DIR / "mccallum_capacity_partial_cue.png"
fig.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()
print(f"\nSaved: {save_path}")

# %% Summary statistics
print("\n" + "=" * 60)
print("SUMMARY — McCallum M* at Different Cue Levels")
print("=" * 60)

for rho in rho_values:
    print(f"\n  rho = {rho:.1f}:")
    print(f"  {'N':>5s}  {'stable':>7s}  {'rec95':>7s}  {'rec80':>7s}  {'rec50':>7s}")
    print(f"  {'-'*5}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}")
    sub = df[df["rho"] == rho]
    for N in N_values:
        vals = {}
        for _, row in sub[sub["N"] == N].iterrows():
            vals[row["cue_level"]] = int(row["M_star"])
        print(f"  {N:5d}  {vals.get('stable',0):7d}  {vals.get('rec95',0):7d}  "
              f"{vals.get('rec80',0):7d}  {vals.get('rec50',0):7d}")

# Average drop from stability to 50% cue
print(f"\n  Average M* drop (stable -> 50% cue):")
for rho in rho_values:
    sub = df[df["rho"] == rho]
    stable_vals = sub[sub["cue_level"] == "stable"]["M_star"].values
    rec50_vals = sub[sub["cue_level"] == "rec50"]["M_star"].values
    if len(stable_vals) > 0 and len(rec50_vals) > 0:
        drop = np.mean(stable_vals) - np.mean(rec50_vals)
        pct = 100 * drop / max(np.mean(stable_vals), 1)
        print(f"    rho={rho:.1f}: {drop:.1f} patterns ({pct:.0f}%)")

# %%
