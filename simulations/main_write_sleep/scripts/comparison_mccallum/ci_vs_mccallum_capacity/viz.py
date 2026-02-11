# %% [markdown]
# # CI vs McCallum M* Capacity — Multi-Cue Comparison
#
# 4-row x 5-column figure:
# - Rows (top->bottom): Stability (100%), 95% cue, 80% cue, 50% cue
# - Columns (left->right): rho = 0.0, 0.2, 0.4, 0.6, 0.8
# - Each subplot: M* vs N with two lines (CI + McCallum)
#
# **Data sources:**
# - McCallum: `data/mccallum_results/mccallum_capacity_partial_cue/M_star_summary.csv`
# - CI: `data/mccallum_results/ci_vs_mccallum_capacity/M_star_summary.csv`

# %% Imports
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

_THIS_DIR = Path(__file__).resolve().parent if '__file__' in dir() else Path.cwd()
_SCRIPTS_DIR = _THIS_DIR.parent.parent
sys.path.insert(0, str(_SCRIPTS_DIR))

try:
    from utils import DATA_DIR, SCRIPT_DIR
except ImportError:
    DATA_DIR   = _SCRIPTS_DIR.parent / "data"
    SCRIPT_DIR = _SCRIPTS_DIR

MCCALLUM_CSV = DATA_DIR / "mccallum_results" / "mccallum_capacity_partial_cue" / "M_star_summary.csv"
CI_CSV       = DATA_DIR / "mccallum_results" / "ci_vs_mccallum_capacity" / "M_star_summary.csv"
PLOT_DIR     = SCRIPT_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)

# %% Style
plt.rcParams.update({
    'font.size': 22,
    'axes.labelsize': 28,
    'axes.titlesize': 28,
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'legend.fontsize': 22,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'font.family': 'sans-serif',
    'axes.linewidth': 1.6,
    'xtick.major.width': 1.6,
    'ytick.major.width': 1.6,
})

# %% Load data
assert MCCALLUM_CSV.exists(), f"McCallum data not found: {MCCALLUM_CSV}\nRun mccallum_capacity_partial_cue/sim.py first."
assert CI_CSV.exists(), f"CI data not found: {CI_CSV}\nRun ci_vs_mccallum_capacity/sim.py first."

df_mc = pd.read_csv(MCCALLUM_CSV)
df_ci = pd.read_csv(CI_CSV)

print(f"McCallum: {len(df_mc)} rows")
print(f"CI:       {len(df_ci)} rows")

# %% Configuration
RHO_VALUES = sorted(set(df_mc["rho"].unique()) & set(df_ci["rho"].unique()))
N_VALUES_MC = sorted(df_mc["N"].unique())
N_VALUES_CI = sorted(df_ci["N"].unique())

CUE_LEVELS = ["stable", "rec95", "rec80", "rec50"]
CUE_LABELS = {
    "stable": "Stability (100%)",
    "rec95":  "95% cue",
    "rec80":  "80% cue",
    "rec50":  "50% cue",
}

print(f"Rho values:       {RHO_VALUES}")
print(f"N values (McC):   {N_VALUES_MC}")
print(f"N values (CI):    {N_VALUES_CI}")

# %% Method styles
METHODS = [
    # (df, label, color, marker, lw)
    (df_ci, "CI",       "#2C3E50", "s", 2.5),
    (df_mc, "McCallum", "#E67E22", "o", 2.5),
]

# %% Compute global y-limit (max M* across both datasets)
global_max = max(
    df_mc["M_star"].max() if len(df_mc) > 0 else 0,
    df_ci["M_star"].max() if len(df_ci) > 0 else 0,
)
y_max = int(np.ceil(global_max * 1.1))  # 10% headroom

# %% Plot: 4 rows x 5 columns
n_rows = len(CUE_LEVELS)
n_cols = len(RHO_VALUES)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows),
                          sharey=True, sharex=True)

for row_idx, cue_level in enumerate(CUE_LEVELS):
    for col_idx, rho in enumerate(RHO_VALUES):
        ax = axes[row_idx, col_idx]

        for df, label, color, marker, lw in METHODS:
            sub = df[(df["rho"] == rho) & (df["cue_level"] == cue_level)].sort_values("N")
            if len(sub) == 0:
                continue

            x = sub["N"].values
            y = sub["M_star"].values

            ax.plot(x, y, color=color, lw=lw, marker=marker, markersize=10,
                    label=label if (row_idx == 0 and col_idx == 0) else None,
                    zorder=3)

        # Titles: rho on top row only
        if row_idx == 0:
            ax.set_title(r"$\rho = " + f"{rho:.1f}" + r"$", fontsize=30)

        # Row label on rightmost column
        if col_idx == n_cols - 1:
            ax.annotate(CUE_LABELS[cue_level], xy=(1.05, 0.5),
                        xycoords='axes fraction', fontsize=24,
                        ha='left', va='center', rotation=-90)

        ax.tick_params(labelsize=24, width=1.6, length=7)
        ax.set_ylim(0, y_max)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        for spine in ax.spines.values():
            spine.set_linewidth(1.4)

# Single shared axis labels
fig.supxlabel("$N$", fontsize=30, y=-0.01)
fig.supylabel("$M^*$", fontsize=30, x=-0.01)

# Single legend at top
fig.legend(*axes[0, 0].get_legend_handles_labels(),
           loc="upper center", ncol=2, fontsize=24, frameon=False,
           bbox_to_anchor=(0.5, 1.03))

plt.tight_layout()
save_path = PLOT_DIR / "ci_vs_mccallum_capacity.png"
fig.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()
print(f"\nSaved: {save_path}")

# %% Summary statistics
print("\n" + "=" * 60)
print("SUMMARY — CI vs McCallum M* Comparison")
print("=" * 60)

for cue_level in CUE_LEVELS:
    print(f"\n  {CUE_LABELS[cue_level]}:")
    print(f"  {'rho':>5s}  {'N':>5s}  {'CI':>5s}  {'McC':>5s}  {'diff':>5s}")
    print(f"  {'-'*5}  {'-'*5}  {'-'*5}  {'-'*5}  {'-'*5}")
    for rho in RHO_VALUES:
        # Use the intersection of N values
        N_common = sorted(set(N_VALUES_MC) & set(N_VALUES_CI))
        for N in N_common:
            ci_row = df_ci[(df_ci["N"] == N) & (df_ci["rho"] == rho) & (df_ci["cue_level"] == cue_level)]
            mc_row = df_mc[(df_mc["N"] == N) & (df_mc["rho"] == rho) & (df_mc["cue_level"] == cue_level)]
            ci_val = int(ci_row["M_star"].iloc[0]) if len(ci_row) > 0 else 0
            mc_val = int(mc_row["M_star"].iloc[0]) if len(mc_row) > 0 else 0
            diff = ci_val - mc_val
            print(f"  {rho:5.1f}  {N:5d}  {ci_val:5d}  {mc_val:5d}  {diff:+5d}")
