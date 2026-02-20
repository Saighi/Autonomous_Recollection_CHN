# %% [markdown]
# # McCallum 1995 — Partial Cue with Varying Correlation (rho)
#
# Loads CSV data produced by `sim.py` and generates a publication-ready
# figure showing pattern recovery (80% cue) as a function of patterns
# learned (M) at different correlation levels rho.
#
# **Expected data:** `data/mccallum_results/mccallum_1995_partial_cue_rho/pr256_rho_{rho}.csv`

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

DATA_PATH = DATA_DIR / "mccallum_results" / "mccallum_1995_partial_cue_rho"
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

# %% Load CSVs for each rho
RHO_VALUES = [0.0, 0.2, 0.4, 0.6, 0.8]

frames = {}
for rho in RHO_VALUES:
    csv_path = DATA_PATH / f"pr256_rho_{rho:.1f}.csv"
    if csv_path.exists():
        frames[rho] = pd.read_csv(csv_path)
        n_trials = frames[rho]["trial"].nunique()
        print(f"  rho={rho:.1f}: {len(frames[rho])} rows, {n_trials} trials")
    else:
        print(f"  rho={rho:.1f}: NOT FOUND ({csv_path})")

assert len(frames) > 0, "No data found! Run sim.py first."

# %% Load naive CSVs for each rho
naive_frames = {}
for rho in RHO_VALUES:
    csv_path = DATA_PATH / f"naive_rho_{rho:.1f}.csv"
    if csv_path.exists():
        naive_frames[rho] = pd.read_csv(csv_path)
        n_trials = naive_frames[rho]["trial"].nunique()
        print(f"  naive rho={rho:.1f}: {len(naive_frames[rho])} rows, {n_trials} trials")
    else:
        print(f"  naive rho={rho:.1f}: NOT FOUND ({csv_path})")

# %% Aggregate mean/std across trials for each rho
stats = {}
for rho, df in frames.items():
    agg = df.groupby("M").agg(
        stable_mean=("stable", "mean"),
        stable_std=("stable", "std"),
        rec80_mean=("recovered_80", "mean"),
        rec80_std=("recovered_80", "std"),
        pseudo_mean=("pseudo", "mean"),
    ).reset_index()
    for col in agg.columns:
        if col.endswith("_std"):
            agg[col] = agg[col].fillna(0.0)
    stats[rho] = agg

# %% Aggregate naive stats
naive_stats = {}
for rho, df in naive_frames.items():
    agg = df.groupby("M").agg(
        stable_mean=("stable", "mean"),
        stable_std=("stable", "std"),
        rec80_mean=("recovered_80", "mean"),
        rec80_std=("recovered_80", "std"),
    ).reset_index()
    for col in agg.columns:
        if col.endswith("_std"):
            agg[col] = agg[col].fillna(0.0)
    naive_stats[rho] = agg

# %% [markdown]
# ## Main figure: Recovery (80% cue) vs M at different rho

# %% Define line styles (gradient from easy to hard correlation)
LINES = {
    0.0: {"color": "#2C3E50", "ls": "-",  "lw": 2.5, "label": r"$\rho = 0.0$"},
    0.2: {"color": "#2980B9", "ls": "-",  "lw": 2.2, "label": r"$\rho = 0.2$"},
    0.4: {"color": "#E67E22", "ls": "-",  "lw": 2.2, "label": r"$\rho = 0.4$"},
    0.6: {"color": "#C0392B", "ls": "-",  "lw": 2.2, "label": r"$\rho = 0.6$"},
    0.8: {"color": "#8E44AD", "ls": "-",  "lw": 2.2, "label": r"$\rho = 0.8$"},
}

# %% Plot — Recovery (80% cue)
fig, ax = plt.subplots(figsize=(8, 6))

for rho in RHO_VALUES:
    if rho not in stats:
        continue
    s = stats[rho]
    st = LINES[rho]
    M = s["M"].values
    y_mean = s["rec80_mean"].values
    y_std  = s["rec80_std"].values

    ax.plot(M, y_mean, color=st["color"], ls=st["ls"], lw=st["lw"],
            label=st["label"], zorder=3)
    ax.fill_between(M, y_mean - y_std, y_mean + y_std,
                    color=st["color"], alpha=0.15, zorder=2)

# --- Naive baselines (dashed black) ---
for rho in RHO_VALUES:
    if rho not in naive_stats:
        continue
    s = naive_stats[rho]
    M_n = s["M"].values
    y_mean = s["rec80_mean"].values
    y_std  = s["rec80_std"].values
    lbl = r"Without rehearsal" if rho == RHO_VALUES[0] else None
    ax.plot(M_n, y_mean, color="black", ls="--", lw=2.0,
            label=lbl, zorder=2)
    ax.fill_between(M_n, y_mean - y_std, y_mean + y_std,
                    color="black", alpha=0.08, zorder=1)

ax.set_xlabel("Patterns learned ($M$)", fontsize=20)
ax.set_ylabel("Recovered patterns (80% cue)", fontsize=20)
M_ref = stats[next(iter(stats))]["M"].values
ax.set_xlim(M_ref[0] - 1, M_ref[-1] + 2)
ax.set_ylim(0, None)
ax.tick_params(labelsize=16, width=1.2, length=5)

ax.legend(loc="upper left", fontsize=14, frameon=False)

for spine in ax.spines.values():
    spine.set_linewidth(1.2)

plt.tight_layout()
save_path = PLOT_DIR / "mccallum_1995_partial_cue_rho_recovery.png"
fig.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()
print(f"Saved: {save_path}")

# %% [markdown]
# ## Stability figure

# %% Plot — Stability (100% cue)
fig, ax = plt.subplots(figsize=(8, 6))

for rho in RHO_VALUES:
    if rho not in stats:
        continue
    s = stats[rho]
    st = LINES[rho]
    M = s["M"].values
    y_mean = s["stable_mean"].values
    y_std  = s["stable_std"].values

    ax.plot(M, y_mean, color=st["color"], ls=st["ls"], lw=st["lw"],
            label=st["label"], zorder=3)
    # ax.fill_between(M, y_mean - y_std, y_mean + y_std,
    #                 color=st["color"], alpha=0.25, zorder=2)

# # --- Naive baselines (dashed black) ---
# for rho in RHO_VALUES:
#     if rho not in naive_stats:
#         continue
#     s = naive_stats[rho]
#     M_n = s["M"].values
#     y_mean = s["stable_mean"].values
#     y_std  = s["stable_std"].values
#     lbl = r"Without rehearsal" if rho == RHO_VALUES[0] else None
#     ax.plot(M_n, y_mean, color="black", ls="--", lw=2.0,
#             label=lbl, zorder=2)
#     ax.fill_between(M_n, y_mean - y_std, y_mean + y_std,
#                     color="black", alpha=0.08, zorder=1)

ax.set_xlabel("Patterns learned ($M$)", fontsize=20)
ax.set_ylabel("Stable patterns", fontsize=20)
ax.set_xlim(M_ref[0] - 1, M_ref[-1] + 2)
ax.set_ylim(0, None)
ax.tick_params(labelsize=16, width=1.2, length=5)

ax.legend(loc="upper left", fontsize=14, frameon=False)

for spine in ax.spines.values():
    spine.set_linewidth(1.2)

plt.tight_layout()
save_path = PLOT_DIR / "mccallum_1995_partial_cue_rho_stability.png"
fig.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()
print(f"Saved: {save_path}")

# %% [markdown]
# ## Summary statistics

# %% Summary
print("\n" + "=" * 60)
print("SUMMARY — Pr256 80% Cue, Varying Correlation")
print("=" * 60)

for rho in RHO_VALUES:
    if rho not in stats:
        continue
    s = stats[rho]
    M = s["M"].values
    y_rec = s["rec80_mean"].values
    y_stb = s["stable_mean"].values
    n_trials = frames[rho]["trial"].nunique()

    peak_rec = np.argmax(y_rec)
    peak_stb = np.argmax(y_stb)

    print(f"\n  rho={rho:.1f} ({n_trials} trials):")
    print(f"    Recovery (80% cue):")
    print(f"      At M={M[-1]:3d}: {y_rec[-1]:.1f}")
    print(f"      Peak:  {y_rec[peak_rec]:.1f} at M={M[peak_rec]}")
    print(f"    Stability:")
    print(f"      At M={M[-1]:3d}: {y_stb[-1]:.1f}")
    print(f"      Peak:  {y_stb[peak_stb]:.1f} at M={M[peak_stb]}")

    for target_M in [15, 30, 50]:
        hits = [i for i, x in enumerate(M) if x >= target_M]
        if hits:
            print(f"    At M={target_M:3d}: stable={y_stb[hits[0]]:.1f}  "
                  f"rec80={y_rec[hits[0]]:.1f}")
