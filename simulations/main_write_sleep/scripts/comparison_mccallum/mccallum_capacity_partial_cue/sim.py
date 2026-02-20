# %% [markdown]
# # McCallum M* Capacity with Partial Cue Levels
#
# Computes M* (maximum reliable capacity) at **4 cue levels** (100%, 95%, 80%, 50%)
# across a grid of network sizes and rho values, showing that capacity drops
# sharply with harder cue requirements.
#
# **Protocol:**
# 1. No base population — patterns incorporated one by one from M=1
# 2. Incremental incorporation with Pr256 pseudorehearsal
# 3. After each step, test all M patterns at each cue level
# 4. Early stopping: failed cue levels are frozen; stability failure aborts the run
# 5. M*_s(cue) = max M where all M patterns recovered at that cue level
#
# **Output:**
# - `data/mccallum_results/mccallum_capacity_partial_cue/raw_results.csv`
# - `data/mccallum_results/mccallum_capacity_partial_cue/M_star_summary.csv`

# %% Imports
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

# %% Paths
_THIS_DIR = Path(__file__).resolve().parent if '__file__' in dir() else Path.cwd()
_SCRIPTS_DIR = _THIS_DIR.parent.parent
sys.path.insert(0, str(_SCRIPTS_DIR))

try:
    from utils import DATA_DIR
except ImportError:
    DATA_DIR = _SCRIPTS_DIR.parent / "data"

OUTPUT_DIR = DATA_DIR / "mccallum_results" / "mccallum_capacity_partial_cue"

# %% [markdown]
# ## Configuration

# %% Configuration
NETWORK_SIZES    = [50, 100, 150, 200, 250]
RHO_VALUES       = [0.0, 0.2, 0.4, 0.6, 0.8]
NUM_SEEDS        = 20
MAX_PATTERNS     = 50
THETA            = 0.9   # Success threshold for M*

MAX_PSEUDOITEMS  = 256
ETA              = 0.1
MAX_EPOCHS       = 500
ERROR_CRITERION  = 0.001
NU_H             = 0.05
SIGMA_INPUT      = 0.5
N_PROBES         = 2000

CUE_LEVELS       = [0.95, 0.80, 0.50]

SEED_BASE        = 42
# ------------------------------------------------------------------

# %% [markdown]
# ## Numba kernels

# %% Correlated pattern generation
@njit(cache=True)
def _generate_correlated_patterns(total, N, rho):
    """Parent-and-redraw: rho=0 -> uncorrelated, rho=1 -> identical."""
    patterns = np.empty((total, N), dtype=np.float64)
    parent = np.empty(N, dtype=np.float64)
    for i in range(N):
        parent[i] = 1.0 if np.random.random() < 0.5 else -1.0

    n_keep = int(round(rho * N))
    n_rand = N - n_keep

    for mu in range(total):
        for i in range(N):
            patterns[mu, i] = parent[i]
        idx = np.arange(N)
        for i in range(N):
            j = i + int(np.random.random() * (N - i))
            idx[i], idx[j] = idx[j], idx[i]
        for i in range(n_rand):
            patterns[mu, idx[i]] = 1.0 if np.random.random() < 0.5 else -1.0

    return patterns


# %% Relaxation & stability
@njit(cache=True)
def _relax_async(weights, state, max_cycles):
    N = state.shape[0]
    for _ in range(max_cycles):
        changed = False
        order = np.random.permutation(N)
        for idx in range(N):
            i = order[idx]
            h = 0.0
            for j in range(N):
                if j != i:
                    h += weights[i, j] * state[j]
            new = 1.0 if h >= 0.0 else -1.0
            if new != state[i]:
                state[i] = new
                changed = True
        if not changed:
            break
    return state


@njit(cache=True)
def _is_stable(weights, pattern, max_cycles):
    state = _relax_async(weights, pattern.copy(), max_cycles)
    for i in range(state.shape[0]):
        if state[i] != pattern[i]:
            return False
    return True


@njit(cache=True)
def _all_stable(weights, patterns, M, max_cycles):
    for mu in range(M):
        if not _is_stable(weights, patterns[mu], max_cycles):
            return False
    return True


# %% Partial cue kernels
@njit(cache=True)
def _create_partial_cue(pattern, informed_fraction):
    N = pattern.shape[0]
    n_keep = int(round(informed_fraction * N))
    cue = np.empty(N, dtype=np.float64)
    idx = np.arange(N)
    for i in range(N):
        j = i + int(np.random.random() * (N - i))
        idx[i], idx[j] = idx[j], idx[i]
    for i in range(N):
        if i < n_keep:
            cue[idx[i]] = pattern[idx[i]]
        else:
            cue[idx[i]] = 1.0 if np.random.random() < 0.5 else -1.0
    return cue


@njit(cache=True)
def _matches_pattern(state, pattern):
    match = True
    inv = True
    for i in range(state.shape[0]):
        if state[i] != pattern[i]:
            match = False
        if state[i] != -pattern[i]:
            inv = False
        if not match and not inv:
            return False
    return True


@njit(cache=True)
def _query_partial_cue(weights, pattern, informed_fraction, max_cycles):
    cue = _create_partial_cue(pattern, informed_fraction)
    state = _relax_async(weights, cue, max_cycles)
    return _matches_pattern(state, pattern)


@njit(cache=True)
def _all_recovered(weights, patterns, M, informed_fraction, max_cycles):
    for mu in range(M):
        if not _query_partial_cue(weights, patterns[mu], informed_fraction, max_cycles):
            return False
    return True


# %% Probing & noise kernels
@njit(cache=True)
def _probe_pseudoitems(weights, N, n_probes, max_items, max_cycles):
    collected = np.empty((max_items, N), dtype=np.float64)
    n_found = 0
    for _ in range(n_probes):
        if n_found >= max_items:
            break
        state = np.empty(N, dtype=np.float64)
        for i in range(N):
            state[i] = 1.0 if np.random.random() < 0.5 else -1.0
        state = _relax_async(weights, state, max_cycles)
        dup = False
        for k in range(n_found):
            match = True
            inv = True
            for i in range(N):
                if state[i] != collected[k, i]:
                    match = False
                if state[i] != -collected[k, i]:
                    inv = False
                if not match and not inv:
                    break
            if match or inv:
                dup = True
                break
        if not dup:
            collected[n_found] = state
            n_found += 1
    return collected[:n_found]


@njit(cache=True)
def _flip_noise(pattern, nu_h):
    noisy = pattern.copy()
    N = pattern.shape[0]
    n_flip = int(round(nu_h * N))
    idx = np.arange(N)
    for i in range(n_flip):
        j = i + int(np.random.random() * (N - i))
        idx[i], idx[j] = idx[j], idx[i]
    for i in range(n_flip):
        noisy[idx[i]] *= -1.0
    return noisy


# %% Training kernel
@njit(cache=True)
def _train_delta_pr(weights, patterns, n_pat, new_idx,
                    eta, max_epochs, err_crit, nu_h, sigma, N):
    smooth = 1.0
    for epoch in range(max_epochs):
        order = np.random.permutation(n_pat)
        epoch_err = 0.0
        for pos in range(n_pat):
            p = order[pos]
            target = patterns[p]
            is_new = (p == new_idx)
            if is_new:
                inp = _flip_noise(target, nu_h)
            else:
                inp = target.copy()
            errors = np.zeros(N)
            any_err = False
            for i in range(N):
                h = 0.0
                for j in range(N):
                    if j != i:
                        h += weights[i, j] * inp[j]
                if is_new:
                    h += np.random.normal(0.0, sigma)
                out = 1.0 if h >= 0.0 else -1.0
                errors[i] = target[i] - out
                if abs(errors[i]) > 0.5:
                    any_err = True
            if any_err:
                for i in range(N):
                    if abs(errors[i]) > 0.5:
                        epoch_err += abs(errors[i]) / 2.0
                        for j in range(N):
                            if i != j:
                                weights[i, j] += eta * errors[i] * inp[j]
        smooth = smooth * 0.9 + epoch_err * 0.1
        if smooth < err_crit:
            return weights, epoch + 1
    return weights, max_epochs


# %% JIT warmup
print("Compiling Numba kernels ...", end=" ", flush=True)
t0 = time.time()
_w = np.zeros((5, 5)); _s = np.ones(5); _p = np.ones((2, 5))
_generate_correlated_patterns(2, 5, 0.5)
_relax_async(_w, _s.copy(), 2)
_is_stable(_w, _s, 2)
_all_stable(_w, _p, 2, 2)
_create_partial_cue(_s, 0.5)
_matches_pattern(_s, _s)
_query_partial_cue(_w, _s, 0.5, 2)
_all_recovered(_w, _p, 2, 0.5, 2)
_probe_pseudoitems(_w, 5, 3, 2, 2)
_flip_noise(_s, 0.2)
_train_delta_pr(_w, _p, 2, 1, 0.1, 2, 0.001, 0.05, 0.5, 5)
del _w, _s, _p
print(f"done ({time.time()-t0:.1f}s)")

# %% [markdown]
# ## Run experiment

# %% Single run
def run_single(N, rho, seed):
    """Run Pr256 pseudorehearsal for one (N, rho, seed) with early stopping.

    No base population — every pattern is incorporated one by one via
    pseudorehearsal, starting from an empty network (M=1, 2, 3, ...).

    Early stopping logic:
    - Track which cue levels are still 'alive' (haven't failed yet).
    - When a cue level fails (not all M patterns recovered), freeze its M*_s
      and stop testing it on subsequent steps.
    - When stability (100%) fails, no harder cue can work either — abort the
      entire run for this seed.
    """
    np.random.seed(seed)

    max_cycles = 4 * N
    patterns = _generate_correlated_patterns(MAX_PATTERNS, N, rho)
    weights = np.zeros((N, N))

    rows = []

    # alive[i] = True means we still test this cue level
    # Order: 0=stable(100%), 1=rec95, 2=rec80, 3=rec50
    alive = [True, True, True, True]
    cue_fractions = [None, 0.95, 0.80, 0.50]  # None = stability check

    def evaluate(M):
        """Evaluate alive cue levels, kill those that fail. Returns results tuple."""
        results = [0, 0, 0, 0]

        # First check stability — if it fails, all fail
        if alive[0]:
            if _all_stable(weights, patterns, M, max_cycles):
                results[0] = 1
            else:
                alive[0] = False
                for k in range(1, 4):
                    alive[k] = False
                return tuple(results)

        # Check partial cues (only alive ones, easiest first)
        for k in range(1, 4):
            if alive[k]:
                if _all_recovered(weights, patterns, M, cue_fractions[k], max_cycles):
                    results[k] = 1
                else:
                    alive[k] = False

        return tuple(results)

    # --- Incremental incorporation from M=1 ---
    for M in range(1, MAX_PATTERNS + 1):
        if not any(alive):
            break

        # Probe for pseudoitems (empty network at M=1 yields ~1 attractor)
        pseudos = _probe_pseudoitems(weights, N, N_PROBES, MAX_PSEUDOITEMS, max_cycles)

        # Training set: pseudoitems + new pattern
        train_set = np.vstack((pseudos, patterns[M-1:M]))
        n_train = train_set.shape[0]

        weights, _ = _train_delta_pr(
            weights, train_set, n_train, n_train - 1,
            ETA, MAX_EPOCHS, ERROR_CRITERION, NU_H, SIGMA_INPUT, N)

        r = evaluate(M)
        rows.append((N, rho, seed, M, r[0], r[1], r[2], r[3], pseudos.shape[0]))

    return rows


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


def compute_M_star_s(rows_for_seed, cue_col_idx):
    """Compute M*_s = max M where the all_X column == 1 for one seed's rows."""
    best = 0
    for row in rows_for_seed:
        M = row[3]       # M is at index 3
        val = row[cue_col_idx]
        if val == 1:
            best = M
    return best


# %% Run all
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
raw_csv = OUTPUT_DIR / "raw_results.csv"
summary_csv = OUTPUT_DIR / "M_star_summary.csv"

total_runs = len(NETWORK_SIZES) * len(RHO_VALUES) * NUM_SEEDS
print(f"\n{'='*60}")
print(f"McCallum M* Capacity — Partial Cue Levels")
print(f"  Network sizes: {NETWORK_SIZES}")
print(f"  Rho values:    {RHO_VALUES}")
print(f"  Seeds:         {NUM_SEEDS}")
print(f"  Max patterns:  {MAX_PATTERNS}")
print(f"  Cue levels:    [1.00, 0.95, 0.80, 0.50]")
print(f"  Total runs:    {total_runs}")
print(f"{'='*60}\n")

all_rows = []
run_count = 0
t_global = time.time()

for N in NETWORK_SIZES:
    for rho in RHO_VALUES:
        for s in range(NUM_SEEDS):
            seed = SEED_BASE + s
            t0 = time.time()
            rows = run_single(N, rho, seed)
            all_rows.extend(rows)
            run_count += 1
            elapsed = time.time() - t0
            # Last row has final M stats
            last = rows[-1]
            print(f"  [{run_count:3d}/{total_runs}] N={N:3d} rho={rho:.1f} seed={seed:2d}: "
                  f"stable={last[4]} rec95={last[5]} rec80={last[6]} rec50={last[7]} "
                  f"({elapsed:.1f}s)")

# %% Save raw results
df_raw = pd.DataFrame(all_rows,
                       columns=["network_size", "rho", "seed", "M",
                                "all_stable", "all_rec95", "all_rec80", "all_rec50",
                                "num_pseudo"])
df_raw.to_csv(raw_csv, index=False)
print(f"\nRaw results saved: {raw_csv} ({len(df_raw)} rows)")

# %% Compute M* summary
# Column indices in the rows tuples: 4=stable, 5=rec95, 6=rec80, 7=rec50
CUE_MAP = {
    "stable":  4,
    "rec95":   5,
    "rec80":   6,
    "rec50":   7,
}

summary_rows = []
for N in NETWORK_SIZES:
    for rho in RHO_VALUES:
        # Gather rows per seed
        seed_rows = {}
        for row in all_rows:
            if row[0] == N and row[1] == rho:
                s = row[2]
                if s not in seed_rows:
                    seed_rows[s] = []
                seed_rows[s].append(row)

        for cue_label, col_idx in CUE_MAP.items():
            M_star_s_list = []
            for s, srows in seed_rows.items():
                m_star_s = compute_M_star_s(srows, col_idx)
                M_star_s_list.append(m_star_s)

            M_star = compute_M_star(M_star_s_list, THETA)
            mean_ms = np.mean(M_star_s_list) if M_star_s_list else 0
            std_ms = np.std(M_star_s_list) if M_star_s_list else 0

            summary_rows.append({
                "N": N,
                "rho": rho,
                "cue_level": cue_label,
                "M_star": M_star,
                "mean_M_star": round(mean_ms, 2),
                "std_M_star": round(std_ms, 2),
                "num_sims": len(M_star_s_list),
            })

df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(summary_csv, index=False)
print(f"M* summary saved: {summary_csv}")

# %% Print summary table
print(f"\n{'='*60}")
print("M* Summary")
print(f"{'='*60}")

for rho in RHO_VALUES:
    print(f"\n  rho = {rho:.1f}:")
    print(f"  {'N':>5s}  {'stable':>7s}  {'rec95':>7s}  {'rec80':>7s}  {'rec50':>7s}")
    print(f"  {'-'*5}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}")
    for N in NETWORK_SIZES:
        vals = {}
        for _, row in df_summary[(df_summary["N"] == N) & (df_summary["rho"] == rho)].iterrows():
            vals[row["cue_level"]] = row["M_star"]
        print(f"  {N:5d}  {vals.get('stable',0):7d}  {vals.get('rec95',0):7d}  "
              f"{vals.get('rec80',0):7d}  {vals.get('rec50',0):7d}")

elapsed_total = time.time() - t_global
print(f"\nTotal elapsed: {elapsed_total:.0f}s ({elapsed_total/60:.1f}min)")
