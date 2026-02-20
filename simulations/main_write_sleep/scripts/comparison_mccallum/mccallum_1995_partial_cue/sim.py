# %% [markdown]
# # McCallum 1995 — Partial Cue Experiment
#
# Runs **only the Pr256 condition** from McCallum's pseudorehearsal protocol
# and, at each incorporation step, tests pattern recovery with partial cues
# at four levels: 95%, 90%, 80%, 50%.
#
# **Protocol:**
# 1. Base population BP=5 trained (no noise)
# 2. Incremental incorporation with Pr256 pseudorehearsal
# 3. After each step, test all M patterns:
#    - Stability (100% cue = fixed-point check)
#    - Recovery from 95% / 90% / 80% / 50% partial cues
#
# **Output:** `data/mccallum_results/mccallum_1995_partial_cue/pr256_partial_cue.csv`
# with columns: `trial, M, stable, recovered_95, recovered_90, recovered_80, recovered_50, pseudo`

# %% Imports
import sys
import time
from pathlib import Path

import numpy as np
from numba import njit

# %% Paths
_THIS_DIR = Path(__file__).resolve().parent if '__file__' in dir() else Path.cwd()
_SCRIPTS_DIR = _THIS_DIR.parent.parent
sys.path.insert(0, str(_SCRIPTS_DIR))

try:
    from utils import DATA_DIR
except ImportError:
    DATA_DIR = _SCRIPTS_DIR.parent / "data"

OUTPUT_DIR = DATA_DIR / "mccallum_results" / "mccallum_1995_partial_cue"

# %% [markdown]
# ## Configuration

# %% Configuration
N_SIZE           = 100
BASE_POP         = 5
MAX_NEW          = 95
MAX_PSEUDOITEMS  = 256
ETA              = 0.1
MAX_EPOCHS       = 500
ERROR_CRITERION  = 0.001
NU_H             = 0.05
SIGMA_INPUT      = 0.5
N_PROBES         = 2000
MAX_CYCLES       = 4 * N_SIZE   # relaxation budget

CUE_LEVELS       = [0.95, 0.90, 0.80, 0.50]

N_TRIALS         = 15      # ← edit this
SEED             = 42      # set to None for random
# ──────────────────────────────────────────────────────────────────

# %% [markdown]
# ## Numba kernels (shared with mccallum_1995)

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
def _count_stable(weights, patterns, M, max_cycles):
    count = 0
    for mu in range(M):
        if _is_stable(weights, patterns[mu], max_cycles):
            count += 1
    return count


# %% Partial cue kernels
@njit(cache=True)
def _create_partial_cue(pattern, informed_fraction):
    """Keep informed_fraction of bits from pattern, randomise the rest."""
    N = pattern.shape[0]
    n_keep = int(round(informed_fraction * N))
    cue = np.empty(N, dtype=np.float64)

    # Shuffle indices to pick which bits to keep
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
    """Check if state matches pattern or its inverse."""
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
    """Create a partial cue, relax, and check if it recovers the pattern."""
    cue = _create_partial_cue(pattern, informed_fraction)
    state = _relax_async(weights, cue, max_cycles)
    return _matches_pattern(state, pattern)


@njit(cache=True)
def _count_recovered(weights, patterns, M, informed_fraction, max_cycles):
    """Count how many of the first M patterns are recovered from partial cues."""
    count = 0
    for mu in range(M):
        if _query_partial_cue(weights, patterns[mu], informed_fraction, max_cycles):
            count += 1
    return count


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


# %% Training kernels
@njit(cache=True)
def _train_base(weights, patterns, n_base, eta, max_epochs, err_crit, N):
    smooth = 1.0
    for epoch in range(max_epochs):
        order = np.random.permutation(n_base)
        epoch_err = 0.0
        for pos in range(n_base):
            target = patterns[order[pos]]
            errors = np.zeros(N)
            any_err = False
            for i in range(N):
                h = 0.0
                for j in range(N):
                    if j != i:
                        h += weights[i, j] * target[j]
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
                                weights[i, j] += eta * errors[i] * target[j]
        smooth = smooth * 0.9 + epoch_err * 0.1
        if smooth < err_crit:
            return weights, epoch + 1
    return weights, max_epochs


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


@njit(cache=True)
def _train_delta_naive(weights, target, eta, max_epochs, err_crit, N):
    """Delta learning on a single pattern — no noise at all."""
    smooth = 1.0
    for epoch in range(max_epochs):
        inp = target.copy()
        errors = np.zeros(N)
        any_err = False
        for i in range(N):
            h = 0.0
            for j in range(N):
                if j != i:
                    h += weights[i, j] * inp[j]
            out = 1.0 if h >= 0.0 else -1.0
            errors[i] = target[i] - out
            if abs(errors[i]) > 0.5:
                any_err = True
        if any_err:
            epoch_err = 0.0
            for i in range(N):
                if abs(errors[i]) > 0.5:
                    epoch_err += abs(errors[i]) / 2.0
                    for j in range(N):
                        if i != j:
                            weights[i, j] += eta * errors[i] * inp[j]
            smooth = smooth * 0.9 + epoch_err * 0.1
        else:
            smooth = smooth * 0.9
        if smooth < err_crit:
            return weights, epoch + 1
    return weights, max_epochs


# %% JIT warmup
print("Compiling Numba kernels ...", end=" ", flush=True)
t0 = time.time()
_w = np.zeros((5, 5)); _s = np.ones(5); _p = np.ones((2, 5))
_relax_async(_w, _s.copy(), 2)
_is_stable(_w, _s, 2)
_count_stable(_w, _p, 2, 2)
_create_partial_cue(_s, 0.5)
_matches_pattern(_s, _s)
_query_partial_cue(_w, _s, 0.5, 2)
_count_recovered(_w, _p, 2, 0.5, 2)
_probe_pseudoitems(_w, 5, 3, 2, 2)
_flip_noise(_s, 0.2)
_train_base(_w, _p, 2, 0.1, 2, 0.001, 5)
_train_delta_pr(_w, _p, 2, 1, 0.1, 2, 0.001, 0.05, 0.5, 5)
_train_delta_naive(_w, _s, 0.1, 2, 0.001, 5)
del _w, _s, _p
print(f"done ({time.time()-t0:.1f}s)")

# %% [markdown]
# ## Run experiment

# %% Main runner
def run_pr256_partial_cue(seed=None, verbose=False):
    """Run Pr256 pseudorehearsal and measure partial-cue recovery at each M."""
    if seed is not None:
        np.random.seed(seed)

    N = N_SIZE
    r = MAX_CYCLES
    total = BASE_POP + MAX_NEW
    patterns = np.random.choice([-1.0, 1.0], size=(total, N))
    weights = np.zeros((N, N))

    # Lists for each metric
    M_list, stable_list, pseudo_list = [], [], []
    rec = {f: [] for f in CUE_LEVELS}

    # --- Train base population ---
    weights, _ = _train_base(weights, patterns[:BASE_POP].copy(),
                             BASE_POP, ETA, MAX_EPOCHS, ERROR_CRITERION, N)

    ns = _count_stable(weights, patterns, BASE_POP, r)
    M_list.append(BASE_POP)
    stable_list.append(ns)
    pseudo_list.append(0)
    for f in CUE_LEVELS:
        rec[f].append(_count_recovered(weights, patterns, BASE_POP, f, r))

    # --- Incremental incorporation ---
    for step in range(MAX_NEW):
        M = BASE_POP + step + 1

        # Probe for pseudoitems
        pseudos = _probe_pseudoitems(weights, N, N_PROBES, MAX_PSEUDOITEMS, r)

        # Build training set: pseudoitems + new pattern
        train_set = np.vstack((pseudos, patterns[M-1:M]))
        n_train = train_set.shape[0]

        # Train
        weights, _ = _train_delta_pr(
            weights, train_set, n_train, n_train - 1,
            ETA, MAX_EPOCHS, ERROR_CRITERION, NU_H, SIGMA_INPUT, N)

        # Evaluate stability
        ns = _count_stable(weights, patterns, M, r)
        M_list.append(M)
        stable_list.append(ns)
        pseudo_list.append(pseudos.shape[0])

        # Evaluate partial-cue recovery at each level
        for f in CUE_LEVELS:
            rec[f].append(_count_recovered(weights, patterns, M, f, r))

        if verbose and (step < 5 or (step+1) % 10 == 0 or step == MAX_NEW - 1):
            r50 = rec[0.50][-1]
            r95 = rec[0.95][-1]
            print(f"  M={M:3d}: stable={ns:3d}  rec95={r95:3d}  rec50={r50:3d}  "
                  f"pseudo={pseudos.shape[0]:3d}")

    return M_list, stable_list, rec, pseudo_list


# %% Run all trials and save CSV
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
csv_path = OUTPUT_DIR / "pr256_partial_cue.csv"

print(f"\n{'='*60}")
print(f"Pr256 Partial Cue — {N_TRIALS} trials, N={N_SIZE}")
print(f"Cue levels: {CUE_LEVELS}")
print(f"{'='*60}")

with open(csv_path, "w") as f:
    f.write("trial,M,stable,recovered_95,recovered_90,recovered_80,recovered_50,pseudo\n")

    for t in range(N_TRIALS):
        t0 = time.time()
        trial_seed = (SEED + t) if SEED is not None else None
        M_list, stable_list, rec, pseudo_list = run_pr256_partial_cue(
            seed=trial_seed, verbose=False)

        for i, M in enumerate(M_list):
            f.write(f"{t},{M},{stable_list[i]},"
                    f"{rec[0.95][i]},{rec[0.90][i]},{rec[0.80][i]},{rec[0.50][i]},"
                    f"{pseudo_list[i]}\n")

        elapsed = time.time() - t0
        print(f"  Trial {t+1:2d}/{N_TRIALS}: "
              f"stable={stable_list[-1]:3d} rec50={rec[0.50][-1]:3d} "
              f"({elapsed:.1f}s)")

print(f"\nSaved: {csv_path}")
print(f"{'='*60}")


# %% [markdown]
# ## Naive baseline (without rehearsal)

# %% Naive runner
def run_naive_partial_cue(seed=None, verbose=False):
    """Naive iterative delta (no rehearsal) with partial-cue recovery."""
    if seed is not None:
        np.random.seed(seed)

    N = N_SIZE
    r = MAX_CYCLES
    total = BASE_POP + MAX_NEW
    patterns = np.random.choice([-1.0, 1.0], size=(total, N))
    weights = np.zeros((N, N))

    M_list, stable_list, pseudo_list = [], [], []
    rec = {f: [] for f in CUE_LEVELS}

    # --- Train base population ---
    weights, _ = _train_base(weights, patterns[:BASE_POP].copy(),
                             BASE_POP, ETA, MAX_EPOCHS, ERROR_CRITERION, N)

    ns = _count_stable(weights, patterns, BASE_POP, r)
    M_list.append(BASE_POP)
    stable_list.append(ns)
    pseudo_list.append(0)
    for f in CUE_LEVELS:
        rec[f].append(_count_recovered(weights, patterns, BASE_POP, f, r))

    # --- Incremental incorporation (no rehearsal) ---
    for step in range(MAX_NEW):
        M = BASE_POP + step + 1

        # Train only on the new pattern — no pseudorehearsal
        weights, _ = _train_delta_naive(
            weights, patterns[M-1], ETA, MAX_EPOCHS, ERROR_CRITERION, N)

        ns = _count_stable(weights, patterns, M, r)
        M_list.append(M)
        stable_list.append(ns)
        pseudo_list.append(0)

        for f in CUE_LEVELS:
            rec[f].append(_count_recovered(weights, patterns, M, f, r))

        if verbose and (step < 5 or (step+1) % 10 == 0 or step == MAX_NEW - 1):
            r50 = rec[0.50][-1]
            r95 = rec[0.95][-1]
            print(f"  M={M:3d}: stable={ns:3d}  rec95={r95:3d}  rec50={r50:3d}")

    return M_list, stable_list, rec, pseudo_list


# %% Run naive trials and save CSV
csv_path_naive = OUTPUT_DIR / "naive_partial_cue.csv"

print(f"\n{'='*60}")
print(f"Naive (no rehearsal) Partial Cue — {N_TRIALS} trials, N={N_SIZE}")
print(f"Cue levels: {CUE_LEVELS}")
print(f"{'='*60}")

with open(csv_path_naive, "w") as f:
    f.write("trial,M,stable,recovered_95,recovered_90,recovered_80,recovered_50,pseudo\n")

    for t in range(N_TRIALS):
        t0 = time.time()
        trial_seed = (SEED + t) if SEED is not None else None
        M_list, stable_list, rec, pseudo_list = run_naive_partial_cue(
            seed=trial_seed, verbose=False)

        for i, M in enumerate(M_list):
            f.write(f"{t},{M},{stable_list[i]},"
                    f"{rec[0.95][i]},{rec[0.90][i]},{rec[0.80][i]},{rec[0.50][i]},"
                    f"{pseudo_list[i]}\n")

        elapsed = time.time() - t0
        print(f"  Trial {t+1:2d}/{N_TRIALS}: "
              f"stable={stable_list[-1]:3d} rec50={rec[0.50][-1]:3d} "
              f"({elapsed:.1f}s)")

print(f"\nSaved: {csv_path_naive}")
print(f"{'='*60}")
