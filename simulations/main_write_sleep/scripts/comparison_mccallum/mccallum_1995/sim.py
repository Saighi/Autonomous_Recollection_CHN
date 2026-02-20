# %% [markdown]
# # McCallum 1995 — Simulation
#
# Runs all conditions from McCallum's Figure 4.23 comparison and saves
# per-trial time-series to CSV in `data/mccallum_results/mccallum_1995/`.
#
# **Conditions:**
# - Pr100 / Pr256 / Pr512 — pseudorehearsal with varying pseudoitem caps
# - Delta (hetero) — pure delta learning, heteroassociative noise
# - Delta (Gaussian) — pure delta learning, Gaussian input noise
#
# **Protocol (asymmetric, corrected):**
# 1. Base population BP=5 trained first (no noise)
# 2. Pp = 2000 probes, r = 4N relaxation, inverse rejected
# 3. Heteroassociative noise on new pattern only
# 4. Asymmetric weight updates (McCallum Eq. 2.7)
# 5. Synchronous delta learning
#
# Output: `data/mccallum_results/mccallum_1995/{condition}.csv`
# with columns: `condition, trial, M, stable, pseudo`

# %% Imports
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
from numba import njit

# %% Paths
_THIS_DIR = Path(__file__).resolve().parent if '__file__' in dir() else Path.cwd()
_SCRIPTS_DIR = _THIS_DIR.parent.parent
sys.path.insert(0, str(_SCRIPTS_DIR))

# Try to import DATA_DIR from project utils; fall back to relative path
try:
    from utils import DATA_DIR
except ImportError:
    DATA_DIR = _SCRIPTS_DIR.parent / "data"

OUTPUT_DIR = DATA_DIR / "mccallum_results" / "mccallum_1995"

# %% [markdown]
# ## Configuration

# %% Configuration
@dataclass
class McCallumConfig:
    """All tuneable parameters for the McCallum 1995 protocol."""
    network_size:    int   = 100
    base_pop:        int   = 5
    max_new:         int   = 95
    n_probes:        int   = 2000
    max_pseudoitems: int   = 256
    eta:             float = 0.1
    max_epochs:      int   = 500
    error_criterion: float = 0.001
    nu_h:            float = 0.05
    sigma_input:     float = 0.5
    pr_noise_hetero: bool  = True
    pr_noise_gauss:  bool  = True

    @property
    def max_cycles(self) -> int:
        return 4 * self.network_size

    @property
    def total_patterns(self) -> int:
        return self.base_pop + self.max_new


# ─── Experiment knobs (edit these) ────────────────────────────────
cfg_100 = McCallumConfig(max_pseudoitems=100)
cfg_256 = McCallumConfig(max_pseudoitems=256)
cfg_512 = McCallumConfig(max_pseudoitems=512)
cfg     = cfg_256   # shared config for delta baselines

N_TRIALS = 10
SEED     = None     # set to int for reproducibility
# ──────────────────────────────────────────────────────────────────

# %% [markdown]
# ## Numba kernels

# %% Shared kernels
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
                    eta, max_epochs, err_crit, nu_h, sigma,
                    use_hetero, use_gauss, N):
    smooth = 1.0
    for epoch in range(max_epochs):
        order = np.random.permutation(n_pat)
        epoch_err = 0.0
        for pos in range(n_pat):
            p = order[pos]
            target = patterns[p]
            is_new = (p == new_idx)
            if is_new and use_hetero:
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
                if is_new and use_gauss:
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
def _train_delta_pure_hetero(weights, target, eta, max_epochs, err_crit, nu_h, N):
    smooth = 1.0
    for epoch in range(max_epochs):
        inp = _flip_noise(target, nu_h)
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


@njit(cache=True)
def _train_delta_pure_gaussian(weights, target, eta, max_epochs, err_crit, sigma, N):
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
            h += np.random.normal(0.0, sigma)
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
_probe_pseudoitems(_w, 5, 3, 2, 2)
_flip_noise(_s, 0.2)
_train_base(_w, _p, 2, 0.1, 2, 0.001, 5)
_train_delta_pr(_w, _p, 2, 1, 0.1, 2, 0.001, 0.05, 0.5, True, False, 5)
_train_delta_pure_hetero(_w, _s, 0.1, 2, 0.001, 0.05, 5)
_train_delta_pure_gaussian(_w, _s, 0.1, 2, 0.001, 0.5, 5)
_train_delta_naive(_w, _s, 0.1, 2, 0.001, 5)
del _w, _s, _p
print(f"done ({time.time()-t0:.1f}s)")

# %% [markdown]
# ## Protocol runners

# %% Protocol runners
@dataclass
class RunResult:
    M:      List[int]
    stable: List[int]
    pseudo: List[int]
    config: McCallumConfig = field(repr=False)


def run_pseudorehearsal(cfg: McCallumConfig, seed=None, verbose=False) -> RunResult:
    if seed is not None:
        np.random.seed(seed)
    N, r = cfg.network_size, cfg.max_cycles
    patterns = np.random.choice([-1.0, 1.0], size=(cfg.total_patterns, N))
    weights = np.zeros((N, N))
    M_list, stable_list, pseudo_list = [], [], []

    weights, _ = _train_base(weights, patterns[:cfg.base_pop].copy(),
                             cfg.base_pop, cfg.eta, cfg.max_epochs,
                             cfg.error_criterion, N)
    ns = _count_stable(weights, patterns, cfg.base_pop, r)
    M_list.append(cfg.base_pop); stable_list.append(ns); pseudo_list.append(0)

    for step in range(cfg.max_new):
        M = cfg.base_pop + step + 1
        pseudos = _probe_pseudoitems(weights, N, cfg.n_probes,
                                     cfg.max_pseudoitems, r)
        train_set = np.vstack((pseudos, patterns[M-1:M]))
        n_train = train_set.shape[0]
        weights, _ = _train_delta_pr(
            weights, train_set, n_train, n_train - 1,
            cfg.eta, cfg.max_epochs, cfg.error_criterion,
            cfg.nu_h, cfg.sigma_input,
            cfg.pr_noise_hetero, cfg.pr_noise_gauss, N)
        ns = _count_stable(weights, patterns, M, r)
        M_list.append(M); stable_list.append(ns)
        pseudo_list.append(pseudos.shape[0])
        if verbose and (step < 5 or (step+1) % 10 == 0 or step == cfg.max_new-1):
            print(f"  Pr M={M:3d}: stable={ns:3d}/{M}  pseudo={pseudos.shape[0]:3d}")
    return RunResult(M_list, stable_list, pseudo_list, cfg)


def run_delta_hetero(cfg: McCallumConfig, seed=None, verbose=False) -> RunResult:
    if seed is not None:
        np.random.seed(seed)
    N, r = cfg.network_size, cfg.max_cycles
    patterns = np.random.choice([-1.0, 1.0], size=(cfg.total_patterns, N))
    weights = np.zeros((N, N))
    M_list, stable_list, pseudo_list = [], [], []

    weights, _ = _train_base(weights, patterns[:cfg.base_pop].copy(),
                             cfg.base_pop, cfg.eta, cfg.max_epochs,
                             cfg.error_criterion, N)
    ns = _count_stable(weights, patterns, cfg.base_pop, r)
    M_list.append(cfg.base_pop); stable_list.append(ns); pseudo_list.append(0)

    for step in range(cfg.max_new):
        M = cfg.base_pop + step + 1
        weights, _ = _train_delta_pure_hetero(
            weights, patterns[M-1], cfg.eta, cfg.max_epochs,
            cfg.error_criterion, cfg.nu_h, N)
        ns = _count_stable(weights, patterns, M, r)
        M_list.append(M); stable_list.append(ns); pseudo_list.append(0)
        if verbose and (step < 5 or (step+1) % 10 == 0 or step == cfg.max_new-1):
            print(f"  Delta(h) M={M:3d}: stable={ns:3d}/{M}")
    return RunResult(M_list, stable_list, pseudo_list, cfg)


def run_delta_gaussian(cfg: McCallumConfig, seed=None, verbose=False) -> RunResult:
    if seed is not None:
        np.random.seed(seed)
    N, r = cfg.network_size, cfg.max_cycles
    patterns = np.random.choice([-1.0, 1.0], size=(cfg.total_patterns, N))
    weights = np.zeros((N, N))
    M_list, stable_list, pseudo_list = [], [], []

    weights, _ = _train_base(weights, patterns[:cfg.base_pop].copy(),
                             cfg.base_pop, cfg.eta, cfg.max_epochs,
                             cfg.error_criterion, N)
    ns = _count_stable(weights, patterns, cfg.base_pop, r)
    M_list.append(cfg.base_pop); stable_list.append(ns); pseudo_list.append(0)

    for step in range(cfg.max_new):
        M = cfg.base_pop + step + 1
        weights, _ = _train_delta_pure_gaussian(
            weights, patterns[M-1], cfg.eta, cfg.max_epochs,
            cfg.error_criterion, cfg.sigma_input, N)
        ns = _count_stable(weights, patterns, M, r)
        M_list.append(M); stable_list.append(ns); pseudo_list.append(0)
        if verbose and (step < 5 or (step+1) % 10 == 0 or step == cfg.max_new-1):
            print(f"  Delta(g) M={M:3d}: stable={ns:3d}/{M}")
    return RunResult(M_list, stable_list, pseudo_list, cfg)


def run_delta_naive(cfg: McCallumConfig, seed=None, verbose=False) -> RunResult:
    """Iterative delta learning on each new pattern — no rehearsal, no noise."""
    if seed is not None:
        np.random.seed(seed)
    N, r = cfg.network_size, cfg.max_cycles
    patterns = np.random.choice([-1.0, 1.0], size=(cfg.total_patterns, N))
    weights = np.zeros((N, N))
    M_list, stable_list, pseudo_list = [], [], []

    weights, _ = _train_base(weights, patterns[:cfg.base_pop].copy(),
                             cfg.base_pop, cfg.eta, cfg.max_epochs,
                             cfg.error_criterion, N)
    ns = _count_stable(weights, patterns, cfg.base_pop, r)
    M_list.append(cfg.base_pop); stable_list.append(ns); pseudo_list.append(0)

    for step in range(cfg.max_new):
        M = cfg.base_pop + step + 1
        weights, _ = _train_delta_naive(
            weights, patterns[M-1], cfg.eta, cfg.max_epochs,
            cfg.error_criterion, N)
        ns = _count_stable(weights, patterns, M, r)
        M_list.append(M); stable_list.append(ns); pseudo_list.append(0)
        if verbose and (step < 5 or (step+1) % 10 == 0 or step == cfg.max_new-1):
            print(f"  Naive M={M:3d}: stable={ns:3d}/{M}")
    return RunResult(M_list, stable_list, pseudo_list, cfg)


# %% CSV helpers
def save_condition_csv(condition: str, results_list: List[RunResult], out_dir: Path):
    """Save per-trial time series: condition, trial, M, stable, pseudo."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{condition}.csv"
    with open(path, "w") as f:
        f.write("condition,trial,M,stable,pseudo\n")
        for t, res in enumerate(results_list):
            for i, m in enumerate(res.M):
                f.write(f"{condition},{t},{m},{res.stable[i]},{res.pseudo[i]}\n")
    print(f"  Saved {path.name} ({len(results_list)} trials, "
          f"{len(results_list[0].M)} steps each)")
    return path


def run_and_save(condition: str, run_fn, cfg: McCallumConfig,
                 n_trials: int, seed, out_dir: Path):
    """Run all trials for one condition and save CSV."""
    print(f"\n{'='*60}")
    print(f"{condition} — {n_trials} trials, N={cfg.network_size}")
    print(f"{'='*60}")

    results = []
    for t in range(n_trials):
        t0 = time.time()
        trial_seed = (seed + t) if seed is not None else None
        res = run_fn(cfg, seed=trial_seed, verbose=False)
        results.append(res)
        print(f"  Trial {t+1:2d}/{n_trials}: "
              f"{res.stable[-1]:3d}/{res.M[-1]} stable, {time.time()-t0:.1f}s")

    save_condition_csv(condition, results, out_dir)
    return results


# %% [markdown]
# ## Run all conditions and save

# %% Pr100
run_and_save("pr100", run_pseudorehearsal, cfg_100, N_TRIALS, SEED, OUTPUT_DIR)

# %% Pr256
run_and_save("pr256", run_pseudorehearsal, cfg_256, N_TRIALS, SEED, OUTPUT_DIR)

# %% Pr512
run_and_save("pr512", run_pseudorehearsal, cfg_512, N_TRIALS, SEED, OUTPUT_DIR)

# %% Delta (hetero)
run_and_save("delta_hetero", run_delta_hetero, cfg, N_TRIALS, SEED, OUTPUT_DIR)

# %% Delta (Gaussian)
run_and_save("delta_gaussian", run_delta_gaussian, cfg, N_TRIALS, SEED, OUTPUT_DIR)

# %% Naive (no rehearsal)
run_and_save("naive", run_delta_naive, cfg, N_TRIALS, SEED, OUTPUT_DIR)

# %% Done
print(f"\n{'='*60}")
print(f"All conditions saved to: {OUTPUT_DIR}")
print(f"{'='*60}")
