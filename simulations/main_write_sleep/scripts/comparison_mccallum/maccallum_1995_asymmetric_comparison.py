# %% [markdown]
# # McCallum 1995 Pseudorehearsal — With Delta Baseline
#
# Reproduces McCallum's Figure 4.23 comparison:
# - **Pr100/256/512**: Pseudorehearsal with varying pseudoitem caps
# - **Delta**: Simple delta learning, no rehearsal (new pattern only)
#
# Both conditions: H100,±, BP=5, patterns added one at a time.
#
# Noise (Section 4.5 defaults):
# - Heteroassociative noise νh=5% on the new pattern
# - No noise on pseudoitems (noiseOnPseudoPatts=False)
# - Convergence criterion on all patterns (new + pseudoitems)
#
# Delta learning baseline (Section 4.4.7):
# - Each new pattern trained alone (no pseudoitems)
# - Gaussian input noise σ=0.5 on the new pattern
# - Same η=0.1, max 500 epochs, error criterion 0.001

# %% Imports
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt
from numba import njit

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
    nu_h:            float = 0.05   # heteroassociative noise fraction
    sigma_input:     float = 0.5    # Gaussian input noise std
    pr_noise_hetero: bool  = True   # apply heteroassociative noise on new pattern
    pr_noise_gauss:  bool  = True  # apply Gaussian input noise on new pattern

    @property
    def max_cycles(self) -> int:
        return 4 * self.network_size

    @property
    def total_patterns(self) -> int:
        return self.base_pop + self.max_new


# ─── Experiment knobs ───────────────────────────────────────────────
cfg_100 = McCallumConfig(
    network_size    = 100,
    base_pop        = 5,
    max_new         = 95,
    max_pseudoitems = 100,
)

cfg_256 = McCallumConfig(
    network_size    = 100,
    base_pop        = 5,
    max_new         = 95,
    max_pseudoitems = 256,
)

cfg_512 = McCallumConfig(
    network_size    = 100,
    base_pop        = 5,
    max_new         = 95,
    max_pseudoitems = 512,
)

# Use cfg_256 as the shared config for delta baselines (Pi irrelevant there)
cfg = cfg_256

N_TRIALS = 15
SEED     = None
# ────────────────────────────────────────────────────────────────────

# %% [markdown]
# ## Numba kernels — shared

# %% Shared kernels (unchanged)
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


# %% Pseudorehearsal delta (configurable noise: hetero, gaussian, both, or none)
@njit(cache=True)
def _train_delta_pr(weights, patterns, n_pat, new_idx,
                    eta, max_epochs, err_crit, nu_h, sigma,
                    use_hetero, use_gauss, N):
    """Delta learning for pseudorehearsal condition.

    Noise applied to the NEW pattern only (noiseOnPseudoPatts=False).
    Pseudoitems are always trained cleanly.

    use_hetero: flip nu_h fraction of input bits (basin of attraction)
    use_gauss:  add N(0,σ) to local field h_i (push decision surface)
    Both can be True simultaneously.

    Convergence criterion on ALL patterns.
    Asymmetric weight updates.
    """
    smooth = 1.0
    for epoch in range(max_epochs):
        order = np.random.permutation(n_pat)
        epoch_err = 0.0
        for pos in range(n_pat):
            p = order[pos]
            target = patterns[p]
            is_new = (p == new_idx)

            # Heteroassociative noise: flip bits in input (new pattern only)
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
                # Gaussian input noise on local field (new pattern only)
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


# %% Pure delta — heteroassociative noise variant
@njit(cache=True)
def _train_delta_pure_hetero(weights, target, eta, max_epochs, err_crit, nu_h, N):
    """Train a SINGLE new pattern with delta learning (no rehearsal).
    Heteroassociative noise: flip nu_h fraction of input bits.
    This is the simplest interpretation of "Delta" in Figure 4.23."""
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


# %% Pure delta — Gaussian input noise variant (Section 4.4.7)
@njit(cache=True)
def _train_delta_pure_gaussian(weights, target, eta, max_epochs, err_crit,
                               sigma, N):
    """Train a SINGLE new pattern with delta learning (no rehearsal).
    Gaussian input noise: add N(0, sigma) to each unit's local field.
    This matches Section 4.4.7: 'absolute Gaussian noise νi=0.5'."""
    smooth = 1.0
    for epoch in range(max_epochs):
        inp = target.copy()  # no bit flips; noise is on the local field
        errors = np.zeros(N)
        any_err = False
        for i in range(N):
            h = 0.0
            for j in range(N):
                if j != i:
                    h += weights[i, j] * inp[j]
            # Add Gaussian noise to the local field
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


# %% Base population training (shared by all conditions)
@njit(cache=True)
def _train_base(weights, patterns, n_base, eta, max_epochs, err_crit, N):
    """Train base population — no noise, no pseudorehearsal."""
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


# %% JIT warmup
print("Compiling Numba kernels ...", end=" ", flush=True)
t0 = time.time()

_w = np.zeros((5, 5))
_s = np.ones(5)
_p = np.ones((2, 5))
_relax_async(_w, _s.copy(), 2)
_is_stable(_w, _s, 2)
_count_stable(_w, _p, 2, 2)
_probe_pseudoitems(_w, 5, 3, 2, 2)
_flip_noise(_s, 0.2)
_train_delta_pr(_w, _p, 2, 1, 0.1, 2, 0.001, 0.05, 0.5, True, False, 5)
_train_delta_pure_hetero(_w, _s, 0.1, 2, 0.001, 0.05, 5)
_train_delta_pure_gaussian(_w, _s, 0.1, 2, 0.001, 0.5, 5)
_train_base(_w, _p, 2, 0.1, 2, 0.001, 5)
del _w, _s, _p

print(f"done ({time.time()-t0:.1f}s)")

# %% [markdown]
# ## Protocol runners

# %% Data structures
@dataclass
class RunResult:
    M:      List[int]
    stable: List[int]
    pseudo: List[int]
    config: McCallumConfig = field(repr=False)


@dataclass
class TrialResults:
    M:          np.ndarray
    stable_all: np.ndarray
    pseudo_all: np.ndarray
    config:     McCallumConfig

    @property
    def stable_mean(self): return self.stable_all.mean(axis=0)
    @property
    def stable_std(self):  return self.stable_all.std(axis=0)
    @property
    def pseudo_mean(self): return self.pseudo_all.mean(axis=0)
    @property
    def pseudo_std(self):  return self.pseudo_all.std(axis=0)
    @property
    def n_trials(self):    return self.stable_all.shape[0]


# %% Pseudorehearsal protocol (Pr256 / Pr512)
def run_pseudorehearsal(cfg: McCallumConfig, seed=None, verbose=False) -> RunResult:
    """Pseudorehearsal condition (Pr256 in Figure 4.23)."""
    if seed is not None:
        np.random.seed(seed)

    N, r = cfg.network_size, cfg.max_cycles
    patterns = np.random.choice([-1.0, 1.0], size=(cfg.total_patterns, N))
    weights  = np.zeros((N, N))

    M_list, stable_list, pseudo_list = [], [], []

    # Base population
    weights, _ = _train_base(
        weights, patterns[:cfg.base_pop].copy(), cfg.base_pop,
        cfg.eta, cfg.max_epochs, cfg.error_criterion, N)
    ns = _count_stable(weights, patterns, cfg.base_pop, r)
    M_list.append(cfg.base_pop); stable_list.append(ns); pseudo_list.append(0)

    # Incremental with pseudorehearsal
    for step in range(cfg.max_new):
        M = cfg.base_pop + step + 1
        pseudos = _probe_pseudoitems(weights, N, cfg.n_probes,
                                     cfg.max_pseudoitems, r)
        n_pseudo = pseudos.shape[0]

        train_set = np.vstack((pseudos, patterns[M-1:M]))
        n_train = train_set.shape[0]

        weights, _ = _train_delta_pr(
            weights, train_set, n_train, n_train - 1,
            cfg.eta, cfg.max_epochs, cfg.error_criterion,
            cfg.nu_h, cfg.sigma_input,
            cfg.pr_noise_hetero, cfg.pr_noise_gauss, N)

        ns = _count_stable(weights, patterns, M, r)
        M_list.append(M); stable_list.append(ns); pseudo_list.append(n_pseudo)

        if verbose and (step < 5 or (step+1) % 10 == 0 or step == cfg.max_new - 1):
            print(f"  Pr M={M:3d}: stable={ns:3d}/{M}  pseudo={n_pseudo:3d}")

    return RunResult(M_list, stable_list, pseudo_list, cfg)


# %% Pure delta protocol — heteroassociative noise
def run_delta_hetero(cfg: McCallumConfig, seed=None, verbose=False) -> RunResult:
    """Pure delta learning, no rehearsal. Heteroassociative noise νh on new patterns."""
    if seed is not None:
        np.random.seed(seed)

    N, r = cfg.network_size, cfg.max_cycles
    patterns = np.random.choice([-1.0, 1.0], size=(cfg.total_patterns, N))
    weights  = np.zeros((N, N))

    M_list, stable_list, pseudo_list = [], [], []

    # Same base population training
    weights, _ = _train_base(
        weights, patterns[:cfg.base_pop].copy(), cfg.base_pop,
        cfg.eta, cfg.max_epochs, cfg.error_criterion, N)
    ns = _count_stable(weights, patterns, cfg.base_pop, r)
    M_list.append(cfg.base_pop); stable_list.append(ns); pseudo_list.append(0)

    # Incremental: train ONLY on the new pattern (no pseudoitems)
    for step in range(cfg.max_new):
        M = cfg.base_pop + step + 1

        weights, _ = _train_delta_pure_hetero(
            weights, patterns[M-1], cfg.eta, cfg.max_epochs,
            cfg.error_criterion, cfg.nu_h, N)

        ns = _count_stable(weights, patterns, M, r)
        M_list.append(M); stable_list.append(ns); pseudo_list.append(0)

        if verbose and (step < 5 or (step+1) % 10 == 0 or step == cfg.max_new - 1):
            print(f"  Delta(h) M={M:3d}: stable={ns:3d}/{M}")

    return RunResult(M_list, stable_list, pseudo_list, cfg)


# %% Pure delta protocol — Gaussian input noise (Section 4.4.7)
def run_delta_gaussian(cfg: McCallumConfig, seed=None, verbose=False) -> RunResult:
    """Pure delta learning, no rehearsal. Gaussian input noise σ=0.5 (Section 4.4.7)."""
    if seed is not None:
        np.random.seed(seed)

    N, r = cfg.network_size, cfg.max_cycles
    patterns = np.random.choice([-1.0, 1.0], size=(cfg.total_patterns, N))
    weights  = np.zeros((N, N))

    M_list, stable_list, pseudo_list = [], [], []

    # Same base population training
    weights, _ = _train_base(
        weights, patterns[:cfg.base_pop].copy(), cfg.base_pop,
        cfg.eta, cfg.max_epochs, cfg.error_criterion, N)
    ns = _count_stable(weights, patterns, cfg.base_pop, r)
    M_list.append(cfg.base_pop); stable_list.append(ns); pseudo_list.append(0)

    # Incremental: train ONLY on the new pattern (no pseudoitems)
    for step in range(cfg.max_new):
        M = cfg.base_pop + step + 1

        weights, _ = _train_delta_pure_gaussian(
            weights, patterns[M-1], cfg.eta, cfg.max_epochs,
            cfg.error_criterion, cfg.sigma_input, N)

        ns = _count_stable(weights, patterns, M, r)
        M_list.append(M); stable_list.append(ns); pseudo_list.append(0)

        if verbose and (step < 5 or (step+1) % 10 == 0 or step == cfg.max_new - 1):
            print(f"  Delta(g) M={M:3d}: stable={ns:3d}/{M}")

    return RunResult(M_list, stable_list, pseudo_list, cfg)


# %% Trial aggregator (generic)
def run_trials(run_fn, cfg, n_trials=10, seed=None, verbose=True, label=""):
    all_stable, all_pseudo = [], []
    M = None
    for t in range(n_trials):
        t0 = time.time()
        trial_seed = (seed + t) if seed is not None else None
        res = run_fn(cfg, seed=trial_seed, verbose=False)
        all_stable.append(res.stable)
        all_pseudo.append(res.pseudo)
        M = res.M
        if verbose:
            print(f"  {label} trial {t+1}/{n_trials}: "
                  f"{res.stable[-1]}/{res.M[-1]} stable, {time.time()-t0:.1f}s")
    return TrialResults(
        M=np.array(M),
        stable_all=np.array(all_stable),
        pseudo_all=np.array(all_pseudo),
        config=cfg,
    )


def save_csv(results: TrialResults, path, label=""):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    N = results.config.network_size
    with open(path, "w") as f:
        f.write("condition,network_size,M,stable_mean,stable_std,"
                "pseudo_mean,pseudo_std\n")
        for i, m in enumerate(results.M):
            f.write(f"{label},{N},{m},"
                    f"{results.stable_mean[i]:.2f},{results.stable_std[i]:.2f},"
                    f"{results.pseudo_mean[i]:.2f},{results.pseudo_std[i]:.2f}\n")
    print(f"CSV saved -> {path}")


# %% [markdown]
# ## Run all conditions

# %% Run pseudorehearsal — Pr100
print("=" * 60)
print(f"Pseudorehearsal (Pr100) — {N_TRIALS} trials")
print("=" * 60)
t0 = time.time()
res_pr100 = run_trials(run_pseudorehearsal, cfg_100, N_TRIALS, SEED, label="Pr100")
print(f"  Total: {time.time()-t0:.1f}s\n")

# %% Run pseudorehearsal — Pr256
print("=" * 60)
print(f"Pseudorehearsal (Pr256) — {N_TRIALS} trials")
print("=" * 60)
t0 = time.time()
res_pr256 = run_trials(run_pseudorehearsal, cfg_256, N_TRIALS, SEED, label="Pr256")
print(f"  Total: {time.time()-t0:.1f}s\n")

# %% Run pseudorehearsal — Pr512
print("=" * 60)
print(f"Pseudorehearsal (Pr512) — {N_TRIALS} trials")
print("=" * 60)
t0 = time.time()
res_pr512 = run_trials(run_pseudorehearsal, cfg_512, N_TRIALS, SEED, label="Pr512")
print(f"  Total: {time.time()-t0:.1f}s\n")

# %% Run pure delta — heteroassociative noise
print("=" * 60)
print(f"Pure delta (hetero νh={cfg.nu_h}) — {N_TRIALS} trials")
print("=" * 60)
t0 = time.time()
res_dh = run_trials(run_delta_hetero, cfg, N_TRIALS, SEED, label="Delta(h)")
print(f"  Total: {time.time()-t0:.1f}s\n")

# %% Run pure delta — Gaussian input noise
print("=" * 60)
print(f"Pure delta (Gaussian σ={cfg.sigma_input}) — {N_TRIALS} trials")
print("=" * 60)
t0 = time.time()
res_dg = run_trials(run_delta_gaussian, cfg, N_TRIALS, SEED, label="Delta(g)")
print(f"  Total: {time.time()-t0:.1f}s\n")

# %% [markdown]
# ## Comparison plot (cf. McCallum Figures 4.23 & 4.24)

# %% Comparison plot — publication figure
M = res_pr256.M

fig, ax = plt.subplots(figsize=(8, 6))

# Perfect recall — solid gray, drawn first so data sits on top
ax.plot(M, M, color="0.45", ls="-", lw=1.8, label="Perfect recall", zorder=1)

# Delta baselines — red tones
ax.plot(M, res_dh.stable_mean, color="#c0392b", ls="-", lw=2.5,
        label=r"Iterative GDA", zorder=2)

# ax.plot(M, res_dg.stable_mean, color="#e74c3c", ls="--", lw=2.2,
#         label=r"Delta (Gaussian $\sigma$=0.5)", zorder=2)

# Pseudorehearsal conditions — cool palette
ax.plot(M, res_pr100.stable_mean, color="#8e44ad", ls="-", lw=2.2,
        label="Pr100", zorder=3)
ax.plot(M, res_pr256.stable_mean, color="#2980b9", ls="-", lw=2.2,
        label="Pr256", zorder=3)
ax.plot(M, res_pr512.stable_mean, color="#16a085", ls="-", lw=2.2,
        label="Pr512", zorder=3)

# Axes
ymax = max(25, int(max(res_pr100.stable_mean.max(),
                       res_pr256.stable_mean.max(),
                       res_pr512.stable_mean.max())) + 5)
ax.set_xlabel("Patterns learned ($M$)", fontsize=20)
ax.set_ylabel("Stable patterns", fontsize=20)
ax.set_xlim(0, M[-1] + 2)
ax.set_ylim(0, ymax)
ax.tick_params(labelsize=16, width=1.2, length=5)

# Legend
ax.legend(loc="upper left", fontsize=15, frameon=False)

# Clean frame
for spine in ax.spines.values():
    spine.set_linewidth(1.2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("mccallum_1995_comparison.png", dpi=200, bbox_inches="tight")
plt.show()
print("Saved: mccallum_1995_comparison.png")

# %% [markdown]
# ## Individual condition plots

# %% Pseudoitems plot (all Pr conditions)
fig, ax = plt.subplots(figsize=(10, 5))

pm100, ps100 = res_pr100.pseudo_mean, res_pr100.pseudo_std
ax.plot(M, pm100, "m-", lw=2, label="Pr100")
ax.fill_between(M, pm100 - ps100, pm100 + ps100, color="m", alpha=0.15)

pm256, ps256 = res_pr256.pseudo_mean, res_pr256.pseudo_std
ax.plot(M, pm256, "b-", lw=2, label="Pr256")
ax.fill_between(M, pm256 - ps256, pm256 + ps256, color="b", alpha=0.15)

pm512, ps512 = res_pr512.pseudo_mean, res_pr512.pseudo_std
ax.plot(M, pm512, "c-", lw=2, label="Pr512")
ax.fill_between(M, pm512 - ps512, pm512 + ps512, color="c", alpha=0.15)

ax.axhline(100, ls="--", color="m", alpha=0.3, label="cap 100")
ax.axhline(256, ls="--", color="b", alpha=0.3, label="cap 256")
ax.axhline(512, ls="--", color="c", alpha=0.3, label="cap 512")
ax.set(xlabel="Patterns learned (M)", ylabel="Pseudoitems found",
       xlim=(0, M[-1]+2))
ax.set_title(f"Pseudoitems per step (Pp={cfg.n_probes}, {N_TRIALS} trials)")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Summary statistics

# %% Summary
print("=" * 60)
print(f"SUMMARY — N={cfg.network_size}, {N_TRIALS} trials")
print("=" * 60)

for label, res in [
    ("Pr100", res_pr100),
    ("Pr256", res_pr256),
    ("Pr512", res_pr512),
    ("Delta (hetero)", res_dh),
    ("Delta (Gauss)", res_dg),
]:
    m = res.stable_mean
    s = res.stable_std
    print(f"\n  {label}:")
    print(f"    Stable at M={M[-1]:3d}: {m[-1]:.1f} ± {s[-1]:.1f}")
    # Find peak
    peak_idx = np.argmax(m)
    print(f"    Peak: {m[peak_idx]:.1f} at M={M[peak_idx]}")
    for target_M in [15, 30, 50]:
        hits = [i for i, mm in enumerate(M) if mm >= target_M]
        if hits:
            idx = hits[0]
            print(f"    Stable at M={target_M:3d}: {m[idx]:.1f} ± {s[idx]:.1f}")