# %% [markdown]
# # McCallum 1995 Pseudorehearsal — Asymmetric Delta Rule
#
# Identical to `mccallum_1995_corrected.py` **except**:
#
# **The weight update is NOT symmetrised.**
#
# McCallum's delta rule (Eq. 2.7): Δw_ij = η * e_i * inp_j
# - Only row i is updated when unit i has an error.
# - The weight matrix is allowed to be (slightly) asymmetric.
#
# The corrected version divides by 2 and mirrors to (j,i), which
# effectively halves the learning rate. McCallum's thesis (p.96) describes
# a sharp phase transition at ~0.14N where stored patterns destabilise
# simultaneously — that cascade requires full-strength plasticity.
#
# This script tests whether removing the symmetrisation reproduces
# the characteristic dip seen in McCallum's Figure 4.23.

# %% Imports
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt
from numba import njit

# %% [markdown]
# ## Configuration

# %% Configuration — edit here
@dataclass
class McCallumConfig:
    """All tuneable parameters for the McCallum 1995 protocol."""
    network_size:    int   = 100
    base_pop:        int   = 5      # BP: patterns learned before pseudorehearsal
    max_new:         int   = 95     # new patterns added incrementally
    n_probes:        int   = 2000   # Pp: random probes per incorporation
    max_pseudoitems: int   = 512    # Pi: cap on unique pseudoitems kept
    eta:             float = 0.1    # delta-rule learning rate
    max_epochs:      int   = 500    # training epochs per incorporation
    error_criterion: float = 0.001  # early-stop smoothed-error threshold
    nu_h:            float = 0.05   # heteroassociative noise (fraction flipped)

    @property
    def max_cycles(self) -> int:
        """Relaxation budget: r = 4N."""
        return 4 * self.network_size

    @property
    def total_patterns(self) -> int:
        return self.base_pop + self.max_new


# ─── Experiment knobs (change these) ────────────────────────────────
cfg = McCallumConfig(
    network_size    = 100,
    base_pop        = 5,
    max_new         = 95,
)

N_TRIALS = 20
SEED     = None        # set to an int for reproducibility
# ────────────────────────────────────────────────────────────────────

# %% [markdown]
# ## Numba kernels
#
# Shared kernels (relaxation, probing, noise) are identical to the
# corrected version. Only `_train_delta_sync` and `_train_base` differ.

# %% Numba kernels — shared (unchanged)
@njit(cache=True)
def _relax_async(weights, state, max_cycles):
    """Asynchronous relaxation until convergence or max_cycles."""
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
    """True iff pattern is a fixed point (strict identity, no inverse)."""
    state = _relax_async(weights, pattern.copy(), max_cycles)
    for i in range(state.shape[0]):
        if state[i] != pattern[i]:
            return False
    return True


@njit(cache=True)
def _count_stable(weights, patterns, M, max_cycles):
    """Count how many of the first M patterns are fixed points."""
    count = 0
    for mu in range(M):
        if _is_stable(weights, patterns[mu], max_cycles):
            count += 1
    return count


@njit(cache=True)
def _probe_pseudoitems(weights, N, n_probes, max_items, max_cycles):
    """Probe network for unique stable states. Returns (n_found, N) array."""
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
    """Flip nu_h fraction of randomly chosen bits."""
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


# %% Numba kernels — ASYMMETRIC delta rule (the change)
@njit(cache=True)
def _train_delta_sync(weights, patterns, n_pat, new_idx,
                      eta, max_epochs, err_crit, nu_h, N):
    """Synchronous delta learning — ASYMMETRIC (McCallum Eq. 2.7).
    Only row i is updated: Δw_ij = η * e_i * inp_j.
    No /2, no mirror to (j,i)."""
    smooth = 1.0
    for epoch in range(max_epochs):
        order = np.random.permutation(n_pat)
        epoch_err = 0.0
        for pos in range(n_pat):
            p = order[pos]
            target = patterns[p]
            inp = _flip_noise(target, nu_h) if p == new_idx else target.copy()
            # synchronous: compute all errors first
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
            # asymmetric update: only row i
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
def _train_base(weights, patterns, n_base, eta, max_epochs, err_crit, N):
    """Train base population — ASYMMETRIC (no noise, no pseudorehearsal)."""
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
_train_delta_sync(_w, _p, 2, 1, 0.1, 2, 0.001, 0.05, 5)
_train_base(_w, _p, 2, 0.1, 2, 0.001, 5)
del _w, _s, _p

print(f"done ({time.time()-t0:.1f}s)")

# %% [markdown]
# ## Protocol runners

# %% Protocol runners
@dataclass
class RunResult:
    """Output of a single protocol run."""
    M:      List[int]
    stable: List[int]
    pseudo: List[int]
    config: McCallumConfig = field(repr=False)


def run_protocol(cfg: McCallumConfig, seed: int = None,
                 verbose: bool = False) -> RunResult:
    """Execute one full McCallum 1995 protocol run."""
    if seed is not None:
        np.random.seed(seed)

    N = cfg.network_size
    r = cfg.max_cycles
    patterns = np.random.choice([-1.0, 1.0], size=(cfg.total_patterns, N))
    weights  = np.zeros((N, N))

    M_list, stable_list, pseudo_list = [], [], []

    # Phase 1 — base population
    if verbose:
        print(f"  Base pop ({cfg.base_pop}) ...", end=" ", flush=True)
    t0 = time.time()
    weights, ep = _train_base(
        weights, patterns[:cfg.base_pop].copy(), cfg.base_pop,
        cfg.eta, cfg.max_epochs, cfg.error_criterion, N)
    ns = _count_stable(weights, patterns, cfg.base_pop, r)
    M_list.append(cfg.base_pop)
    stable_list.append(ns)
    pseudo_list.append(0)
    if verbose:
        print(f"{ns}/{cfg.base_pop} stable, {ep} ep, {time.time()-t0:.1f}s")

    # Phase 2 — incremental pseudorehearsal
    for step in range(cfg.max_new):
        M = cfg.base_pop + step + 1
        t0 = time.time()

        pseudos = _probe_pseudoitems(weights, N, cfg.n_probes,
                                     cfg.max_pseudoitems, r)
        n_pseudo = pseudos.shape[0]

        train_set = np.vstack((pseudos, patterns[M-1:M]))
        n_train   = train_set.shape[0]

        weights, ep = _train_delta_sync(
            weights, train_set, n_train, n_train - 1,
            cfg.eta, cfg.max_epochs, cfg.error_criterion, cfg.nu_h, N)

        ns = _count_stable(weights, patterns, M, r)
        M_list.append(M)
        stable_list.append(ns)
        pseudo_list.append(n_pseudo)

        if verbose and (step < 5 or (step+1) % 10 == 0 or step == cfg.max_new - 1):
            print(f"  M={M:3d}: stable={ns:3d}/{M}  pseudo={n_pseudo:3d}  "
                  f"ep={ep:3d}  {time.time()-t0:.1f}s")

    return RunResult(M_list, stable_list, pseudo_list, cfg)


@dataclass
class TrialResults:
    """Aggregated results over multiple trials."""
    M:          np.ndarray
    stable_all: np.ndarray   # (n_trials, n_steps)
    pseudo_all: np.ndarray   # (n_trials, n_steps)
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


def run_trials(cfg: McCallumConfig, n_trials: int = 10,
               seed: int = None, verbose: bool = True) -> TrialResults:
    """Run n_trials independent protocol runs and aggregate."""
    all_stable, all_pseudo = [], []
    M = None
    for t in range(n_trials):
        t0 = time.time()
        trial_seed = (seed + t) if seed is not None else None
        res = run_protocol(cfg, seed=trial_seed, verbose=False)
        all_stable.append(res.stable)
        all_pseudo.append(res.pseudo)
        M = res.M
        if verbose:
            print(f"  Trial {t+1}/{n_trials}: "
                  f"{res.stable[-1]}/{res.M[-1]} stable, {time.time()-t0:.1f}s")
    return TrialResults(
        M=np.array(M),
        stable_all=np.array(all_stable),
        pseudo_all=np.array(all_pseudo),
        config=cfg,
    )


def save_csv(results: TrialResults, path):
    """Write per-step mean/std to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    N = results.config.network_size
    with open(path, "w") as f:
        f.write("network_size,M,stable_mean,stable_std,pseudo_mean,pseudo_std\n")
        for i, m in enumerate(results.M):
            f.write(f"{N},{m},"
                    f"{results.stable_mean[i]:.2f},{results.stable_std[i]:.2f},"
                    f"{results.pseudo_mean[i]:.2f},{results.pseudo_std[i]:.2f}\n")
    print(f"CSV saved -> {path}")

# %% [markdown]
# ## Run single trial (verbose)

# %% Single run
print("=" * 60)
print(f"ASYMMETRIC delta — N={cfg.network_size}, BP={cfg.base_pop}, "
      f"new={cfg.max_new}, Pp={cfg.n_probes}")
print("=" * 60)

t_start = time.time()
single = run_protocol(cfg, seed=SEED, verbose=True)
print(f"\nDone in {time.time()-t_start:.1f}s")

# %% Plot single run
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.plot(single.M, single.stable, "b-", lw=2, label=f"Pr{cfg.max_pseudoitems}")
ax.plot(single.M, single.M, "k--", lw=1, alpha=0.4, label="perfect recall")
ax.set(xlabel="Patterns learned (M)", ylabel="Stable patterns",
       xlim=(0, single.M[-1]+2), ylim=(0, single.M[-1]+5))
ax.set_title(f"McCallum 1995 ASYMMETRIC — N={cfg.network_size} (single run)")
ax.legend(loc="upper left"); ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(single.M, single.pseudo, "r-", lw=2)
ax.axhline(cfg.max_pseudoitems, ls="--", color="k", alpha=0.4,
           label=f"max ({cfg.max_pseudoitems})")
ax.set(xlabel="Patterns learned (M)", ylabel="Pseudoitems found",
       xlim=(0, single.M[-1]+2))
ax.set_title(f"Pseudoitems per step (Pp={cfg.n_probes})")
ax.legend(); ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Run multiple trials

# %% Multiple trials
print("=" * 60)
print(f"ASYMMETRIC — Running {N_TRIALS} trials ...")
print("=" * 60)

t_start = time.time()
results = run_trials(cfg, n_trials=N_TRIALS, seed=SEED)
elapsed = time.time() - t_start
print(f"\nTotal: {elapsed:.1f}s ({elapsed/N_TRIALS:.1f}s per trial)")

# %% Plot averaged results
M    = results.M
mean = results.stable_mean

fig, ax = plt.subplots(figsize=(7, 5.5))

ax.plot(M, M, color="0.45", ls="-", lw=1.8, label="Perfect recall", zorder=1)
ax.plot(M, mean, color="#c0392b", ls="-", lw=2.5, label="Delta", zorder=2)

ax.set_xlabel("Patterns learned ($M$)", fontsize=18)
ax.set_ylabel("Stable patterns", fontsize=18)
ax.set_xlim(0, M[-1] + 2)
ax.set_ylim(0, M[-1] + 5)
ax.tick_params(labelsize=15, width=1.2, length=5)
ax.legend(loc="upper left", fontsize=16, frameon=False)

for spine in ax.spines.values():
    spine.set_linewidth(1.2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Summary

# %% Summary statistics
print("=" * 60)
print(f"SUMMARY (ASYMMETRIC) — N={cfg.network_size}, {N_TRIALS} trials")
print("=" * 60)
print(f"Stable at M={M[-1]}: {mean[-1]:.1f} +/- {std[-1]:.1f}")
for target_M in [20, 50]:
    hits = [i for i, m in enumerate(M) if m >= target_M]
    if hits:
        idx = hits[0]
        print(f"Stable at M={target_M}: {mean[idx]:.1f} +/- {std[idx]:.1f}")

# %% [markdown]
# ## (Optional) Save CSV
#
# Uncomment to export:
# ```python
# save_csv(results, "mccallum_1995_asymmetric_results.csv")
# ```

# %% CLI entry point (used when running as a script, ignored in notebook)
def _cli_main():
    import argparse
    p = argparse.ArgumentParser(
        description="McCallum 1995 pseudorehearsal (ASYMMETRIC delta rule)")
    p.add_argument("-N", "--network-size", type=int, nargs="+", default=[100])
    p.add_argument("--base-pop",    type=int,   default=5)
    p.add_argument("--max-new",     type=int,   default=95)
    p.add_argument("--probes",      type=int,   default=2000)
    p.add_argument("--max-pseudo",  type=int,   default=256)
    p.add_argument("--eta",         type=float, default=0.1)
    p.add_argument("--max-epochs",  type=int,   default=500)
    p.add_argument("--nu-h",        type=float, default=0.05)
    p.add_argument("--trials",      type=int,   default=10)
    p.add_argument("--seed",        type=int,   default=None)
    p.add_argument("--csv",         type=str,   default=None)
    p.add_argument("--plot-dir",    type=str,   default=None)
    p.add_argument("--no-plot",     action="store_true")
    p.add_argument("-q", "--quiet", action="store_true")
    args = p.parse_args()

    for N in args.network_size:
        c = McCallumConfig(
            network_size=N, base_pop=args.base_pop, max_new=args.max_new,
            n_probes=args.probes, max_pseudoitems=args.max_pseudo,
            eta=args.eta, max_epochs=args.max_epochs, nu_h=args.nu_h)

        print("=" * 60)
        print(f"ASYMMETRIC — N={N}, BP={c.base_pop}, new={c.max_new}, "
              f"Pp={c.n_probes}, Pi={c.max_pseudoitems}")
        print("=" * 60)

        res = run_trials(c, n_trials=args.trials,
                         seed=args.seed, verbose=not args.quiet)

        m = res.stable_mean
        print(f"\nStable at M={res.M[-1]}: {m[-1]:.1f} +/- {res.stable_std[-1]:.1f}")

        if args.csv:
            csv_path = Path(args.csv)
            if len(args.network_size) > 1:
                csv_path = csv_path.with_stem(f"{csv_path.stem}_N{N}")
            save_csv(res, csv_path)

        if not args.no_plot:
            save = None
            if args.plot_dir:
                save = Path(args.plot_dir) / f"mccallum_1995_asymmetric_N{N}.png"
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(res.M, res.stable_mean, "b-", lw=2)
            ax.fill_between(res.M, m - res.stable_std, m + res.stable_std, alpha=0.25)
            ax.plot(res.M, res.M, "k--", lw=1, alpha=0.4)
            ax.set(xlabel="Patterns learned (M)", ylabel="Stable patterns")
            ax.set_title(f"McCallum 1995 ASYMMETRIC — N={N}, {args.trials} trials")
            ax.grid(alpha=0.3)
            plt.tight_layout()
            if save:
                save.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(save, dpi=150, bbox_inches="tight")
                print(f"Plot -> {save}")
            else:
                plt.show()
        print()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        _cli_main()
