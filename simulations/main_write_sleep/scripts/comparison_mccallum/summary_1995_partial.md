# Summary of McCallum 1995 Pseudorehearsal Experiments

This document summarises the four related experiments that reproduce and extend McCallum's 1995 pseudorehearsal protocol. All experiments share the same base model (a discrete Hopfield network with asymmetric delta learning) and the same incremental incorporation loop. They differ in **what is measured** and **what is varied**.

---

## Common protocol

All experiments follow the same incorporation procedure:

1. **Base population** (BP = 5): the first 5 random bipolar patterns are trained together with no noise and no pseudorehearsal (pure delta learning).
2. **Incremental incorporation** (patterns 6 to $M_{\max}$): for each new pattern:
   - Probe the network with 2000 random states, relax each to convergence, and collect up to $P_{\text{items}}$ unique pseudoitems (rejecting duplicates and inverses).
   - Build a training set = pseudoitems + the new pattern.
   - Train with the delta rule ($\eta = 0.1$, up to 500 epochs). The new pattern receives heteroassociative noise ($\nu_h = 5\%$ bit flips) and Gaussian input noise ($\sigma = 0.5$); pseudoitems are trained clean.
   - Evaluate performance (stability and/or partial-cue recovery).

All experiments also include a **"without rehearsal" baseline** where step 2 is replaced by training only on the new pattern (no probing, no pseudoitems, no noise) -- pure iterative delta learning, demonstrating catastrophic forgetting.

### Shared parameters

| Parameter | Value |
|-----------|-------|
| Network size $N$ | 100 (unless swept) |
| Learning rate $\eta$ | 0.1 |
| Max epochs per step | 500 |
| Early stopping | smoothed error < 0.001 |
| Heteroassociative noise $\nu_h$ | 0.05 |
| Gaussian noise $\sigma$ | 0.5 |
| Probes $P_{\text{probes}}$ | 2000 |
| Pseudoitem cap $P_{\text{items}}$ | 256 (Pr256 condition) |
| Relaxation budget | $4N$ cycles |
| Patterns correlation $\rho$ | 0.0 (unless swept) |

---

## Experiment 1: `mccallum_1995`

**Goal:** Reproduce McCallum's Figure 4.23 -- compare pseudorehearsal with varying pseudoitem caps against delta-learning baselines.

**What varies across conditions (not across runs):**

| Condition | Pseudorehearsal? | Pseudoitem cap | Noise |
|-----------|-------------------|----------------|-------|
| Pr100 | Yes | 100 | Hetero + Gaussian |
| Pr256 | Yes | 256 | Hetero + Gaussian |
| Pr512 | Yes | 512 | Hetero + Gaussian |
| Iterative GDA | No | -- | Hetero only |
| Without rehearsal | No | -- | None |

**What is measured:** number of stable patterns (fixed-point check, 100% cue) at each $M$ from 5 to 100.

**Key question:** Does pseudorehearsal prevent catastrophic forgetting, and does the number of pseudoitems matter?

**Main result:** The "without rehearsal" baseline collapses rapidly (catastrophic forgetting). Pseudorehearsal maintains a plateau of ~15-20 stable patterns, scaling with the pseudoitem cap (Pr512 > Pr256 > Pr100). The Iterative GDA baseline (delta with heteroassociative noise, no pseudoitems) also forgets but less abruptly than the naive baseline due to the noise regularisation.

**Comparison with McCallum's original data:** The scaling with the number of pseudoitems and the order of magnitude of the average number of stable states match McCallum's results. However, we do not reproduce the "low-load bump" visible in McCallum's figures, where for $M < 0.14N$ all patterns become successfully stable when using pseudorehearsal. In our reproduction this bump is smaller -- for low $M$ the stable count is below perfect recall rather than matching it. Multiple conditions were tested (varying the noise levels, the Gaussian and heteroassociative noise parameters, and other aspects of the training and querying protocol) but the bump remained small. The important agreement is on the overall scaling behaviour and the order of magnitude of the stable-state plateau.

**Figures produced:**
- `mccallum_1995_comparison.png` -- Stable patterns vs $M$ for all conditions
- `mccallum_1995_pseudoitems.png` -- Number of pseudoitems found per step (Pr conditions only)

---

## Experiment 2: `mccallum_1995_partial_cue`

**Goal:** Go beyond stability and measure how well pseudorehearsal supports **pattern completion** from degraded cues.

**Difference from Experiment 1:**
- Only the **Pr256** condition and the **without rehearsal** baseline are run.
- After each incorporation step, all $M$ patterns are tested not just for stability (100% cue) but also for **recovery from partial cues** at four levels: 95%, 90%, 80%, and 50%.

A partial cue keeps the specified fraction of bits from the target pattern and randomises the rest, then the network relaxes and we check whether it converges to the target (or its inverse).

**What is measured:** for each $M$ and each cue level, the number of recovered patterns.

**Key question:** How does the quality of memory (measured by tolerance to degraded cues) degrade as more patterns are incorporated?

**Main result:** Recovery degrades gracefully from 95% cue down to 50% cue. At 50% cue, far fewer patterns are recovered than are technically stable -- meaning many stored patterns have very narrow basins of attraction. The "without rehearsal" baseline shows uniformly poor recovery at all cue levels.

**Comparison with McCallum's original data:** Same observation as Experiment 1 -- the overall level of recovered patterns and the benefit of pseudorehearsal over the naive baseline are consistent with McCallum's results, but the low-load bump (where all patterns are perfectly recovered for small $M$) is smaller in our reproduction. This holds across all cue levels.

**Figures produced:**
- `mccallum_1995_partial_cue.png` -- Recovery at each cue level vs $M$, with the "without rehearsal" baseline in dashed black

---

## Experiment 3: `mccallum_1995_partial_cue_rho`

**Goal:** Investigate how **pattern correlation** ($\rho$) affects pseudorehearsal performance under partial cues.

**Difference from Experiment 2:**
- Patterns are **correlated** instead of random. Correlation is controlled via a parent-and-redraw method: all patterns share a common parent, and each pattern re-randomises $(1 - \rho) \times N$ of its bits. $\rho = 0$ gives uncorrelated patterns; $\rho = 0.8$ gives highly similar patterns.
- Only the **80% cue level** is tested (plus stability).
- The experiment is repeated for $\rho \in \{0.0, 0.2, 0.4, 0.6, 0.8\}$.
- Both the **Pr256** condition and the **without rehearsal** baseline are run for each $\rho$.

**What is measured:** for each $(\rho, M)$ pair, the number of stable and recovered (80% cue) patterns.

**Key question:** Does pattern similarity make pseudorehearsal harder? Do correlated patterns interfere more during consolidation?

**Main result:** Higher $\rho$ makes both stability and recovery harder -- correlated patterns create overlapping basins and confuse the pseudorehearsal process. The "without rehearsal" baseline degrades even faster with high $\rho$, confirming that pseudorehearsal provides meaningful protection even for correlated patterns.

**Comparison with McCallum's original data:** Same observation as Experiments 1 and 2 -- the scaling behaviour and the order of magnitude of stable/recovered patterns are consistent with McCallum's results, but the low-load bump remains smaller in our reproduction. This discrepancy persists across all tested $\rho$ values.

**Figures produced:**
- `mccallum_1995_partial_cue_rho_recovery.png` -- 80% cue recovery vs $M$, one coloured line per $\rho$, dashed black for "without rehearsal"
- `mccallum_1995_partial_cue_rho_stability.png` -- Stability vs $M$, one coloured line per $\rho$, dashed black for "without rehearsal"

---

## Experiment 4: `mccallum_capacity_partial_cue`

**Goal:** Measure the **capacity** ($M^*$) of McCallum's pseudorehearsal as a function of network size and correlation, at multiple cue levels.

**Difference from Experiments 1-3:**
- This experiment uses the **$M^*$ protocol** rather than tracking a time-series of stable counts.
- Patterns are incorporated one by one from $M = 1$ (no base population).
- At each step, all $M$ patterns are tested for stability and for recovery at 95%, 80%, and 50% cues.
- $M^*_s(\text{cue})$ for a single run = the largest $M$ where **all** $M$ patterns are recovered at that cue level.
- The aggregate $M^*$ across seeds uses the 90th-percentile rule: $M^* = \max M$ such that $\geq 90\%$ of runs achieved $M^*_s \geq M$.
- The experiment sweeps over **network sizes** $N \in \{50, 100, 150, 200, 250\}$ and **correlations** $\rho \in \{0.0, 0.2, 0.4, 0.6, 0.8\}$, with 20 seeds per $(N, \rho)$.

**What is measured:** $M^*$ at each cue level, for each $(N, \rho)$ configuration.

**Key question:** How does pseudorehearsal capacity scale with network size, and how much does requiring partial-cue recovery (rather than just stability) reduce that capacity?

**Important note:** All simulations in this experiment use **Pr512** (pseudoitem cap of 512), which is the maximum number of pseudorehearsal data points McCallum uses.

**Comparison with McCallum:** This experiment cannot be directly compared to McCallum's original figures because it uses a different protocol ($M^*$ capacity metric rather than a time-series of stable counts). The observations below are specific to this scaling experiment.

**Main results:**
- As pattern correlation $\rho$ increases and/or the cue size decreases, the capacity for perfect retrieval drops. At 50% cue, capacity essentially collapses across all network sizes and correlations.
- Crucially, **storage capacity does not scale with network size**. $M^*$ remains roughly flat as $N$ increases from 50 to 250, suggesting that the pseudorehearsal mechanism itself -- not the network size -- is the bottleneck. This is a notable limitation of the method.
- Higher $\rho$ reduces $M^*$ at all cue levels, confirming that correlated patterns are harder to consolidate via pseudorehearsal.

**Figures produced:**
- `mccallum_capacity_partial_cue.png` -- One subplot per $\rho$, with $M^*$ vs $N$ lines for each cue level

---

## Overview table

| Experiment | Conditions | Patterns $\rho$ | Evaluation metric | Network sizes | Key axis of variation |
|-----------|------------|------------------|-------------------|---------------|----------------------|
| `mccallum_1995` | Pr100, Pr256, Pr512, GDA, naive | 0.0 | Stability only | 100 | Pseudoitem cap |
| `mccallum_1995_partial_cue` | Pr256, naive | 0.0 | Stability + 4 cue levels | 100 | Cue level |
| `mccallum_1995_partial_cue_rho` | Pr256, naive | 0.0--0.8 | Stability + 80% cue | 100 | Pattern correlation $\rho$ |
| `mccallum_capacity_partial_cue` | Pr256 | 0.0--0.8 | $M^*$ at 4 cue levels | 50--250 | Network size $N$ |

---

## Narrative thread for the thesis

These four experiments build on each other:

1. **Experiment 1** establishes that pseudorehearsal works and that more pseudoitems help (reproduction of McCallum's result). The "without rehearsal" baseline quantifies catastrophic forgetting.

2. **Experiment 2** shows that stability alone overestimates memory quality -- many stable patterns are only recoverable with near-perfect cues, meaning their basins of attraction are very narrow.

3. **Experiment 3** adds pattern correlation as a difficulty factor and shows that pseudorehearsal degrades gracefully with increasing $\rho$, while the naive baseline collapses.

4. **Experiment 4** measures capacity as a scalar ($M^*$) across a grid of $(N, \rho)$, producing the kind of scaling curves that can be directly compared to other methods (AR, Hebbian, Storkey) in the broader comparison framework.
