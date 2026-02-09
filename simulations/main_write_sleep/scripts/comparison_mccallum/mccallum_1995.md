# McCallum 1995 — Pseudorehearsal in Discrete Hopfield Networks

Reproduction of the pseudorehearsal experiment from McCallum's thesis
(Figure 4.23), comparing pseudorehearsal with varying pseudoitem caps
against pure delta-learning baselines.

---

## 1. Model: Discrete Hopfield Network (DHN)

A fully connected recurrent network with bipolar units.

| Property | Value |
|----------|-------|
| Units | $N$ bipolar neurons, $\psi_i \in \{-1, +1\}$ |
| Weights | $N \times N$ real-valued matrix $W$, initialised to zero |
| Self-connections | None: $W_{ii} = 0$ enforced after every update |
| Symmetry | **Asymmetric** ($W_{ij} \neq W_{ji}$) — natural for delta learning |
| Bias | None |

### 1.1 Dynamics (asynchronous relaxation)

At each step one randomly selected unit $i$ is updated:

$$
\psi_i \leftarrow \mathrm{sign}\!\Bigl(\sum_{j \neq i} W_{ij}\,\psi_j\Bigr)
\qquad \text{where } \mathrm{sign}(h) =
\begin{cases} +1 & h \ge 0 \\ -1 & h < 0 \end{cases}
$$

One **cycle** = $N$ random unit updates (each unit updated once on average).
Relaxation runs for at most $r = 4N$ cycles, stopping early if no unit
changes during a full cycle (convergence).

### 1.2 Stability check

A pattern $\boldsymbol{s}^\mu$ is **stable** if relaxing from $\boldsymbol{s}^\mu$
itself returns $\boldsymbol{s}^\mu$ unchanged:

$$
\mathrm{relax}(W, \boldsymbol{s}^\mu, r{=}4N) = \boldsymbol{s}^\mu
$$

This is the evaluation metric used in the plots (y-axis = number of stable
patterns out of $M$).

---

## 2. Learning: synchronous delta rule

The weight update for one training pattern (McCallum Eq. 2.7):

$$
\Delta W_{ij} = \eta\,(D_i - \psi_i)\,\psi_j^{\text{input}}
$$

| Symbol | Meaning |
|--------|---------|
| $\eta = 0.1$ | Learning rate |
| $D_i$ | Desired output (target pattern value $s_i^\mu$) |
| $\psi_i = \mathrm{sign}(h_i)$ | Actual output of unit $i$ |
| $\psi_j^{\text{input}}$ | Input presented to the network (possibly noisy) |

After each weight update: $W_{ii} \leftarrow 0$.

### 2.1 Noise (applied to new patterns only)

Two noise sources regularise learning of the new pattern. Neither is
applied to pseudoitems.

| Noise type | Application | Effect |
|------------|-------------|--------|
| **Heteroassociative** $\nu_h = 0.05$ | Before computing outputs, flip 5% of randomly chosen input bits | Builds basins of attraction around the pattern |
| **Gaussian input** $\sigma = 0.5$ | Add $\mathcal{N}(0, 0.5^2)$ to each unit's local field before sign | Smooths the energy landscape |

### 2.2 Training loop (one incorporation step)

Given training set $\mathcal{T}$ (pseudoitems + new pattern), train for up
to $E_{\max} = 500$ epochs:

```
smoothed_error = 1.0

for epoch = 1 to 500:
    shuffle T
    epoch_errors = 0

    for each pattern s in T:
        if s is the NEW pattern:
            input = flip_noise(s, nu_h=0.05)        # heteroassociative
        else:                                         # pseudoitem
            input = s                                 # no noise

        for each unit i:
            h_i = sum_{j!=i} W[i,j] * input[j]
            if s is the NEW pattern:
                h_i += Normal(0, 0.5)                 # Gaussian input noise
            psi_i = sign(h_i)
            error_i = s[i] - psi_i

            if |error_i| > 0.5:                       # error is +/-2 or 0
                W[i,:] += eta * error_i * input
                W[i,i] = 0
                epoch_errors += |error_i| / 2

    smoothed_error = 0.9 * smoothed_error + 0.1 * epoch_errors
    if smoothed_error < 0.001:
        break                                         # early stopping
```

---

## 3. Pseudorehearsal: probing for pseudoitems

Before incorporating a new pattern, probe the network to discover its
current stable states. These become the **pseudoitems** — a compressed
memory of what the network already knows.

```
pseudoitems = []
seen = {}

for probe = 1 to P_probes (=2000):
    state = random {-1, +1}^N              # uniform random probe
    state = relax(W, state, r=4N)          # converge to attractor

    if state and -state not in seen:        # reject duplicates + inverses
        seen.add(state)
        pseudoitems.append(state)

    if |pseudoitems| >= P_items:            # cap reached
        break

return pseudoitems
```

| Parameter | Value | Description |
|-----------|-------|-------------|
| $P_{\text{probes}}$ | 2000 | Random probes sent into the network |
| $P_{\text{items}}$ | 100 / 256 / 512 | Maximum unique pseudoitems to keep (varies by condition) |

In practice the actual number found is often well below the cap, because
many probes converge to the same attractors.

---

## 4. Full protocol

### 4.1 Base population (BP = 5)

Train the first 5 patterns together as a batch with **no noise** and
**no pseudorehearsal** (pure delta learning on all 5 simultaneously).

### 4.2 Incremental incorporation (patterns 6 to 100)

For each new pattern $M = 6, 7, \ldots, 100$:

1. **Probe** the current network to collect pseudoitems (Section 3)
2. **Build training set** $\mathcal{T} = \{\text{pseudoitems}\} \cup \{s^M\}$
3. **Train** with delta learning (Section 2.2):
   - Noise on $s^M$ only, pseudoitems trained clean
4. **Evaluate** stability of all $M$ patterns (Section 1.2)
5. Record: $M$, number of stable patterns, number of pseudoitems found

The loop does **not** stop on failure — it tracks the stable count
through all 100 patterns.

---

## 5. Experimental conditions

Five conditions are tested on the same protocol. They differ only in
what happens at step 2-3 above:

| Condition | Mode | Pseudorehearsal? | Noise on new pattern | Pseudoitem cap |
|-----------|------|-------------------|---------------------|----------------|
| **Pr100** | 0 | Yes | Hetero + Gaussian | 100 |
| **Pr256** | 0 | Yes | Hetero + Gaussian | 256 |
| **Pr512** | 0 | Yes | Hetero + Gaussian | 512 |
| **Delta (hetero)** | 1 | No | Hetero only | N/A |
| **Delta (Gaussian)** | 2 | No | Gaussian only | N/A |

The **Delta** baselines incorporate each new pattern by training only
on that single pattern (no probing, no pseudoitems). This demonstrates
catastrophic forgetting — the network overwrites old memories.

The **Pr** conditions show that pseudorehearsal mitigates forgetting:
probed pseudoitems preserve old memories while the new pattern is learned.

---

## 6. Parameters summary

### Network

| Parameter | Value |
|-----------|-------|
| $N$ | 100 |
| Activations | $\{-1, +1\}$ |
| $W$ init | zeros |
| $W_{ii}$ | 0 (enforced) |
| Symmetry | Asymmetric |

### Learning

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Learning rate | $\eta$ | 0.1 |
| Max epochs per step | $E_{\max}$ | 500 |
| Early-stop threshold | $\delta_l$ | 0.001 |
| Error smoothing | | 0.9 |
| Hetero noise rate | $\nu_h$ | 0.05 (5%) |
| Gaussian noise std | $\sigma$ | 0.5 |

### Pseudorehearsal

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Probes | $P_{\text{probes}}$ | 2000 |
| Pseudoitem cap | $P_{\text{items}}$ | 100 / 256 / 512 |
| Inverse rejection | | Yes |

### Protocol

| Parameter | Value |
|-----------|-------|
| Base population | BP = 5 |
| New patterns | 95 (total M = 100) |
| Relaxation cycles | $r = 4N = 400$ |
| Trials per condition | 15 |
| Pattern correlation $\rho$ | 0.0 (uncorrelated) |
| Pattern sparsity | 0.5 |

### 7.2 Step 1 — Run simulations

```bash
python3 scripts/comparison_mccallum/mccallum_1995_sim_cpp.py
```

This script:

1. Builds the C++ binary (`make bin/mccallum`)
2. For each of the 5 conditions, generates a JSON config:
   ```json
   {
     "type": "mccallum",
     "output_dir": "data/mccallum_results/mccallum_1995_raw/pr256",
     "base_params": {
       "network_size": 100,
       "max_patterns": 100,
       "rho": 0.0,
       "base_pop": 5,
       "stop_on_failure": 0,
       "mode": 0,
       "max_pseudoitems": 256,
       "eta": 0.1,
       "max_epochs": 500,
       "n_probes": 2000,
       "nu_h": 0.05,
       "sigma_input": 0.5
     },
     "varying_params": {
       "seed": [0, 1, 2, ..., 14]
     }
   }
   ```
3. Calls `bin/mccallum config.json` (C++ runs all seeds in parallel)
4. Parses `results.data` from each `sim_nb_X/` directory
5. Aggregates into `data/mccallum_results/mccallum_1995/{condition}.csv`

### 7.3 Step 2 — Generate plots

```bash
python3 scripts/comparison_mccallum/mccallum_1995_viz.py
```

This script:

1. Loads each `{condition}.csv`
2. Groups by $M$, computes mean and std of stable count across trials
3. Plots the main comparison figure:
   - x-axis: Patterns learned ($M$)
   - y-axis: Stable patterns (mean across trials)
   - Gray diagonal: perfect recall reference
   - Red line: Iterative GDA (delta_hetero)
   - Coloured lines: Pr100, Pr256, Pr512
4. Plots the pseudoitems figure (Pr conditions only):
   - x-axis: Patterns learned ($M$)
   - y-axis: Number of pseudoitems found
   - Horizontal dashed lines: cap for each condition (100, 256, 512)
5. Prints summary statistics

### 7.4 Editable knobs

In `mccallum_1995_sim_cpp.py`:

| Variable | Default | Effect |
|----------|---------|--------|
| `NETWORK_SIZE` | 100 | Number of neurons |
| `BASE_POP` | 5 | Base population size |
| `MAX_PATTERNS` | 100 | Total patterns (BP + new) |
| `N_TRIALS` | 15 | Independent repetitions per condition |
| `SEED_OFFSET` | 0 | Starting seed value |
| `RUN` | all 5 | Which conditions to run |

In `mccallum_1995_viz.py`:

| Variable | Default | Effect |
|----------|---------|--------|
| `SHOW` | `["delta_hetero", "pr100", "pr256", "pr512"]` | Which conditions appear in the plot |
| `STYLE` | (dict) | Colours, line styles, labels |

## 10. Expected results

From McCallum's Figure 4.23 ($N = 100$, uncorrelated patterns):

- **Delta baselines** (no pseudorehearsal): stable count collapses
  rapidly. By $M \approx 15$ the network retains very few patterns —
  classic catastrophic forgetting.

- **Pr100**: Moderate protection. The 100-pseudoitem cap limits memory
  preservation. Stable count stays above the delta baseline but well
  below perfect recall.

- **Pr256**: Standard McCallum result. After an initial dip around
  $M \approx 15$, the stable count slowly recovers and plateaus around
  15-20 stable patterns.

- **Pr512**: Best pseudorehearsal performance. More pseudoitems means
  better memory preservation. The dip is shallower and the plateau is
  higher.

The key insight: pseudorehearsal prevents catastrophic forgetting by
rehearsing the network's own stable states alongside new patterns.
More pseudoitems = better preservation, but with diminishing returns
since many probes converge to the same attractors.

---

## 11. Relationship to the broader comparison

This experiment reproduces McCallum's original figure. The broader
thesis comparison (`mccallum_sim.py` + `viz_mccallum.py`) runs the
same C++ backend across a grid of network sizes and correlation values,
computing $M^*$ (the maximum patterns retrievable with 90% reliability
under 50% partial cues). That comparison places McCallum's method
alongside:

| Method | Network | Learning | Consolidation |
|--------|---------|----------|---------------|
| McCallum PR | DHN | Delta + pseudorehearsal | Probing (spurious = pseudoitem) |
| AR (CI) | CHN | Batch GDA | Sleep (spurious = failure) |
| Hebbian | DHN | One-shot Hebbian | None |
| Storkey | DHN | One-shot Storkey | None |
