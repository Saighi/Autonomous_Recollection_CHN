# McCallum Pseudorehearsal: Computing M* for Thesis Comparison

## 1. Objective

Compute the perfect retrieval capacity $M^*$ using McCallum's pseudorehearsal method (from his 2007 PhD thesis, "Catastrophic Forgetting and the Pseudorehearsal Solution in Hopfield Networks"). The results will be added to the comparison figure (Fig. comparison_summary) in the thesis alongside:
- Continuous Incorporation (CI) — CHN
- Hebbian plasticity — DHN
- Storkey learning rule — DHN
- Iterative GDA — CHN

The evaluation must use **the same protocol** as these existing methods (Alg. evaluation_procedure in the manuscript).

---

## 2. Experimental Grid

| Parameter        | Values                                      |
|------------------|---------------------------------------------|
| Network size $N$ | 50, 100, 150, 200, 250                      |
| Correlation $\rho$ | 0.0, 0.2, 0.4, 0.6                        |
| Simulations $S$  | 10 per $(N, \rho)$ configuration            |
| Success threshold $\theta$ | 0.9                              |
| Max patterns $M_{\max}$ | 50                                    |

Total configurations: 5 × 4 = 20. Total simulation runs: 20 × 10 = 200.

---

## 3. Pattern Generation

Use the same correlated pattern generation as in the thesis (Alg. correlated_patterns). Patterns are generated in **{0, 1} space** and then converted to **{+1, −1} space** for the DHN.

### 3.1. Algorithm (Alg. correlated_patterns)

```
Input: N (network size), p (number of patterns), ρ (correlation parameter)

1. Generate parent pattern x_parent ∈ {0,1}^N where each bit is 0 or 1 with prob 0.5
2. Set k = floor((1 − ρ) × N)    # number of bits to randomize
3. For each pattern μ = 1, ..., p:
   a. Copy: x^μ = x_parent
   b. Randomly select k distinct indices from {1, ..., N}
   c. For each selected index j: set x^μ_j = random choice from {0, 1} with equal prob
```

### 3.2. Conversion to bipolar

For use in the DHN, convert each pattern:

$$s^{\mu}_i = 2 x^{\mu}_i - 1 \quad \Rightarrow \quad s^{\mu}_i \in \{-1, +1\}$$

**Important**: Store both representations. The {0, 1} patterns are the "ground truth" for reporting results (consistent with the other methods in the thesis). The {+1, −1} patterns are used internally by the DHN.

---

## 4. Network Architecture: Discrete Hopfield Network (DHN)

- **Units**: $N$ bipolar neurons with states $\psi_i \in \{-1, +1\}$
- **Weights**: $N \times N$ real-valued matrix $W$, initialized to all zeros
- **No self-connections**: $W_{ii} = 0$ enforced at all times
- **Asymmetric weights**: $W_{ij} \neq W_{ji}$ in general (delta learning produces asymmetric weights; this is consistent with McCallum's implementation)
- **No bias**: bias unit is not used

### 4.1. Update Rule (retrieval dynamics)

Asynchronous update: at each step, one randomly selected unit $i$ is updated:

$$\psi_i \leftarrow \text{sign}\left(\sum_{j \neq i} W_{ij} \psi_j\right)$$

where $\text{sign}(h) = +1$ if $h \geq 0$, $-1$ otherwise.

**Relaxation**: repeat asynchronous updates for a maximum of $r_{\max} = 4N$ **cycles**, where one cycle = $N$ random unit updates (so each unit is updated on average once per cycle). Stop early if no unit changes state during a full cycle (convergence reached).

---

## 5. McCallum's Pseudorehearsal Algorithm

The core idea: when incorporating a new pattern, first **probe** the current network to discover its stable states (both learnt and spurious), then **retrain** the network on the new pattern together with these recovered stable states (pseudoitems).

### 5.1. Delta Learning Rule

For auto-associative learning in the DHN, the delta rule is:

$$\Delta W_{ij} = \eta \, (D_i - \psi_i) \, \psi_j^{\text{input}}$$

where:
- $\eta = 0.1$ is the learning constant
- $D_i$ is the desired output (= original pattern value $s_i^{\mu}$)
- $\psi_i$ is the actual output: $\psi_i = \text{sign}(h_i)$ with $h_i = \sum_j W_{ij} \psi_j^{\text{input}} + \text{noise}_i$
- $\psi_j^{\text{input}}$ is the input presented to the network (possibly noisy)

After each weight update, enforce $W_{ii} = 0$.

### 5.2. Noise During Training

Two types of noise, applied **only to the new pattern** (NOT to pseudoitems):

**(a) Heteroassociative noise** ($\nu_h = 5\%$): Before computing outputs, flip 5% of randomly chosen input bits. This creates a slightly heteroassociative task that builds basins of attraction.

```
input = copy(pattern)
n_flip = round(0.05 * N)
flip_indices = random_choice(N, n_flip, replace=False)
input[flip_indices] *= -1
```

**(b) Input noise** (absolute Gaussian, $\sigma = 0.5$): Add Gaussian noise to each unit's local field before applying the sign function.

```
h_i = sum_j(W_ij * input_j) + Normal(0, 0.5^2)
```

**For pseudoitems**: no heteroassociative noise, no input noise. The pseudoitems are trained as a pure auto-associative task.

### 5.3. Training Procedure (one incorporation step)

Given the current weight matrix $W$, the training set $\mathcal{T}$ (new pattern + pseudoitems), train for up to $E_{\max} = 500$ epochs. Each epoch:

```
epoch_errors = 0
for each pattern s in shuffled(T):
    if s is the new pattern:
        input = apply_heteroassociative_noise(s, nu_h=0.05)
        for each unit i:
            h_i = sum_j(W[i,j] * input[j]) + Normal(0, 0.5^2)
            psi_i = sign(h_i)
            error_i = s[i] - psi_i
            if error_i != 0:
                W[i, :] += eta * error_i * input  # vector update
                W[i, i] = 0                        # enforce no self-connection
            epoch_errors += abs(error_i) / 2       # count errors (error is ±2 or 0)
    else:  # pseudoitem
        for each unit i:
            h_i = sum_j(W[i,j] * s[j])  # no noise
            psi_i = sign(h_i)
            error_i = s[i] - psi_i
            if error_i != 0:
                W[i, :] += eta * error_i * s
                W[i, i] = 0
            epoch_errors += abs(error_i) / 2

# Early stopping: smoothed error criterion
smoothed_error = smoothed_error * 0.9 + epoch_errors
if smoothed_error < error_criterion:  # error_criterion = 0.001
    break
```

### 5.4. Probing Phase (generating pseudoitems)

Before each new pattern incorporation (for patterns 2, 3, ..., $M$):

```
pseudoitems = []
seen = set()

for probe = 1 to P_probes (= 2000):
    # Generate random probe: each unit +1 or -1 with equal probability
    state = random_choice({-1, +1}, size=N)
    
    # Relax to stable state
    state = relax(W, state, max_cycles=4*N)
    
    # Store unique stable states
    key = tuple(state)
    if key not in seen:
        seen.add(key)
        pseudoitems.append(state)
    
    if len(pseudoitems) >= P_items (= 256):
        break

# Also discard the inverse (-state) if it duplicates a known state
```

The relaxation function uses asynchronous updates as described in Section 4.1.

### 5.5. Full Incorporation Procedure

```
Input: pattern sequence s^1, s^2, ..., s^M_max (in {+1, -1})

Initialize W = zeros(N, N)

For M = 1 to M_max:
    if M == 1:
        # First pattern: train with delta learning alone
        T = {s^1}
    else:
        # Probe network to find pseudoitems
        pseudoitems = probe_network(W, P_probes=2000, P_items=256)
        T = {s^M} ∪ pseudoitems    # new pattern + pseudoitems
    
    # Train network on T
    train_delta(W, T, new_pattern=s^M, epochs=500, eta=0.1, 
                error_criterion=0.001)
    
    # === EVALUATION (from Alg. evaluation_procedure) ===
    # Query all stored patterns with 50% partial cues
    all_retrieved = True
    for mu = 1 to M:
        success = query_pattern(W, s^mu, cue_fraction=0.5)
        if not success:
            all_retrieved = False
            break
    
    if not all_retrieved:
        M_star_s = M - 1
        break
    else:
        M_star_s = M

return M_star_s
```

**Important distinction:** Spurious states recovered during the probing phase (Section 5.4) do **not** constitute a failure — they are collected as pseudoitems alongside learnt patterns. This contrasts with CI, where encountering a spurious state during sleep terminates the incorporation. For McCallum's method, failure is determined **only** during the querying phase: if any stored pattern, when queried with a 50% partial cue, does not return the correct pattern (whether the network converges to a spurious state or to a different stored pattern), the process breaks and $M^*_s = M - 1$.

---

## 6. Query Procedure (50% Partial Cues)

This must be **identical** to the query procedure used for the other methods in the comparison.

```
function query_pattern(W, target_pattern, cue_fraction=0.5):
    N = len(target_pattern)
    n_informed = round(cue_fraction * N)
    
    # Create partial cue
    informed_indices = random_choice(N, n_informed, replace=False)
    cue = random_choice({-1, +1}, size=N)  # random for uninformed units
    cue[informed_indices] = target_pattern[informed_indices]  # set informed units
    
    # Relax
    result = relax(W, cue, max_cycles=4*N)
    
    # Check if retrieval is correct
    return np.array_equal(result, target_pattern)
```

---

## 7. Computing $M^*$

For each configuration $(N, \rho)$:

```
For s = 1 to S (= 20):
    Generate M_max patterns with correlation ρ (in {0,1}, then convert to {+1,-1})
    Run the incorporation procedure → get M_star_s[s]

# Compute M*: maximum M such that at least fraction θ of simulations achieved M_star_s ≥ M
For M = M_max down to 0:
    fraction_success = count(M_star_s >= M) / S
    if fraction_success >= theta (= 0.9):
        M_star = M
        break
else:
    M_star = 0
```

---

## 8. All Parameters Summary

### McCallum's DHN Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Unit type | {+1, −1} | Bipolar units (sign activation) |
| $W_{ii}$ | 0 | No self-connections |
| Symmetric weights | No | Asymmetric (natural for delta learning) |
| Bias | No | No bias unit |

### Delta Learning Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| $\eta$ | 0.1 | Learning constant |
| $E_{\max}$ | 500 | Maximum training epochs per incorporation |
| $\delta_l$ | 0.001 | Error criterion for early stopping |
| Error tail | 0.9 | Smoothing factor for error criterion |
| $\nu_h$ | 0.05 (5%) | Heteroassociative noise (new patterns only) |
| $\sigma_{\text{input}}$ | 0.5 | Absolute Gaussian input noise std (new patterns only) |
| Pseudoitem noise | None | No noise applied to pseudoitems |

### Pseudorehearsal Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| $P_{\text{probes}}$ | 2000 | Number of random probes to find stable states |
| $P_{\text{items}}$ | 256 | Maximum number of unique pseudoitems to collect |
| Coding ratio for probes | 0.5 | Each probe unit is +1 or −1 with equal probability |
| Unique pseudoitems | Yes | Only store distinct stable states |

**Note on pseudoitem count vs network capacity:** The cap of 256 pseudoitems is McCallum's parameter for N=100 and is kept fixed for all N. In practice, the actual number of unique stable states found by 2000 probes is typically much lower than 256 — especially for small or lightly loaded networks, because many probes converge to the same attractors. The pseudoitems are already stable states of the current weight matrix, so delta learning on them requires minimal correction (near-zero initial errors). The heavy lifting is only on the one new pattern. Training often hits the 500-epoch limit without fully converging, which is expected and explains the limited effective capacity of the method.

### Relaxation Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Update type | Asynchronous | One random unit updated per step |
| Max cycles | $4N$ | Maximum relaxation duration |
| Convergence | No change in full cycle | Stop early if stable |

### Evaluation Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| $S$ | 10 | Independent simulations per configuration |
| $\theta$ | 0.9 | Success threshold for $M^*$ |
| $M_{\max}$ | 50 | Maximum patterns tested |
| Cue fraction | 0.5 | 50% partial cues for querying |
| Pattern space | {0, 1} | Patterns generated in binary space |

---

## 9. Implementation Notes

### 9.1. Computational Cost

The dominant cost is the **delta learning training** (up to 500 epochs per incorporation) and the **probing phase** (2000 relaxations per incorporation).

**Estimated cost per simulation** (rough):
- Per training epoch: $O(N^2 \times |\mathcal{T}|)$ where $|\mathcal{T}| \leq 257$
- Per incorporation: up to $500 \times 257 \times N^2$ operations (training) + $2000 \times 4N \times N$ operations (probing)
- Per simulation: up to $50$ incorporations (but typically far fewer before failure)

For $N = 50$: each simulation should complete in seconds.
For $N = 100$: each simulation should complete in seconds to a minute.
For $N = 250$: each simulation may take a few minutes.

The full grid (200 runs) should complete in under an hour with parallelization.

---

## 10. Key Differences Between McCallum's Method and CI

For reference when writing the thesis text describing this comparison:

| Aspect | McCallum's Pseudorehearsal | Continuous Incorporation (CI) |
|--------|---------------------------|-------------------------------|
| Network type | DHN (discrete, bipolar {+1,−1}) | CHN (continuous) |
| Learning rule | Delta learning | GDA |
| Sleep/rehearsal | Random probes → stable states (pseudoitems) | Autonomous retrieval via self-inhibition |
| What is rehearsed | All stable states (learnt + spurious) | Only stored patterns (via AR) |
| Spurious states | Treated as useful information; rehearsed alongside learnt patterns | Treated as failure; sleep aborts if spurious state found |
| Noise in training | Heteroassociative + input noise on new patterns | Not applicable (GDA uses gradient) |
| Weight symmetry | Asymmetric | Symmetric |
| Locality | Local (delta rule) | Local (GDA) |
| External memory | None needed (pseudoitems extracted from network itself) | None needed (AR extracts patterns autonomously) |

---

## 11. Sanity Check

Before running the full grid, validate the implementation by reproducing McCallum's results for $N = 100$ with **uncorrelated random patterns** ($\rho = 0.0$, equivalent to his 50% coding ratio random patterns).

Expected behavior (from McCallum's Figure 4.23):
- Simple pseudorehearsal (Pr256) in $H_{100,\pm}$: approximately 10–15 stable patterns after initial dip, slowly increasing to ~18 stable patterns by pattern 95.
- This means for our $M^*$ metric with $\theta = 0.9$, we should expect $M^* \approx 10$–$15$ for $N = 100$, $\rho = 0.0$.

Run a quick test with $N = 100$, $\rho = 0.0$, $S = 5$ and verify the order of magnitude is correct before launching the full grid.

**Additional sanity check**: for each run, log the actual number of unique pseudoitems found (to verify it is typically well below 256, especially for small N and low loads).
