# McCallum Pseudorehearsal Comparison

This directory contains scripts for comparing four memory capacity methods:

1. **McCallum Pseudorehearsal** - DHN with delta learning + probing
2. **AR (Continuous Incorporation)** - CHN with sleep consolidation
3. **Hebbian** - DHN one-shot learning
4. **Storkey** - DHN incremental learning

## Experimental Grid

| Parameter | Values |
|-----------|--------|
| Network size N | 50, 100, 150, 200, 250 |
| Correlation rho | 0.0, 0.2, 0.4, 0.6 |
| Simulations S | 10 per (N, rho) |
| Success threshold theta | 0.9 |
| Max patterns M_max | 50 |

**Total**: 5 x 4 = 20 configurations, 200 simulation runs per method.

## Key Algorithmic Differences

| Aspect | McCallum | AR (CI) | Hebbian/Storkey |
|--------|----------|---------|-----------------|
| Network type | DHN ({+1,-1}) | CHN (continuous [0,1]) | DHN ({+1,-1}) |
| Learning | Delta rule + noise | Batch GDA | One-shot |
| Consolidation | Pseudorehearsal (probes) | Sleep (autonomous retrieval) | None |
| Spurious during consolidation | OK (becomes pseudoitem) | **FAILURE** | N/A |
| Spurious during query | FAILURE | FAILURE | FAILURE |

## Running the Comparison

### Step 1: Build the C++ simulations

```bash
cd /path/to/main_write_sleep
make
```

This will compile `bin/mccallum`, `bin/ar_incremental`, `bin/dhn_train`, and `bin/dhn_query`.

### Step 2: Run McCallum simulations

Open `mccallum_sim.py` in VSCode and run each cell:
1. Configuration: sets up parameters
2. Build & run: launches C++ simulations
3. Collect: aggregates results to CSV

### Step 3: Run AR simulations

Open `ar_sim.py` in VSCode and run each cell.

### Step 4: Run DHN simulations

Open `dhn_sim.py` in VSCode and run each cell.
This uses the existing `dhn_train` and `dhn_query` infrastructure.

### Step 5: Generate comparison figure

Open `viz_mccallum.py` in VSCode and run all cells.
Output: `scripts/plots/mccallum_comparison.png`

## Data Directory Structure

```
data/mccallum_results/
├── mccallum/
│   ├── sim_nb_0/ ... sim_nb_199/
│   │   ├── results.data       # Per-M metrics
│   │   ├── patterns.data      # Stored patterns
│   │   ├── weights.data       # Final weight matrix
│   │   └── parameters.data    # Config + M*
│   └── all_simulation_data.csv
├── ar/
│   ├── sim_nb_0/ ... sim_nb_199/
│   └── all_simulation_data.csv
├── hebbian/
│   └── all_simulation_data.csv
└── storkey/
    └── all_simulation_data.csv
```

## McCallum Algorithm Details

### Delta Learning Rule

```
Delta W_ij = eta * (D_i - psi_i) * psi_j^input
```

Where:
- eta = 0.1 (learning rate)
- D_i = desired output (pattern value)
- psi_i = actual output after sign(h_i)
- psi_j^input = input (possibly noisy)

### Training Noise (new patterns only)

1. **Heteroassociative noise** (nu_h = 5%): flip 5% of input bits
2. **Input noise** (sigma = 0.5): add Gaussian noise to local field

Pseudoitems receive NO noise during training.

### Probing Phase

Before each new pattern (for M > 1):
1. Generate 2000 random probes
2. Relax each to stable state (4N cycles max)
3. Collect up to 256 unique pseudoitems
4. Include both learnt patterns and spurious states

### Evaluation

For each M, query all patterns 1..M with 50% partial cues.
If any query fails: M* = M-1 and stop.

## AR Algorithm Details

### Sleep Phase

After adding pattern M (for M > 1):
1. Reset inhibition
2. Run autonomous retrievals from neutral state
3. Apply diagonal inhibitory plasticity after each retrieval
4. If spurious state found: FAIL (M* = M-1)
5. Continue until all patterns retrieved

### Training

After successful sleep:
- Retrain on all current patterns using batch GDA
- Uses momentum optimization

### Evaluation

Same as McCallum: 50% partial cue queries.

## Computing M*

For each (N, rho) configuration:
1. Run S=10 simulations, each returning M*_s
2. M* = max M such that >= 90% of simulations achieved M*_s >= M

```python
def compute_M_star(M_star_list, theta=0.9):
    for M in range(max(M_star_list), -1, -1):
        fraction = sum(1 for m in M_star_list if m >= M) / len(M_star_list)
        if fraction >= theta:
            return M
    return 0
```

## Parameter Reference

### McCallum DHN

| Parameter | Value | Description |
|-----------|-------|-------------|
| eta | 0.1 | Learning rate |
| E_max | 500 | Max epochs per incorporation |
| error_criterion | 0.001 | Early stopping threshold |
| nu_h | 0.05 | 5% heteroassociative noise |
| sigma_input | 0.5 | Gaussian input noise std |
| P_probes | 2000 | Number of probes |
| P_items | 256 | Max pseudoitems |
| max_cycles | 4*N | Relaxation limit |

### AR (CHN)

| Parameter | Value | Description |
|-----------|-------|-------------|
| leak | 1.0 | Membrane time constant |
| drive_target | 6.0 | Target activation |
| learning_rate | 0.0001 | GDA learning rate |
| momentum_coef | 0.9 | Momentum coefficient |
| delta | 0.01 | Integration timestep |
| beta | 0.1 | Inhibitory plasticity rate |
| stddev_dynamics | 0.01 | Noise std |
| max_sleep_queries | 100 | Max retrieval cycles |

### DHN (Hebbian/Storkey)

- One-shot learning (no iterative training)
- Hebbian: W_ij += (1/N) * xi_i * xi_j
- Storkey: W_ij += (1/N) * [xi_i*xi_j - xi_i*h_j - h_i*xi_j]

## Expected Results

Based on McCallum's Figure 4.23 (N=100, rho=0):
- McCallum M* ~ 10-15 patterns
- Hebbian M* ~ 13.8% of N = ~14 patterns (theoretical)
- Storkey M* ~ 42% of N = ~42 patterns (theoretical)

The AR method's capacity depends on sleep effectiveness.

## Sanity Checks

Before running full grid, validate with:
- N=100, rho=0.0, S=5
- Expected McCallum M* ~ 10-15

Log the number of pseudoitems found per incorporation to verify it's typically well below 256.

## References

- McCallum, R. A. (2007). "Catastrophic Forgetting and the Pseudorehearsal Solution in Hopfield Networks." PhD Thesis.
- Hopfield, J. J. (1982). "Neural networks and physical systems with emergent collective computational abilities."
- Storkey, A. (1997). "Increasing the capacity of a Hopfield network without sacrificing functionality."
