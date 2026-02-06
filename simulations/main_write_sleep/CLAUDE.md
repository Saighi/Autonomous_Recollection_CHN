# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
# Build all simulations
make

# Build specific simulation
make bin/write_leak
make bin/sleep_leak

# List available simulations
make list

# Clean build artifacts
make clean
make distclean  # Also removes obj/ and bin/ directories
```

To add a new simulation: create `simulations/your_sim.cc` and run `make`. The makefile auto-discovers all `.cc` files in `simulations/`.

## Architecture Overview

This is a neural network simulation framework for studying memory storage and sleep-based memory consolidation using two network types:

1. **CHN (Continuous Hopfield Network)**: Sigmoid activations [0,1], batch GDA learning
2. **DHN (Discrete Hopfield Network)**: Bipolar activations {-1,+1}, Hebbian/Storkey/delta learning

### Core Components

**Network class** (`include/network.hpp`, `src/network.cc`):
- Recurrent neural network with sigmoid transfer function
- Weight matrix (excitatory) and inhibition matrix (diagonal self-inhibition used in sleep)
- AVX2 SIMD-optimized operations for dot products and gradient descent
- Key iteration methods:
  - `iterate()` - Basic dynamics
  - `noisy_iterate()` - With Gaussian noise
  - `depressed_iterate()` - With diagonal inhibition (sleep mode)
  - `noisy_depressed_iterate()` - Combined noise + inhibition
- Learning methods use gradient descent with momentum and per-neuron bias

**Utils** (`include/utils.hpp`, `src/utils.cc`):
- `SimulationConfig` struct for configurable simulation parameters
- Pattern generation with balanced sparsity (`generatePatterns`)
- Winner-take-all functions (`assignBoolToTopNValues`)
- Parallel simulation launcher with thread pooling
- Data collection for aggregating results across parameter sweeps

### Simulation Pipeline

1. **Write phase** (`simulations/write_leak.cc`):
   - Creates fully-connected network
   - Generates sparse binary patterns
   - Trains attractors via `derivative_gradient_descent_with_bias_and_momentum_avx()`
   - Saves weights, connectivity, and patterns to `data/trained_networks/`

2. **Sleep phase** (`simulations/sleep_leak.cc`):
   - Loads pre-trained network from write phase
   - Runs autonomous retrieval cycles from neutral state
   - Uses `pot_inhib_symmetric()` to strengthen diagonal inhibition after each retrieval
   - Tracks pattern retrieval and spurious attractors
   - Saves results to `data/all_data_splited/sleep_simulations/`

### Data Flow

```
simulations/write_leak.cc
    → data/trained_networks/{sim_name}/sim_nb_X/
        ├── weights.data
        ├── connectivity.data
        ├── patterns.data
        └── parameters.data

simulations/sleep_leak.cc (reads from trained_networks)
    → data/all_data_splited/sleep_simulations/{sim_name}/sim_nb_X/
        ├── results.data (retrieval metrics per iteration)
        ├── patterns.data
        └── parameters.data
```

### Parameter Sweeps

Both simulations use `generateCombinations()` to create Cartesian products of parameter ranges, then run each combination in parallel threads (default: 20 threads max). Results are aggregated into `all_simulation_data.csv` via `collectSimulationData()` or `collectSimulationDataSeries()`.

### Key Parameters

- `leak`: Network leak rate (membrane time constant inverse)
- `drive_target`: Target activation strength for stored patterns
- `sparsity`: Fraction of active units per pattern (0 to 1)
- `rho`: Pattern correlation (1=identical, 0=maximally different)
- `beta`: Inhibitory plasticity rate during sleep
- `delta`: Integration timestep
- `noise_dynamics`: Enable stochastic noise in network dynamics
- `stddev_dynamics`: Noise standard deviation for stochastic dynamics

### DHN Components

**DiscreteHopfield class** (`include/discrete_hopfield.hpp`, `src/discrete_hopfield.cc`):
- Bipolar {-1,+1} activations, asymmetric weights, zero diagonal
- AVX2-optimized dot products (`avxDotProduct`, `avxDotProductNoDiag`)
- Learning rules:
  - `trainHebbian()`: W_ij += (1/N) * xi_i * xi_j (capacity ~13.8% N)
  - `trainStorkey()`: W_ij += (1/N) * [xi_i*xi_j - xi_i*h_j - h_i*xi_j] (capacity ~42% N)
- Dynamics: `runSynchronous()`, `runAsynchronous()`, `runSynchronousUntilConvergence()`
- Query: `createPartialCue()` + `matchesPattern()` (matches pattern OR its inverse)

**DHN Simulations**:
- `dhn_train.cc`: Train networks with Hebbian (learning_rule=0) or Storkey (learning_rule=1)
- `dhn_query.cc`: Query trained networks with partial cues (informed_fraction parameter)
- `mccallum.cc`: McCallum's 2007 pseudorehearsal algorithm (delta learning + probing)
- `ar_incremental.cc`: AR/CI incremental learning with sleep consolidation

## Python/C++ Workflow

The recommended workflow uses Python for experiment design and visualization, with C++ handling computation.

### Quick Start

```python
# In scripts/workflow.py or your own notebook
import sys; sys.path.insert(0, 'scripts')
from utils import *

# 1. Generate patterns
patterns = generate_patterns(k=10, n=100, sparsity=0.5, rho=0.5)

# 2. Setup and run training
config = setup_write_experiment(
    name="my_experiment",
    patterns=patterns,
    params={"leak": 1.0, "drive_target": 6.0, "learning_rate": 0.0001},
    varying_params={"leak": [0.5, 1.0, 2.0]}  # Optional sweep
)
run_cpp("write", config)

# 3. Setup and run sleep
sleep_config = setup_sleep_experiment(
    name="my_experiment_sleep",
    trained_networks_dir=DATA_DIR / "trained_networks" / "my_experiment",
    params={"beta": 0.1, "delta": 0.01, "max_queries": 200}
)
run_cpp("sleep", sleep_config)

# 4. Load results
results = load_results(DATA_DIR / "sleep_results" / "my_experiment_sleep")
```

### File Organization

```
scripts/
├── utils.py      # I/O functions, experiment setup, C++ launcher
└── workflow.py   # Example VSCode notebook-style script (#%% cells)

data/
├── configs/           # JSON configs generated by Python
├── trained_networks/  # Output from write simulations
└── sleep_results/     # Output from sleep simulations
```

### Generic Simulations

- `bin/write` - Generic training, reads JSON config
- `bin/sleep` - Generic sleep, reads JSON config
- `bin/write_leak`, `bin/sleep_leak` - Original hardcoded versions

## Heterogeneous Sparsity (C++ Native Pattern Generation)

### Overview

The framework supports generating patterns with **heterogeneous sparsities** (different patterns have different sparsity levels) natively in C++, enabling better parallelization for large-scale experiments.

### Usage in Python

```python
# Enable C++ native heterogeneous generation
write_config = setup_write_experiment(
    name="my_experiment",
    patterns=None,  # Don't provide patterns - C++ will generate them
    pattern_metadata=None,  # C++ will generate metadata too
    params={
        "use_heterogeneous_sparsity": 1,  # Enable heterogeneous mode
        "mean_sparsity": 0.5,              # Center of distribution (P(0) convention)
        "sparsity_width": 0.4,             # Full width (sparsities in [0.3, 0.7])
        "rho": 0.3,                        # Pattern correlation
        # ... other training params
    },
    varying_params={
        "network_size": [200, 250, 300],
        "num_patterns": [5, 8, 11],
        # ... other swept params
    },
    native_pattern_generation=True  # Enable C++ native generation
)
```

### Key Parameters

- `use_heterogeneous_sparsity`: Set to 1 to enable heterogeneous mode (0 = uniform sparsity)
- `mean_sparsity`: Center of sparsity distribution using P(0) convention (fraction inactive)
- `sparsity_width`: Full width of uniform distribution (e.g., 0.4 → range [0.3, 0.7])
- Pattern sparsities are uniformly distributed: `[mean - width/2, mean + width/2]`

### Pattern Metadata

C++ writes `pattern_metadata.json` containing per-pattern sparsity values:

```json
{
  "patterns": [
    {"index": 0, "sparsity": 0.32, "nb_active": 170},
    {"index": 1, "sparsity": 0.58, "nb_active": 105},
    ...
  ]
}
```

Load in Python using:
```python
from utils import read_pattern_metadata
metadata = read_pattern_metadata(sim_dir / "pattern_metadata.json")
```

### Script Organization

```
scripts/
├── SR_load/              # Large parameter sweep scripts (legacy location)
├── explo/                # Exploratory combined scripts (simulation + viz)
├── recovery_cinematic/   # Simulation-only scripts (data generation)
└── viz/                  # Visualization-only scripts (plotting)
```

**Recommended pattern**: Separate simulation and visualization:
- Put data generation in `recovery_cinematic/` (runs C++ simulations)
- Put plotting in `viz/` (loads results, creates figures)

### Example Scripts

- `scripts/explo/heterogeneous_sparsity_native.py` - Single config demo with visualization
- `scripts/recovery_cinematic/heterogeneous_nb_query_sim.py` - Multi-config sweep (200 reps)
- `scripts/viz/heterogeneous_nb_query_viz.py` - Plots query number vs sparsity
- `scripts/SR_heterogeneous_sparsity_sim.py` - Large sweep varying sparsity_width

## McCallum Comparison Framework

Compares four memory capacity methods in `scripts/comparison_mccallum/`:

| Method | Network | Learning | Consolidation |
|--------|---------|----------|---------------|
| McCallum | DHN | Delta rule + noise | Pseudorehearsal (spurious→pseudoitem) |
| AR (CI) | CHN | Batch GDA | Sleep (spurious→FAILURE) |
| Hebbian | DHN | One-shot | None |
| Storkey | DHN | One-shot | None |

### Running the Comparison

```bash
# 1. Build all simulations
make

# 2. Run each method (VSCode notebook cells)
python scripts/comparison_mccallum/mccallum_sim.py  # McCallum pseudorehearsal
python scripts/comparison_mccallum/ar_sim.py        # AR incremental
python scripts/comparison_mccallum/dhn_sim.py       # Hebbian + Storkey

# 3. Generate publication figure
python scripts/comparison_mccallum/viz_mccallum.py
```

### Experimental Grid

- Network sizes: [50, 75, 100, 125, 150, 175, 200, 225, 250]
- Pattern correlation ρ: [0.0, 0.2, 0.4, 0.5, 0.6]
- Seeds: 10 per (N, ρ)
- Success threshold θ: 0.9
- Max patterns: 50

### M* Computation

M* = max M such that ≥90% of simulations achieved M*_s ≥ M

```python
def compute_M_star(M_star_list, theta=0.9):
    for M in range(max(M_star_list), -1, -1):
        if sum(1 for m in M_star_list if m >= M) / len(M_star_list) >= theta:
            return M
    return 0
```

### Key Parameters

**McCallum DHN**: eta=0.1, E_max=500, nu_h=0.05 (5% noise), sigma_input=0.5, P_probes=2000, P_items=256

**AR (CHN)**: leak=1.0, drive_target=6.0, learning_rate=0.0001, momentum=0.9, beta=0.1

### Data Location

```
data/mccallum_results/
├── mccallum/     # McCallum results + M_star_summary.csv
├── ar/           # AR results + M_star_summary.csv
├── hebbian/      # DHN Hebbian results
└── storkey/      # DHN Storkey results
```

### Sanity Check

N=100, ρ=0.0: McCallum M* ≈ 10-15, Hebbian M* ≈ 14, Storkey M* ≈ 12 (with 50% cues)
