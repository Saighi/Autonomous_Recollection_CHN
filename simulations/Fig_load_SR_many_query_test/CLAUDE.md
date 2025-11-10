# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a C++ simulation framework for studying **Spontaneous Recollection in Continuous Hopfield Networks (CHN)**. The project simulates neural network dynamics with learned patterns, focusing on how networks spontaneously retrieve memories during "sleep" phases with inhibitory plasticity.

## Build System

The project uses a custom Makefile with aggressive optimization flags:

```bash
# Build the simulation executable
make

# Build and run immediately
make run

# Clean and rebuild
rm -rf obj simulations && make
```

**Key compilation flags:**
- `-std=c++17` - Uses C++17 standard
- `-O3 -march=native` - Aggressive optimization with AVX vectorization
- `-pthread` - Multi-threading support

## Architecture

### Core Components

**Network Class** (`src/network.cc`, `include/network.hpp`):
- Implements a recurrent neural network with continuous dynamics
- Uses **AVX-optimized** dot products for performance (`avx_dot_product`)
- Key methods:
  - `noisy_iterate()` / `noisy_depressed_iterate()`: Network dynamics with Gaussian noise
  - `pot_inhib_symmetric()`: Diagonal inhibitory plasticity (only self-connections)
  - `reinforce_attractor()`: Hebbian-like learning rule
  - `transfer()`: Sigmoid activation function
- State vectors: `activity_list` (pre-activation), `rate_list` (post-activation), `derivative_activity_list`
- Matrix structures: `weight_matrix` (excitatory), `inhib_matrix` (inhibitory), `connectivity_matrix` (boolean connectivity)

**Simulation Logic** (`src/simulations.cc`):
- `run_sleep()`: Main simulation function for a single trial
  - Loads pre-trained networks from disk
  - Runs retrieval cycles until all patterns found or spurious states emerge
  - Each cycle: biased dynamics (with depression) → free dynamics (without depression)
  - Applies inhibitory plasticity after each retrieval
  - Saves results per simulation in individual directories
- `main()`: Orchestrates parallel simulations
  - Loads trained networks from `../../../data/all_data_splited/trained_networks_fast/`
  - Generates parameter combinations (e.g., varying beta, delta, noise)
  - Launches up to 20 concurrent threads
  - Aggregates all results into single CSV

**Utilities** (`src/utils.cc`, `include/utils.hpp`):
- `run_net_sim_choice()`: Iterates network until convergence (max derivative < epsilon)
- `assignBoolToTopNValues()`: Converts continuous rates to binary patterns (winner-take-all)
- `collectSimulationDataSeries()`: Aggregates all simulation results into `all_simulation_data.csv`
- Pattern generation and I/O utilities
- Parameter combination generation for sweeps

### Key Algorithmic Details

**Inhibitory Plasticity (Diagonal Only):**
The code uses **diagonal-only inhibition** in `pot_inhib_symmetric()` at src/network.cc:555:
```cpp
if(i==j){
    inhib_matrix[i][j] += pot_rate * (rate_list[j] * (rate_list[i] - 0.5));
}
```
This means only self-inhibition is plastic, not cross-neuron inhibition.

**Network Dynamics:**
- Two-phase retrieval per iteration:
  1. **Biased phase** (`depressed=true`): Inhibition suppresses previously retrieved patterns
  2. **Free phase** (`depressed=false`): Pure excitatory dynamics
- Convergence criterion: `max(abs(rate_new - rate_old)) < epsilon`

**Multi-threading:**
- Uses mutex/condition_variable for thread pool (max 20 threads)
- Each thread processes one parameter combination for one trained network
- Thread-safe due to no shared state modification

## Data Flow

1. **Input:** Pre-trained networks from `../../../data/all_data_splited/trained_networks_fast/`
   - Each subdirectory contains: `parameters.data`, `weights.data`, `connectivity.data`, `patterns.data`
2. **Processing:** For each trained network × parameter combination:
   - Run retrieval cycles (up to 2000 or until spurious state)
   - Track: patterns found, spurious states, convergence iterations
3. **Output:** Results saved to `../../../data/all_data_splited/sleep_simulations/<sim_name>/`
   - Per-simulation directories: `sim_nb_<N>/`
   - Aggregated CSV: `all_simulation_data.csv`

## Running Simulations

**Modify simulation parameters in `src/simulations.cc:174-197`:**
```cpp
string sim_name = "your_experiment_name";
string inputs_name = "your_trained_networks_folder";
vector<double> beta = {0.025, 0.05, 0.1, 0.5, 10};  // Inhibitory plasticity rates
```

**Parameter sweep variables:**
- `beta`: Inhibitory plasticity rate
- `delta`: Time step for integration
- `noise`: Enable/disable Gaussian noise (0 or 1)
- `stddev`: Standard deviation of noise
- `save`: Save full trajectories (0 or 1, expensive!)

**Execute:**
```bash
make && ./simulations
```

## Important Notes

- **Memory usage:** Setting `save=1` creates trajectory files for every retrieval, generating large amounts of data
- **Performance:** AVX optimizations require compatible CPU (use `-march=native`)
- **Thread safety:** Random number generator is per-thread (seeded in `run_sleep` at line 23)
- **Commented code:** Much legacy code is commented out (e.g., full inhibition matrix, old iterate methods). The active code uses diagonal inhibition only.
- **Pattern representation:** Binary patterns stored as `vector<bool>`, but network operates on continuous rates [0,1]
