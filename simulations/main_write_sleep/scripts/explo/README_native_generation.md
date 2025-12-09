# C++ Native Heterogeneous Pattern Generation

## Overview

The `heterogeneous_sparsity_native.py` script demonstrates C++ native pattern generation with heterogeneous sparsity. This approach is more efficient than Python-based generation because:

1. **Parallelization**: Each simulation thread generates its own unique pattern set
2. **No file I/O overhead**: Patterns are generated directly in memory
3. **Parameter sweeps**: Can easily sweep across different heterogeneity parameters

## Quick Start

### 1. Direct C++ Usage (JSON config)

```bash
./bin/write data/configs/demo_native_heterogeneous.json
```

**Config format:**
```json
{
  "type": "write",
  "output_dir": "/path/to/output",
  "native_pattern_generation": true,
  "base_params": {
    "network_size": 250,
    "num_patterns": 10,
    "use_heterogeneous_sparsity": 1,
    "mean_sparsity": 0.5,
    "sparsity_width": 0.4,
    "rho": 0.3,
    "leak": 1.0,
    "drive_target": 6.0,
    "learning_rate": 0.0001,
    "max_iter": 100000
  },
  "varying_params": {
    "sparsity_width": [0.2, 0.4, 0.6]
  }
}
```

This will generate 3 simulations, each with **unique pattern sets** with different heterogeneity levels.

### 2. Python Wrapper Script

```python
from utils import setup_write_experiment, run_cpp

config = setup_write_experiment(
    name="my_experiment",
    patterns=None,  # No patterns - C++ generates them
    params={
        # Training params
        "leak": 1.0,
        "drive_target": 6.0,
        "learning_rate": 0.0001,
        "max_iter": 100000,
        # Pattern generation params
        "network_size": 250,
        "num_patterns": 10,
        "use_heterogeneous_sparsity": 1,
        "mean_sparsity": 0.5,
        "sparsity_width": 0.4,
        "rho": 0.3,
    },
    native_pattern_generation=True
)

run_cpp("write", config)
```

### 3. Full Workflow Script

See `heterogeneous_sparsity_native.py` for a complete workflow that:
- Builds C++ executables
- Runs training with native generation
- Runs sleep simulation
- Analyzes and visualizes results

**Note**: Requires pandas and matplotlib. Install with:
```bash
pip install pandas matplotlib
```

## Generated Output

### Metadata File

C++ automatically generates `pattern_metadata.json` in each simulation directory:

```json
{
  "version": 1,
  "num_patterns": 10,
  "network_size": 250,
  "generation_method": "heterogeneous",
  "global_params": {
    "mean_sparsity": 0.5,
    "sparsity_width": 0.4,
    "rho": 0.3
  },
  "patterns": [
    {"index": 0, "sparsity": 0.608, "nb_active": 98},
    {"index": 1, "sparsity": 0.388, "nb_active": 153},
    ...
  ]
}
```

### Pattern Files

Standard pattern data files are also generated:
- `patterns.data`: Binary patterns (space-separated 0/1)
- `weights.data`: Trained weight matrix
- `connectivity.data`: Network connectivity
- `parameters.data`: All simulation parameters

## Sleep Simulation

Sleep simulation automatically reads the metadata and tracks which pattern was recovered:

```json
{
  "type": "sleep",
  "input_dir": "/path/to/trained_networks",
  "output_dir": "/path/to/sleep_results",
  "base_params": {
    "beta": 0.025,
    "delta": 0.01,
    "max_queries": 200,
    "noise_dynamics": 1,
    "stddev_dynamics": 0.01
  }
}
```

Results include `recovered_pattern_idx` column showing which pattern (0 to K-1) was recovered on each query, or -1 for spurious patterns.

## Parameter Sweeps

You can sweep across heterogeneity parameters:

```json
{
  "varying_params": {
    "sparsity_width": [0.1, 0.2, 0.3, 0.4, 0.5]
  }
}
```

This generates 5 simulations, each with a **different pattern set** at different heterogeneity levels. All simulations run in parallel.

## Comparison: Python vs C++ Generation

| Feature | Python Generation | C++ Native Generation |
|---------|------------------|----------------------|
| **Pattern I/O** | Write to file, read from file | Generated in memory |
| **Parallelization** | Shared patterns across sims | Unique patterns per sim |
| **Parameter sweeps** | Same patterns for all | Different patterns for each |
| **Performance** | File I/O overhead | No overhead |
| **Use case** | Single pattern set, analysis | Parameter sweeps, bulk sims |

## Example: Sparsity Width Sweep

```json
{
  "native_pattern_generation": true,
  "base_params": {
    "network_size": 250,
    "num_patterns": 10,
    "use_heterogeneous_sparsity": 1,
    "mean_sparsity": 0.5,
    "rho": 0.3,
    ...
  },
  "varying_params": {
    "sparsity_width": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
  }
}
```

**Result**: 7 parallel simulations testing heterogeneity levels from uniform (0.0) to very heterogeneous (0.6). Each simulation gets its own unique pattern set.

## Algorithm

C++ uses the **parent/redraw** algorithm:

1. Generate parent pattern with `P(0) = mean_sparsity`
2. Compute `k = (1-rho) * N` positions to redraw
3. For each pattern i:
   - Sample `sparsity_i ~ Uniform(mean ± width/2)`
   - Start from parent
   - Redraw k random positions with `P(0) = sparsity_i`
   - Record actual sparsity in metadata

This is consistent with the existing C++ new-mode algorithm and ensures proper pattern correlation controlled by `rho`.
