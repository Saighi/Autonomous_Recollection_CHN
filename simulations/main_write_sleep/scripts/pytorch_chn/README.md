# PyTorch GPU-Accelerated CHN Implementation

This module provides a PyTorch/CUDA implementation of Continuous Hopfield Network (CHN) simulations, designed to replace the slower C++ backend for large-scale parameter sweeps.

## Overview

**Goal**: Run CHN simulations on GPU for faster processing of larger networks (N=1000+, K=100+ patterns).

**Key Features**:
- GPU acceleration via PyTorch CUDA
- Batched parallelism (multiple networks processed simultaneously)
- Adam optimizer for ~100x faster training convergence
- C++ compatible output format for existing analysis scripts
- Progress bars with real-time statistics

## File Structure

```
scripts/
├── pytorch_chn/
│   ├── __init__.py       # Module exports, device detection
│   ├── network.py        # ContinuousHopfieldNetwork, BatchedCHN classes
│   ├── learning.py       # Adam/SGD training with batched gradients
│   ├── sleep.py          # Sleep phase with diagonal inhibition
│   ├── patterns.py       # Pattern generation utilities
│   └── README.md         # This file
│
└── run_chn_sim_pytorch.py  # Main simulation script
```

## Technical Details

### Transfer Function

Uses **standard sigmoid** `torch.sigmoid(u)` → [0, 1] to match C++ implementation:
- Patterns encoded as {0, 1} floats
- Neutral state = 0.5
- Target drives: +6.0 for active (→0.997), -6.0 for inactive (→0.003)

### Training (learning.py)

**Adam optimizer** (`train_patterns_adam()`):
- Converges in ~100-200 iterations instead of ~100,000 for vanilla GDA
- N=1000, K=50 trains in ~0.3 seconds on RTX 3050

**Batched gradient computation**:
```python
# All K patterns processed at once
target_rates = torch.sigmoid(target_drives)  # [K, N]
u_hat = (target_rates @ W) / leak            # [K, N]
errors = target_drives - u_hat               # [K, N]
delta_W = (errors.T @ target_rates) / K      # [N, N] - averaged gradient
```

### Sleep Phase (sleep.py)

- Autonomous retrieval from neutral state (v=0.5)
- Diagonal inhibition potentiation after each retrieval
- Pattern matching via overlap threshold
- Tracks: found patterns, spurious attractors, AR success metric

### Network Classes (network.py)

**ContinuousHopfieldNetwork**: Single network operations
- `step()` - Basic Euler integration
- `depressed_step()` - With diagonal inhibition (sleep)
- `pot_inhib_diag(beta)` - Potentiate self-inhibition
- `reset_to_neutral()` - Reset to v=0.5

**BatchedCHN**: Process B networks in parallel
- Same interface but with batched tensors [B, N, ...]
- Uses `torch.bmm` for batched matrix-vector products

## Usage

### Quick Start

```python
import torch
from pytorch_chn import (
    ContinuousHopfieldNetwork,
    train_patterns_adam,
    run_sleep_phase,
    generate_patterns
)

# Generate patterns
patterns = generate_patterns(k=10, n=200, sparsity=0.5, rho=0.5, device='cuda')

# Create and train network
network = ContinuousHopfieldNetwork(n_neurons=200, device='cuda')
W, converged, history = train_patterns_adam(
    network.W, patterns,
    target_drive=6.0,
    learning_rate=0.01,
    max_iter=500,
    tolerance=0.1
)
network.set_weights(W)

# Run sleep phase
results = run_sleep_phase(
    network, patterns,
    max_queries=200,
    beta=0.1,
    delta=0.01,
    noise_stddev=0.01
)

print(f"Found {len(results.found_patterns)}/{patterns.shape[0]} patterns")
print(f"Spurious attractors: {results.n_spurious}")
```

### Running Full Simulation

```bash
# From scripts/ directory
python run_chn_sim_pytorch.py
```

This runs parameter sweeps with progress bars showing:
- Training: convergence rate, avg iterations, GPU memory
- Sleep: patterns found, spurious rate, AR success rate

### Output Format

Results saved in C++ compatible format:
- `parameters.data`: key=value text file
- `patterns.data`: space-separated 0/1
- `results.data`: CSV with columns matching C++ output

## Performance Benchmarks

Tested on RTX 3050:

| Network Size | Patterns | Training Time | Notes |
|--------------|----------|---------------|-------|
| N=200, K=10  | 10       | 1.9s (109 iter) | Adam, tolerance=0.1 |
| N=1000, K=50 | 50       | 0.27s | Adam, tolerance=0.1 |

GPU batching provides additional speedup for parameter sweeps by processing multiple networks simultaneously.

## Dependencies

```
torch>=2.0.0  # With CUDA support
numpy
pandas
tqdm
```

Verify CUDA: `torch.cuda.is_available()` should return True.

## Environment Notes

- Developed with PyTorch 2.6.0+cu124 in anaconda3 base environment (Python 3.9)
- Results are plain CSV files, can be analyzed in any Python environment
- Keep simulation in CUDA-enabled env, visualization can use separate env

## Design Decisions

1. **Standard sigmoid [0,1]** vs symmetric [-0.5, 0.5]: Matches C++ implementation for result compatibility

2. **Adam optimizer**: Much faster than SGD+momentum, suitable for equilibrium-based learning

3. **Batched pattern gradients**: Process all K patterns simultaneously, average gradients - maximizes GPU utilization

4. **C++ compatible output**: Enables use of existing analysis scripts without modification

5. **Progress bars via tqdm**: PyTorch has no built-in progress system; tqdm is the standard lightweight solution

## Relationship to C++ Implementation

This PyTorch version is designed to produce equivalent results to the C++ backend:
- Same transfer function (standard sigmoid)
- Same pattern encoding ({0, 1})
- Same output format
- Same metrics (AR success, spurious count, etc.)

The main difference is speed: GPU parallelism enables faster processing of large parameter sweeps.
