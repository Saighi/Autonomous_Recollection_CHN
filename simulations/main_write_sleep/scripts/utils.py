"""
Utilities for Python/C++ workflow.
Handles I/O in formats compatible with C++ simulation code.
"""

import json
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, List, Dict, Any, Optional
from itertools import product

# Base paths (relative to scripts/ directory)
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
BIN_DIR = PROJECT_DIR / "bin"
DATA_DIR = PROJECT_DIR / "data"


# =============================================================================
# Pattern Generation
# =============================================================================

def _generate_base_pattern(n: int, nb_winners: int) -> np.ndarray:
    """Generate a base binary pattern with nb_winners ones at the start."""
    pattern = np.zeros(n, dtype=bool)
    pattern[:nb_winners] = True
    return pattern


def _generate_noisy_balanced_pattern(base_pattern: np.ndarray, num_flips: int) -> np.ndarray:
    """Generate a noisy version by flipping pairs (one 1->0, one 0->1)."""
    pattern = base_pattern.copy()
    n = len(pattern)

    for _ in range(num_flips):
        ones = np.where(pattern)[0]
        zeros = np.where(~pattern)[0]
        if len(ones) > 0 and len(zeros) > 0:
            pattern[np.random.choice(ones)] = False
            pattern[np.random.choice(zeros)] = True

    return pattern


def generate_patterns(k: int, n: int, sparsity: float = 0.5, rho: float = 0.5) -> np.ndarray:
    """
    Generate K unique sparse binary patterns.

    Args:
        k: Number of patterns
        n: Pattern size (network size)
        sparsity: Fraction of active units per pattern (0 to 1)
        rho: Pattern correlation (1=identical, 0=maximally different).
             Flips (1-rho)*nb_winners bits between patterns.

    Returns:
        Array of shape (k, n) with boolean patterns
    """
    nb_winners = max(1, int(sparsity * n))
    base = _generate_base_pattern(n, nb_winners)
    num_flips = int((1 - rho) * nb_winners)
    patterns, seen = [], set()

    while len(patterns) < k:
        new_pattern = _generate_noisy_balanced_pattern(base, num_flips)
        key = tuple(new_pattern)
        if key not in seen:
            seen.add(key)
            patterns.append(new_pattern)

    return np.array(patterns)


# =============================================================================
# File I/O (C++ compatible formats)
# =============================================================================

def write_patterns(patterns: np.ndarray, filepath: Union[str, Path]) -> None:
    """Write patterns in C++ compatible format (space-separated 0/1)."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        for pattern in patterns:
            f.write(' '.join(str(int(x)) for x in pattern) + '\n')


def read_patterns(filepath: Union[str, Path]) -> np.ndarray:
    """Read patterns from C++ format file."""
    patterns = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                patterns.append([x == '1' for x in line.strip().split()])
    return np.array(patterns, dtype=bool)


def write_matrix(matrix: np.ndarray, filepath: Union[str, Path]) -> None:
    """Write matrix in C++ compatible format (space-separated)."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        for row in matrix:
            f.write(' '.join(str(x) for x in row) + '\n')


def read_matrix(filepath: Union[str, Path]) -> np.ndarray:
    """Read matrix from C++ format file."""
    rows = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                rows.append([float(x) for x in line.strip().split()])
    return np.array(rows)


def read_parameters(filepath: Union[str, Path]) -> Dict[str, float]:
    """Read parameters from C++ key=value format."""
    params = {}
    with open(filepath, 'r') as f:
        for line in f:
            if '=' in line:
                key, value = line.strip().split('=', 1)
                params[key] = float(value)
    return params


# =============================================================================
# Generic Experiment Setup
# =============================================================================

def setup_write_experiment(
    name: str,
    patterns: Optional[np.ndarray] = None,
    params: Optional[Dict[str, Any]] = None,
    varying_params: Optional[Dict[str, List]] = None,
    output_dir: Optional[Path] = None,
    run_name: Optional[str] = None,
    native_pattern_generation: bool = False
) -> Path:
    """
    Setup a write/training experiment.

    Args:
        name: Experiment name
        patterns: Binary patterns to store (n_patterns x network_size).
                  Required if native_pattern_generation=False.
        params: Base simulation parameters
        varying_params: Parameters to sweep {param_name: [values]}
        output_dir: Where to save (default: data/trained_networks/name)
        run_name: Optional subfolder to group multiple runs under the same
            experiment name without clobbering previous outputs.
        native_pattern_generation: If True, C++ generates patterns internally.
                                   Requires network_size, num_patterns, sparsity, rho
                                   in params or varying_params.

    Returns:
        Path to config file
    """
    import warnings

    if params is None:
        params = {}
    if varying_params is None:
        varying_params = {}

    # Validation based on mode
    if native_pattern_generation:
        # Check required parameters exist
        required = ["network_size", "num_patterns", "sparsity", "rho"]
        all_params = set(params.keys()) | set(varying_params.keys())
        missing = [p for p in required if p not in all_params]
        if missing:
            raise ValueError(
                f"native_pattern_generation=True requires: {missing}. "
                f"Provide in params or varying_params."
            )
        if patterns is not None:
            warnings.warn("patterns argument ignored when native_pattern_generation=True")
    else:
        # File mode: patterns required
        if patterns is None:
            raise ValueError("patterns required when native_pattern_generation=False")

    # Directory setup
    if output_dir is None:
        output_dir = DATA_DIR / "trained_networks" / name
    if run_name:
        output_dir = output_dir / run_name

    config_dir = DATA_DIR / "configs" / name
    if run_name:
        config_dir = config_dir / run_name
    config_dir.mkdir(parents=True, exist_ok=True)

    # Build config based on mode
    if native_pattern_generation:
        # Native mode: no patterns_file needed
        full_params = dict(params)

        # Compute nb_winners if sparsity and network_size are fixed (not varying)
        if "sparsity" in full_params and "network_size" in full_params:
            sparsity = full_params["sparsity"]
            network_size = full_params["network_size"]
            full_params["nb_winners"] = max(1, int(sparsity * network_size))

        config = {
            "type": "write",
            "native_pattern_generation": True,
            "output_dir": str(output_dir),
            "base_params": full_params,
            "varying_params": varying_params
        }
    else:
        # File mode: existing behavior
        network_size = patterns.shape[1]
        nb_winners = int(patterns[0].sum())
        sparsity = nb_winners / network_size

        full_params = {
            "network_size": network_size,
            "sparsity": sparsity,
            "nb_winners": nb_winners,
            "num_patterns": len(patterns),
            **params
        }

        config = {
            "type": "write",
            "native_pattern_generation": False,
            "patterns_file": str(config_dir / "patterns.data"),
            "output_dir": str(output_dir),
            "base_params": full_params,
            "varying_params": varying_params
        }

        # Save patterns
        write_patterns(patterns, config_dir / "patterns.data")

    # Save config
    config_path = config_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    return config_path


def setup_sleep_experiment(
    name: str,
    trained_networks_dir: Union[str, Path],
    params: Dict[str, Any],
    varying_params: Optional[Dict[str, List]] = None,
    output_dir: Optional[Path] = None,
    run_name: Optional[str] = None
) -> Path:
    """
    Setup a sleep experiment on pre-trained networks.

    Args:
        name: Experiment name
        trained_networks_dir: Path to trained networks (from write experiment)
        params: Sleep simulation parameters
        varying_params: Parameters to sweep
        output_dir: Where to save results
        run_name: Optional subfolder to group runs under the same experiment
            name without overwriting results.

    Returns:
        Path to config file
    """
    if output_dir is None:
        output_dir = DATA_DIR / "sleep_results" / name
    if run_name:
        output_dir = output_dir / run_name

    config_dir = DATA_DIR / "configs" / name
    if run_name:
        config_dir = config_dir / run_name
    config_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "type": "sleep",
        "input_dir": str(trained_networks_dir),
        "output_dir": str(output_dir),
        "base_params": params,
        "varying_params": varying_params or {}
    }

    config_path = config_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    return config_path


# =============================================================================
# C++ Execution
# =============================================================================

def run_cpp(executable: str, config_path: Union[str, Path], verbose: bool = True) -> subprocess.CompletedProcess:
    """
    Run a C++ simulation with a config file.

    Args:
        executable: Name of executable ('write' or 'sleep')
        config_path: Path to JSON config
        verbose: Print output in real-time
    """
    exe_path = BIN_DIR / executable

    if not exe_path.exists():
        raise FileNotFoundError(f"Executable not found: {exe_path}. Run 'make' first.")

    cmd = [str(exe_path), str(config_path)]

    if verbose:
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=str(PROJECT_DIR))
    else:
        result = subprocess.run(cmd, cwd=str(PROJECT_DIR), capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error: {result.stderr}")

    return result


def build(verbose: bool = True) -> bool:
    """Build all C++ simulations."""
    result = subprocess.run(
        ["make", "-j4"],
        cwd=str(PROJECT_DIR),
        capture_output=not verbose,
        text=True
    )
    return result.returncode == 0


# =============================================================================
# Results Loading
# =============================================================================

def load_results(results_dir: Union[str, Path]) -> pd.DataFrame:
    """Load aggregated CSV results from a simulation run."""
    csv_path = Path(results_dir) / "all_simulation_data.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Results not found: {csv_path}")
    return pd.read_csv(csv_path)


def load_simulation(sim_dir: Union[str, Path]) -> Dict[str, Any]:
    """Load all data from a single simulation folder."""
    sim_dir = Path(sim_dir)
    data = {'parameters': read_parameters(sim_dir / "parameters.data")}

    if (sim_dir / "weights.data").exists():
        data['weights'] = read_matrix(sim_dir / "weights.data")
    if (sim_dir / "connectivity.data").exists():
        data['connectivity'] = read_matrix(sim_dir / "connectivity.data").astype(bool)
    if (sim_dir / "patterns.data").exists():
        data['patterns'] = read_patterns(sim_dir / "patterns.data")
    if (sim_dir / "results.data").exists():
        data['results'] = pd.read_csv(sim_dir / "results.data")

    return data


def list_simulations(results_dir: Union[str, Path]) -> List[Path]:
    """List all simulation folders in a results directory."""
    results_dir = Path(results_dir)
    return sorted([d for d in results_dir.iterdir()
                   if d.is_dir() and d.name.startswith("sim_nb_")])


def load_trajectories(sim_dir: Union[str, Path]) -> List[np.ndarray]:
    """Load trajectory files (results_0.data, results_1.data, ...) from a simulation."""
    sim_dir = Path(sim_dir)
    trajectories = []
    idx = 0
    while (sim_dir / f"results_{idx}.data").exists():
        trajectories.append(read_matrix(sim_dir / f"results_{idx}.data"))
        idx += 1
    return trajectories
