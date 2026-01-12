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

def generate_patterns_old(k: int, n: int, sparsity: float = 0.5, rho: float = 0.5) -> np.ndarray:
    """
    OLD pattern generator (balanced flips).

    Conventions:
        - sparsity = fraction of ACTIVE units (P(x_i = 1))
        - Typically used with sparsity = 0.5.

    Algorithm:
        1. Build a base pattern with nb_winners = sparsity * n ones at the start.
        2. Set num_flips = floor((1 - rho) * nb_winners).
        3. For each pattern:
           - start from base
           - for each flip:
               * pick a random 1 -> set to 0
               * pick a random 0 -> set to 1
           - this keeps the number of ones exactly constant.
    """
    nb_winners = max(1, int(sparsity * n))
    base = np.zeros(n, dtype=bool)
    base[:nb_winners] = True

    num_flips = int((1.0 - rho) * nb_winners)
    patterns, seen = [], set()

    while len(patterns) < k:
        pattern = base.copy()
        for _ in range(num_flips):
            ones = np.where(pattern)[0]
            zeros = np.where(~pattern)[0]
            if ones.size > 0 and zeros.size > 0:
                pattern[np.random.choice(ones)] = False
                pattern[np.random.choice(zeros)] = True
        key = tuple(pattern.tolist())
        if key not in seen:
            seen.add(key)
            patterns.append(pattern)

    return np.array(patterns, dtype=bool)


def generate_patterns_new(k: int, n: int, sparsity: float = 0.5, rho: float = 0.5) -> np.ndarray:
    """
    NEW pattern generator (parent + redraw).

    Conventions:
        - sparsity s = P(x_i = 0) (fraction of inactive units)
        - density = 1 - s = P(x_i = 1)

    Algorithm:
        1. Generate a parent pattern x^parent with P(0)=s, P(1)=1-s.
        2. Set k_flips = floor((1 - rho) * n).
        3. For each pattern:
           - start from parent
           - choose k_flips distinct indices
           - at each chosen index, redraw bit: 0 with prob s, 1 with prob 1-s.
    """
    s = float(np.clip(sparsity, 0.0, 1.0))
    r = float(np.clip(rho, 0.0, 1.0))

    parent = (np.random.rand(n) > s)  # True ≡ 1

    k_flips = int((1.0 - r) * n)
    k_flips = max(0, min(k_flips, n))

    patterns = []
    seen = set()

    while len(patterns) < k:
        pattern = parent.copy()

        if k_flips > 0:
            idx = np.random.choice(n, size=k_flips, replace=False)
            pattern[idx] = (np.random.rand(k_flips) > s)

        key = tuple(pattern.tolist())
        if key not in seen:
            seen.add(key)
            patterns.append(pattern)

    return np.array(patterns, dtype=bool)


def generate_patterns(k: int, n: int, sparsity: float = 0.5, rho: float = 0.5) -> np.ndarray:
    """
    Default pattern generator used by workflow scripts.

    Currently this is the OLD balanced-flip generator, mainly used with
    sparsity = 0.5 (fraction of active units).
    """
    return generate_patterns_old(k, n, sparsity=sparsity, rho=rho)


def generate_patterns_heterogeneous(
    k: int,
    n: int,
    mean_sparsity: float = 0.5,
    sparsity_width: float = 0.2,
    rho: float = 0.5
) -> tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Generate patterns with heterogeneous sparsities using new generator algorithm.

    Each pattern i:
    1. Sample sparsity_i ~ Uniform(mean - width/2, mean + width/2)
    2. Generate pattern with P(0)=sparsity_i, P(1)=1-sparsity_i
    3. For correlation: use shared "core" bits based on rho

    Sparsity = P(0) = fraction of INACTIVE units (new convention).
    Example: mean_sparsity=0.8 with width=0.2 → sparsities in [0.7, 0.9]
             meaning 10-30% of units are active.

    Args:
        k: Number of patterns to generate
        n: Network size (neurons)
        mean_sparsity: Center of sparsity distribution
        sparsity_width: Full width of uniform distribution around mean
        rho: Pattern correlation (controls how many bits are shared)

    Returns:
        patterns: np.ndarray of shape (k, n) with dtype=bool
        per_pattern_metadata: List of dicts with per-pattern info
            Each dict has: {"index": int, "sparsity": float, "nb_active": int}
    """
    # Clamp parameters
    mean_s = float(np.clip(mean_sparsity, 0.0, 1.0))
    half_width = sparsity_width / 2.0
    r = float(np.clip(rho, 0.0, 1.0))

    # Compute number of "core" positions that stay constant (based on rho)
    n_core = int(r * n)
    n_variable = n - n_core

    # Generate core pattern at mean sparsity
    core_pattern = (np.random.rand(n_core) > mean_s) if n_core > 0 else np.array([], dtype=bool)

    patterns = []
    per_pattern_metadata = []
    seen = set()

    while len(patterns) < k:
        # Sample sparsity for this pattern
        s_i = mean_s + np.random.uniform(-half_width, half_width)
        s_i = float(np.clip(s_i, 0.01, 0.99))  # Avoid degenerate patterns

        # Generate variable portion with this pattern's sparsity
        variable_part = (np.random.rand(n_variable) > s_i) if n_variable > 0 else np.array([], dtype=bool)

        # Combine core and variable parts
        pattern = np.concatenate([core_pattern, variable_part])

        # Shuffle to avoid positional artifacts
        np.random.shuffle(pattern)

        key = tuple(pattern.tolist())
        if key not in seen:
            seen.add(key)
            patterns.append(pattern)

            # Compute actual sparsity (fraction of zeros)
            nb_active = int(pattern.sum())
            actual_sparsity = 1.0 - (nb_active / n)

            per_pattern_metadata.append({
                "index": len(patterns) - 1,
                "sparsity": actual_sparsity,
                "nb_active": nb_active
            })

    return np.array(patterns, dtype=bool), per_pattern_metadata


# =============================================================================
# Pattern Metadata I/O
# =============================================================================

def write_pattern_metadata(metadata: Dict[str, Any], filepath: Union[str, Path]) -> None:
    """Write pattern metadata to JSON file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(metadata, f, indent=2)


def read_pattern_metadata(filepath: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """
    Read pattern metadata from JSON file.

    Returns None if file doesn't exist (for backward compatibility).
    """
    filepath = Path(filepath)
    if not filepath.exists():
        return None
    with open(filepath, 'r') as f:
        return json.load(f)


def create_pattern_metadata(
    patterns: np.ndarray,
    generation_params: Optional[Dict[str, Any]] = None,
    per_pattern_data: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Create metadata dict from patterns array.

    If per_pattern_data is None, computes sparsity from patterns.
    This allows backward-compatible metadata generation for existing patterns.

    Args:
        patterns: Pattern array (num_patterns x network_size)
        generation_params: Optional dict with generation parameters
            (e.g., {"method": "heterogeneous", "mean_sparsity": 0.5, ...})
        per_pattern_data: Optional list of per-pattern dicts
            (if None, computed from patterns)

    Returns:
        Metadata dict ready for JSON serialization
    """
    num_patterns, network_size = patterns.shape

    # Generate per-pattern data if not provided
    if per_pattern_data is None:
        per_pattern_data = []
        for i in range(num_patterns):
            nb_active = int(patterns[i].sum())
            sparsity = 1.0 - (nb_active / network_size)  # P(0) convention
            per_pattern_data.append({
                "index": i,
                "sparsity": sparsity,
                "nb_active": nb_active
            })

    metadata = {
        "version": 1,
        "num_patterns": num_patterns,
        "network_size": network_size,
        "generation_method": generation_params.get("method", "unknown") if generation_params else "unknown",
        "global_params": generation_params or {},
        "patterns": per_pattern_data
    }

    return metadata


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
    pattern_metadata: Optional[Dict[str, Any]] = None,
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
        pattern_metadata: Optional metadata dict for patterns (from create_pattern_metadata).
                          If provided, will be saved alongside patterns for later use.
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
        all_params = set(params.keys()) | set(varying_params.keys())

        # Basic required params
        required = ["network_size", "num_patterns"]

        # Check if heterogeneous mode
        use_heterogeneous = params.get("use_heterogeneous_sparsity", 0) > 0.5

        if use_heterogeneous:
            # Heterogeneous mode: mean_sparsity and sparsity_width have defaults
            # rho also has a default, so no additional required params
            pass
        else:
            # Fixed sparsity mode: sparsity is required
            required.append("sparsity")

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

        # Save pattern metadata if provided
        if pattern_metadata is not None:
            metadata_path = config_dir / "pattern_metadata.json"
            write_pattern_metadata(pattern_metadata, metadata_path)
            config["metadata_file"] = str(metadata_path)

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


def setup_query_experiment(
    name: str,
    trained_networks_dir: Union[str, Path],
    params: Dict[str, Any],
    varying_params: Optional[Dict[str, List]] = None,
    output_dir: Optional[Path] = None,
    run_name: Optional[str] = None
) -> Path:
    """
    Setup a query experiment to test partial cue recovery on pre-trained networks.

    Args:
        name: Experiment name
        trained_networks_dir: Path to trained networks (from write experiment)
        params: Query simulation parameters (e.g., informed_fraction, delta)
        varying_params: Parameters to sweep (e.g., {"informed_fraction": [0.2, 0.1, 0.05, 0.02]})
                        informed_fraction = portion of units that retain pattern values
                        (the rest are set to neutral 0.5)
        output_dir: Where to save results (default: data/query_results/name)
        run_name: Optional subfolder to group runs under the same experiment

    Returns:
        Path to config file
    """
    if output_dir is None:
        output_dir = DATA_DIR / "query_results" / name
    if run_name:
        output_dir = output_dir / run_name

    config_dir = DATA_DIR / "configs" / name
    if run_name:
        config_dir = config_dir / run_name
    config_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "type": "query",
        "input_dir": str(trained_networks_dir),
        "output_dir": str(output_dir),
        "base_params": params,
        "varying_params": varying_params or {}
    }

    config_path = config_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    return config_path


def setup_dhn_train_experiment(
    name: str,
    params: Dict[str, Any],
    varying_params: Optional[Dict[str, List]] = None,
    output_dir: Optional[Path] = None,
    run_name: Optional[str] = None
) -> Path:
    """
    Setup a Discrete Hopfield Network training experiment.

    Trains DHN networks using Hebbian or Storkey learning rules.
    Patterns are generated natively in C++ based on parameters.

    Args:
        name: Experiment name
        params: Base simulation parameters including:
            - network_size: Number of neurons
            - num_patterns: Number of patterns to store
            - sparsity: Pattern sparsity (fraction of active units, default 0.5)
            - rho: Pattern correlation (0=uncorrelated, 1=identical, default 0.5)
            - learning_rule: 0 for Hebbian, 1 for Storkey (default 0)
            - seed: Random seed for reproducibility (optional)
        varying_params: Parameters to sweep {param_name: [values]}
            Common sweeps:
            - network_size: [100, 200, ..., 1000]
            - num_patterns: [10, 15, ..., 100]
            - rho: [0.1, 0.25, 0.5, 0.75, 1.0]
            - learning_rule: [0, 1] to compare Hebbian vs Storkey
            - seed: list(range(N)) for N repetitions
        output_dir: Where to save trained networks (default: data/trained_networks/name)
        run_name: Optional subfolder for grouping runs

    Returns:
        Path to config file

    Example:
        >>> config = setup_dhn_train_experiment(
        ...     name="comparison_dhn_hebbian",
        ...     params={"sparsity": 0.5, "learning_rule": 0},
        ...     varying_params={
        ...         "network_size": [100, 200, 300],
        ...         "num_patterns": [10, 20, 30],
        ...         "rho": [0.1, 0.5, 1.0],
        ...         "seed": list(range(2))  # 2 repetitions
        ...     }
        ... )
        >>> run_cpp("dhn_train", config)
    """
    if params is None:
        params = {}
    if varying_params is None:
        varying_params = {}

    # Validate required parameters
    all_params = set(params.keys()) | set(varying_params.keys())
    required = ["network_size", "num_patterns"]
    missing = [p for p in required if p not in all_params]
    if missing:
        raise ValueError(f"DHN training requires: {missing}")

    # Directory setup
    if output_dir is None:
        output_dir = DATA_DIR / "trained_networks" / name
    if run_name:
        output_dir = output_dir / run_name

    config_dir = DATA_DIR / "configs" / name
    if run_name:
        config_dir = config_dir / run_name
    config_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "type": "dhn_train",
        "native_pattern_generation": True,
        "output_dir": str(output_dir),
        "base_params": params,
        "varying_params": varying_params
    }

    config_path = config_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    return config_path


def setup_dhn_query_experiment(
    name: str,
    trained_networks_dir: Union[str, Path],
    params: Dict[str, Any],
    varying_params: Optional[Dict[str, List]] = None,
    output_dir: Optional[Path] = None,
    run_name: Optional[str] = None
) -> Path:
    """
    Setup a DHN partial cue query experiment on pre-trained DHN networks.

    Tests trained DHN networks with partial cues to measure retrieval capacity.
    For each stored pattern:
    1. Keep informed_fraction of units with pattern values
    2. Set remaining units to random {-1, +1}
    3. Run asynchronous dynamics until convergence
    4. Check if network recovers the correct pattern (or its inverse)

    Args:
        name: Experiment name
        trained_networks_dir: Path to trained DHN networks (from dhn_train)
        params: Query simulation parameters including:
            - informed_fraction: Fraction of units keeping pattern values (default 0.5)
            - nb_dynamics_steps: Number of async update sweeps (default 10)
        varying_params: Parameters to sweep, e.g.:
            - informed_fraction: [0.9, 0.5, 0.2, 0.1]
        output_dir: Where to save results (default: data/query_results/name)
        run_name: Optional subfolder for grouping runs

    Returns:
        Path to config file

    Example:
        >>> config = setup_dhn_query_experiment(
        ...     name="comparison_dhn_hebbian_query",
        ...     trained_networks_dir=DATA_DIR / "trained_networks" / "comparison_dhn_hebbian",
        ...     params={"nb_dynamics_steps": 10},
        ...     varying_params={
        ...         "informed_fraction": [0.9, 0.5, 0.2, 0.1]
        ...     }
        ... )
        >>> run_cpp("dhn_query", config)
    """
    if output_dir is None:
        output_dir = DATA_DIR / "query_results" / name
    if run_name:
        output_dir = output_dir / run_name

    config_dir = DATA_DIR / "configs" / name
    if run_name:
        config_dir = config_dir / run_name
    config_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "type": "dhn_query",
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


def load_final_results(results_dir: Union[str, Path]) -> pd.DataFrame:
    """
    Load final_results.csv (one row per simulation, final iteration only).

    Falls back to computing from all_simulation_data.csv if final_results.csv
    doesn't exist yet (for backward compatibility with older results).

    Args:
        results_dir: Directory containing simulation results

    Returns:
        DataFrame with one row per simulation containing final state metrics
    """
    results_dir = Path(results_dir)

    # Check for consolidated database first
    db_path = results_dir / "experiment.db"
    if db_path.exists():
        data = load_consolidated_experiment(db_path)
        return data['final_results']

    # Check for final_results.csv (new format)
    final_csv = results_dir / "final_results.csv"
    if final_csv.exists():
        return pd.read_csv(final_csv)

    # Fallback: compute from all_simulation_data.csv (old format)
    all_csv = results_dir / "all_simulation_data.csv"
    if all_csv.exists():
        df = pd.read_csv(all_csv)
        if 'sim_ID' in df.columns and 'query_iter' in df.columns:
            idx_last = df.groupby('sim_ID')['query_iter'].idxmax()
            return df.loc[idx_last].copy().reset_index(drop=True)
        else:
            # For write simulations, return as-is (already one row per sim)
            return df

    raise FileNotFoundError(f"No results found in {results_dir}")


# =============================================================================
# Trajectory Analysis
# =============================================================================

def compute_correlations(
    traj_list: List[np.ndarray],
    patterns_arr: np.ndarray,
    symmetric_transfer: bool = False
) -> tuple[np.ndarray, List[int]]:
    """
    Compute correlations between network states and target patterns over time.

    For symmetric transfer networks, patterns are transformed from {0,1} to {-0.5, 0.5}
    to match the network's output range (sigmoid(x) - 0.5).

    Args:
        traj_list: List of trajectory arrays, each shape (timesteps, neurons)
        patterns_arr: Target patterns array, shape (num_patterns, neurons), values in {0, 1}
        symmetric_transfer: If True, transform patterns to {-0.5, 0.5} for symmetric networks

    Returns:
        correlations: Array of shape (total_timesteps, num_patterns) with correlation values
        lengths: List of trajectory lengths (timesteps per query)

    Example:
        >>> trajectories = load_trajectories(sim_dir)
        >>> patterns = read_patterns(sim_dir / "patterns.data")
        >>> corr, lens = compute_correlations(trajectories, patterns, symmetric_transfer=True)
    """
    # Transform patterns for symmetric transfer
    if symmetric_transfer:
        patterns_transformed = patterns_arr - 0.5  # {0,1} -> {-0.5, 0.5}
    else:
        patterns_transformed = patterns_arr

    all_corr = []
    lengths = []

    for traj in traj_list:
        corr_this_query = []
        for t in range(traj.shape[0]):
            state = traj[t].astype(float)
            pattern_corrs = []

            for p in range(patterns_transformed.shape[0]):
                a = state
                b = patterns_transformed[p]

                # Handle edge case: zero vectors
                if np.allclose(a, 0) or np.allclose(b, 0):
                    c = 0.0
                else:
                    # Pearson correlation: dot product / (norm_a * norm_b)
                    c = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

                pattern_corrs.append(c)

            corr_this_query.append(pattern_corrs)

        all_corr.extend(corr_this_query)
        lengths.append(len(corr_this_query))

    return np.array(all_corr), lengths


# =============================================================================
# Binary Blob Parsing (for SQLite storage)
# =============================================================================

import struct

def read_binary_matrix(blob: bytes) -> np.ndarray:
    """
    Parse binary matrix blob (double values).

    Format: [rows:uint32][cols:uint32][data:float64[rows*cols]]
    """
    rows, cols = struct.unpack('<II', blob[:8])
    data = np.frombuffer(blob[8:], dtype=np.float64)
    return data.reshape(rows, cols)


def read_bitpacked_matrix(blob: bytes) -> np.ndarray:
    """
    Parse bitpacked boolean matrix blob.

    Format: [rows:uint32][cols:uint32][packed_bits:uint8[...]]
    """
    rows, cols = struct.unpack('<II', blob[:8])
    total_bits = rows * cols
    packed = np.frombuffer(blob[8:], dtype=np.uint8)
    unpacked = np.unpackbits(packed)[:total_bits]
    return unpacked.reshape(rows, cols).astype(bool)


def _matrix_to_blob(matrix: np.ndarray) -> bytes:
    """Convert numpy matrix to binary blob (for SQLite storage)."""
    rows, cols = matrix.shape
    header = struct.pack('<II', rows, cols)
    return header + matrix.astype(np.float64).tobytes()


def _bool_matrix_to_blob(matrix: np.ndarray) -> bytes:
    """Convert boolean numpy matrix to bitpacked blob."""
    rows, cols = matrix.shape
    header = struct.pack('<II', rows, cols)
    packed = np.packbits(matrix.flatten().astype(np.uint8))
    return header + packed.tobytes()


# =============================================================================
# SQLite Consolidation
# =============================================================================

import sqlite3


def load_consolidated_experiment(db_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load experiment from consolidated SQLite database.

    Args:
        db_path: Path to experiment.db

    Returns:
        Dict with:
        - 'simulations': DataFrame with sim_id and params
        - 'results': DataFrame with all time series results
        - 'final_results': DataFrame with one row per simulation (final state only)
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = sqlite3.connect(db_path)

    # Load simulations
    sim_df = pd.read_sql("SELECT sim_id, params FROM simulations", conn)
    sim_df['params'] = sim_df['params'].apply(json.loads)

    # Load results
    results_df = pd.read_sql("SELECT * FROM results", conn)

    # Compute final results (last row per simulation)
    if len(results_df) > 0 and 'query_iter' in results_df.columns:
        idx_last = results_df.groupby('sim_id')['query_iter'].idxmax()
        final_df = results_df.loc[idx_last].copy().reset_index(drop=True)
    else:
        final_df = results_df.copy()

    conn.close()

    return {
        'simulations': sim_df,
        'results': results_df,
        'final_results': final_df
    }


def load_simulation_matrices(db_path: Union[str, Path], sim_id: int) -> Dict[str, np.ndarray]:
    """
    Load weight/connectivity/pattern matrices for a specific simulation from SQLite.

    Args:
        db_path: Path to experiment.db
        sim_id: Simulation ID to load

    Returns:
        Dict with 'weights', 'connectivity', 'patterns' numpy arrays
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.execute(
        "SELECT weights, connectivity, patterns FROM simulations WHERE sim_id = ?",
        (sim_id,)
    )
    row = cursor.fetchone()
    conn.close()

    if row is None:
        raise ValueError(f"Simulation {sim_id} not found")

    return {
        'weights': read_binary_matrix(row[0]) if row[0] else None,
        'connectivity': read_bitpacked_matrix(row[1]) if row[1] else None,
        'patterns': read_bitpacked_matrix(row[2]) if row[2] else None
    }


def consolidate_experiment(
    results_dir: Union[str, Path],
    output_db: Optional[Union[str, Path]] = None,
    delete_folders: bool = False,
    include_weights: bool = True
) -> Path:
    """
    Consolidate sim_nb_X folders into single SQLite archive.

    This provides easy archiving and moving of experiment data.

    Args:
        results_dir: Directory containing sim_nb_X folders
        output_db: Output database path (default: results_dir/experiment.db)
        delete_folders: If True, remove sim_nb_X folders after successful consolidation
        include_weights: If True, include weight matrices in archive (larger file)

    Returns:
        Path to created database
    """
    import shutil

    results_dir = Path(results_dir)
    if output_db is None:
        output_db = results_dir / "experiment.db"
    output_db = Path(output_db)

    # Try C++ consolidate binary first (faster)
    consolidate_bin = BIN_DIR / "consolidate"
    if consolidate_bin.exists():
        result = subprocess.run(
            [str(consolidate_bin), str(results_dir), str(output_db)],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            if delete_folders:
                _cleanup_sim_folders(results_dir)
            return output_db
        # Fall through to Python implementation if C++ fails

    # Pure Python consolidation
    _consolidate_python(results_dir, output_db, include_weights)

    if delete_folders:
        _cleanup_sim_folders(results_dir)

    return output_db


def _consolidate_python(results_dir: Path, output_db: Path, include_weights: bool = True):
    """Pure Python consolidation implementation."""
    conn = sqlite3.connect(output_db)

    # Create tables
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS simulations (
            sim_id INTEGER PRIMARY KEY,
            params TEXT,
            weights BLOB,
            connectivity BLOB,
            patterns BLOB
        );
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sim_id INTEGER,
            query_iter INTEGER,
            nb_fnd_pat INTEGER,
            nb_spurious INTEGER,
            nb_iter_biased INTEGER,
            nb_iter_free INTEGER,
            all_recovered_before_spurious INTEGER,
            FOREIGN KEY(sim_id) REFERENCES simulations(sim_id)
        );
        CREATE INDEX IF NOT EXISTS idx_results_sim ON results(sim_id);
    """)

    sim_dirs = [d for d in results_dir.iterdir()
                if d.is_dir() and d.name.startswith("sim_nb_")]

    for sim_dir in sim_dirs:
        sim_id = int(sim_dir.name.split("_")[-1])

        # Read parameters
        params = {}
        if (sim_dir / "parameters.data").exists():
            params = read_parameters(sim_dir / "parameters.data")

        # Read and convert matrices to blobs
        weights_blob = None
        conn_blob = None
        patterns_blob = None

        if include_weights and (sim_dir / "weights.data").exists():
            weights = read_matrix(sim_dir / "weights.data")
            weights_blob = _matrix_to_blob(weights)

        if (sim_dir / "connectivity.data").exists():
            connectivity = read_matrix(sim_dir / "connectivity.data").astype(bool)
            conn_blob = _bool_matrix_to_blob(connectivity)

        if (sim_dir / "patterns.data").exists():
            patterns = read_patterns(sim_dir / "patterns.data")
            patterns_blob = _bool_matrix_to_blob(patterns)

        # Insert simulation
        conn.execute(
            "INSERT OR REPLACE INTO simulations VALUES (?, ?, ?, ?, ?)",
            (sim_id, json.dumps(params), weights_blob, conn_blob, patterns_blob)
        )

        # Insert results
        if (sim_dir / "results.data").exists():
            results = pd.read_csv(sim_dir / "results.data")
            for _, row in results.iterrows():
                conn.execute(
                    """INSERT INTO results
                       (sim_id, query_iter, nb_fnd_pat, nb_spurious,
                        nb_iter_biased, nb_iter_free, all_recovered_before_spurious)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (sim_id,
                     int(row.get('query_iter', 0)),
                     int(row.get('nb_fnd_pat', 0)),
                     int(row.get('nb_spurious', 0)),
                     int(row.get('nb_iter_biased', 0)),
                     int(row.get('nb_iter_free', 0)),
                     int(row.get('all_recovered_before_spurious', 0)))
                )

    conn.commit()
    conn.close()
    print(f"Consolidated {len(sim_dirs)} simulations to {output_db}")


def _cleanup_sim_folders(results_dir: Path):
    """Remove sim_nb_X folders after consolidation."""
    import shutil

    sim_dirs = [d for d in results_dir.iterdir()
                if d.is_dir() and d.name.startswith("sim_nb_")]

    for sim_dir in sim_dirs:
        shutil.rmtree(sim_dir)

    print(f"Removed {len(sim_dirs)} simulation folders")


# =============================================================================
# Visualization Utilities
# =============================================================================

def plot_density_3d_bw(data_dict, xlabel='Value', ylabel='Parameter', zlabel='Probability density',
                       depth_spacing=1.0, alpha=0.6, n_points=200,
                       bandwidth=None, view_elev=20, view_azim=60,
                       figsize=(10, 8), xlim=None):
    """
    3D density plot with black/white styling for comparing distributions.

    Creates a 3D visualization where each parameter value gets its own
    density curve, stacked along the y-axis for visual comparison.

    Parameters:
    -----------
    data_dict : dict
        Mapping of labels (floats/ints) to data arrays.
        Example: {0.5: weights_array_05, 1.0: weights_array_10, ...}
    xlabel : str
        X-axis label (the measured quantity)
    ylabel : str
        Y-axis label (the parameter being varied)
    depth_spacing : float
        Spacing between density curves along y-axis
    alpha : float
        Fill transparency (0=transparent, 1=opaque)
    n_points : int
        Number of points for density curve smoothness
    bandwidth : float, optional
        KDE bandwidth. Options:
        - None: Use scipy's default (Scott's rule)
        - 'scott': Scott's rule
        - 'silverman': Silverman's rule
        - float: Manual bandwidth value
    view_elev, view_azim : float
        3D viewing angles (elevation and azimuth in degrees)
    figsize : tuple
        Figure size (width, height)
    xlim : tuple, optional
        X-axis limits as (min, max). If None, auto-computed from data.

    Returns:
    --------
    fig, ax : matplotlib figure and 3D axis
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from scipy.stats import gaussian_kde

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Sort by label for consistent ordering
    sorted_items = sorted(data_dict.items(), key=lambda x: x[0])
    labels = [item[0] for item in sorted_items]
    datasets = [item[1] for item in sorted_items]

    n_curves = len(sorted_items)

    # Find global x-range for consistent axis
    if xlim is not None:
        x_plot_min, x_plot_max = xlim
    else:
        all_data = np.concatenate(datasets)
        x_min, x_max = all_data.min(), all_data.max()
        x_range = x_max - x_min
        x_plot_min = x_min - 0.1 * x_range
        x_plot_max = x_max + 0.1 * x_range
    x_points = np.linspace(x_plot_min, x_plot_max, n_points)

    max_density = 0

    for i, (label, data) in enumerate(sorted_items):
        # Compute KDE
        if len(data) > 1:
            kde = gaussian_kde(data)

            # Apply bandwidth setting
            if bandwidth is not None:
                if isinstance(bandwidth, str):
                    if bandwidth.lower() == 'scott':
                        kde.set_bandwidth(bw_method='scott')
                    elif bandwidth.lower() == 'silverman':
                        kde.set_bandwidth(bw_method='silverman')
                elif isinstance(bandwidth, (int, float)):
                    kde.set_bandwidth(bw_method=bandwidth)

            density = kde(x_points)
        else:
            density = np.zeros_like(x_points)
            if len(data) == 1:
                closest_idx = np.argmin(np.abs(x_points - data[0]))
                density[closest_idx] = 1.0

        max_density = max(max_density, density.max())
        y_pos = i * depth_spacing

        # Create y-array for plotting
        y_curve = np.full_like(x_points, y_pos)

        # Plot density curve (black line, thick)
        ax.plot(x_points, y_curve, density, color='black', linewidth=2.5)

        # Create filled polygon vertices
        verts = []
        for j in range(len(x_points)):
            verts.append([x_points[j], y_pos, density[j]])
        # Close polygon along baseline
        for j in reversed(range(len(x_points))):
            verts.append([x_points[j], y_pos, 0])

        # Add filled polygon (uniform gray fill, black edge)
        poly = Poly3DCollection([verts], alpha=alpha,
                                facecolor='0.7',
                                edgecolor='black',
                                linewidth=0.5)
        ax.add_collection3d(poly)

    # Axis labels
    ax.set_xlabel(xlabel, fontsize=20, labelpad=15)
    ax.set_ylabel(ylabel, fontsize=20, labelpad=15)
    # Use text2D for zlabel (set_zlabel gets clipped)
    ax.set_zlabel('')
    ax.text2D(-0.12, 0.5, zlabel, transform=ax.transAxes, fontsize=20, rotation=90, va='center')

    # Y-tick labels (parameter values)
    y_ticks = [i * depth_spacing for i in range(n_curves)]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'{l}' for l in labels], fontsize=16)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='z', labelsize=16)

    # Viewing angle
    ax.view_init(elev=view_elev, azim=view_azim)

    # Invert x-axis so higher values are on the right
    ax.invert_xaxis()

    # Set aspect ratio
    ax.set_box_aspect([1, depth_spacing * 0.8, 0.6])

    # Wide margins to prevent zlabel clipping
    plt.subplots_adjust(left=0.15, right=0.85, bottom=0.1, top=0.9)
    return fig, ax
