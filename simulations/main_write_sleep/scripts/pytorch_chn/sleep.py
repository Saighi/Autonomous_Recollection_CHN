"""
Sleep phase implementation for Continuous Hopfield Networks.

Implements autonomous retrieval with diagonal inhibition potentiation.
"""

import torch
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass

from .network import ContinuousHopfieldNetwork, BatchedCHN


@dataclass
class SleepResult:
    """Results from a single sleep query."""
    matched_pattern: int  # -1 if spurious
    n_steps: int
    converged: bool


@dataclass
class SleepPhaseResults:
    """Results from full sleep phase."""
    found_patterns: set  # Set of pattern indices found
    n_spurious: int  # Number of spurious attractors encountered
    queries: List[SleepResult]  # Per-query results
    all_recovered_before_spurious: bool  # AR success metric


def detect_pattern_match(
    v: torch.Tensor,
    patterns: torch.Tensor,
    threshold: float = 0.8
) -> int:
    """Check if network state matches any pattern.

    Uses overlap (correlation) to determine match.

    Args:
        v: Current firing rates [N]
        patterns: Binary patterns [K, N] in {0, 1}
        threshold: Minimum overlap to consider a match

    Returns:
        Pattern index if matched, -1 if spurious
    """
    # Convert patterns to expected rates (0 -> ~0, 1 -> ~1)
    # At drive +/-6, rates are ~0.997 and ~0.003

    # Binarize v at 0.5 threshold
    v_binary = (v > 0.5).float()

    # Compute overlap with each pattern
    K, N = patterns.shape
    patterns_float = patterns.float()

    # Overlap = fraction of matching bits
    matches = (v_binary == patterns_float).float().mean(dim=1)  # [K]

    # Also check inverse patterns
    matches_inv = (v_binary == (1 - patterns_float)).float().mean(dim=1)  # [K]

    # Take max of normal and inverse
    best_overlap = torch.max(matches, matches_inv)
    best_idx = best_overlap.argmax().item()
    best_score = best_overlap[best_idx].item()

    if best_score >= threshold:
        return best_idx
    return -1


def detect_pattern_match_batched(
    v: torch.Tensor,
    patterns: torch.Tensor,
    threshold: float = 0.8
) -> torch.Tensor:
    """Check pattern matches for batched networks.

    Args:
        v: Firing rates [B, N]
        patterns: Patterns [B, K, N] or [K, N] (broadcast)
        threshold: Minimum overlap

    Returns:
        Pattern indices [B], -1 for spurious
    """
    B, N = v.shape

    # Binarize
    v_binary = (v > 0.5).float()  # [B, N]

    # Handle patterns shape
    if patterns.dim() == 2:
        # Same patterns for all networks
        K = patterns.shape[0]
        patterns = patterns.unsqueeze(0).expand(B, K, N)
    else:
        K = patterns.shape[1]

    patterns_float = patterns.float()  # [B, K, N]

    # Expand v for comparison
    v_expanded = v_binary.unsqueeze(1)  # [B, 1, N]

    # Compute matches
    matches = (v_expanded == patterns_float).float().mean(dim=2)  # [B, K]
    matches_inv = (v_expanded == (1 - patterns_float)).float().mean(dim=2)  # [B, K]

    best_overlap = torch.max(matches, matches_inv)  # [B, K]
    best_scores, best_indices = best_overlap.max(dim=1)  # [B]

    # Mark spurious as -1
    result = best_indices.clone()
    result[best_scores < threshold] = -1

    return result


def run_until_convergence(
    network: ContinuousHopfieldNetwork,
    max_steps: int = 1000,
    noise_stddev: float = 0.01,
    convergence_tol: float = 1e-4,
    check_every: int = 10
) -> int:
    """Run network until convergence or max steps.

    Args:
        network: The CHN network
        max_steps: Maximum integration steps
        noise_stddev: Noise standard deviation
        convergence_tol: Convergence tolerance
        check_every: Check convergence every N steps

    Returns:
        Number of steps taken
    """
    prev_v = network.v.clone()

    for step in range(max_steps):
        network.depressed_step(noise_stddev)

        if (step + 1) % check_every == 0:
            v_change = (network.v - prev_v).abs().max().item()
            if v_change < convergence_tol:
                return step + 1
            prev_v = network.v.clone()

    return max_steps


def run_sleep_query(
    network: ContinuousHopfieldNetwork,
    patterns: torch.Tensor,
    max_steps: int = 1000,
    noise_stddev: float = 0.01,
    convergence_tol: float = 1e-4,
    match_threshold: float = 0.8
) -> SleepResult:
    """Run single autonomous retrieval query.

    1. Start from neutral state
    2. Run depressed dynamics until convergence
    3. Check if converged state matches a pattern

    Args:
        network: CHN network with trained weights
        patterns: Binary patterns [K, N]
        max_steps: Max integration steps
        noise_stddev: Noise for dynamics
        convergence_tol: Convergence tolerance
        match_threshold: Pattern match threshold

    Returns:
        SleepResult with match info
    """
    # Reset to neutral
    network.reset_to_neutral()

    # Run until convergence
    n_steps = run_until_convergence(
        network, max_steps, noise_stddev, convergence_tol
    )

    # Check pattern match
    matched_pattern = detect_pattern_match(network.v, patterns, match_threshold)

    return SleepResult(
        matched_pattern=matched_pattern,
        n_steps=n_steps,
        converged=(n_steps < max_steps)
    )


def run_sleep_phase(
    network: ContinuousHopfieldNetwork,
    patterns: torch.Tensor,
    max_queries: int = 200,
    beta: float = 0.1,
    delta: float = 0.01,
    noise_stddev: float = 0.01,
    max_steps_per_query: int = 1000,
    match_threshold: float = 0.8,
    stop_on_spurious: bool = False,
    stop_on_all_found: bool = False,
    verbose: bool = False
) -> SleepPhaseResults:
    """Run full sleep phase with inhibition potentiation.

    For each query:
    1. Reset to neutral
    2. Run depressed dynamics until convergence
    3. Check pattern match
    4. Potentiate inhibition
    5. Track metrics

    Args:
        network: CHN network with trained weights
        patterns: Binary patterns [K, N]
        max_queries: Maximum number of queries
        beta: Inhibition potentiation rate
        delta: Integration timestep
        noise_stddev: Noise standard deviation
        max_steps_per_query: Max steps per query
        match_threshold: Pattern match threshold
        stop_on_spurious: Stop if spurious attractor found
        stop_on_all_found: Stop if all patterns found
        verbose: Print progress

    Returns:
        SleepPhaseResults
    """
    # Update network parameters
    network.delta = delta
    network.reset_inhibition()

    K = patterns.shape[0]
    found_patterns = set()
    n_spurious = 0
    queries = []
    all_recovered_before_spurious = False

    for q in range(max_queries):
        result = run_sleep_query(
            network, patterns,
            max_steps=max_steps_per_query,
            noise_stddev=noise_stddev,
            match_threshold=match_threshold
        )

        queries.append(result)

        if result.matched_pattern >= 0:
            found_patterns.add(result.matched_pattern)
            if verbose:
                print(f"  Query {q+1}: found pattern {result.matched_pattern} "
                      f"({len(found_patterns)}/{K} found)")
        else:
            n_spurious += 1
            if verbose:
                print(f"  Query {q+1}: spurious attractor")

            # Check if all patterns found before this spurious
            if len(found_patterns) == K and n_spurious == 1:
                all_recovered_before_spurious = True

            if stop_on_spurious:
                break

        # Potentiate inhibition
        network.pot_inhib_diag(beta)

        # Check early stop
        if stop_on_all_found and len(found_patterns) == K:
            all_recovered_before_spurious = (n_spurious == 0)
            break

    # Final check for AR success
    if len(found_patterns) == K and n_spurious == 0:
        all_recovered_before_spurious = True

    return SleepPhaseResults(
        found_patterns=found_patterns,
        n_spurious=n_spurious,
        queries=queries,
        all_recovered_before_spurious=all_recovered_before_spurious
    )


def run_sleep_phase_batched(
    networks: BatchedCHN,
    patterns_list: List[torch.Tensor],
    max_queries: int = 200,
    beta: float = 0.1,
    delta: float = 0.01,
    noise_stddev: float = 0.01,
    max_steps_per_query: int = 1000,
    match_threshold: float = 0.8,
    verbose: bool = False
) -> List[Dict]:
    """Run sleep phase on multiple networks in parallel.

    Args:
        networks: BatchedCHN with B networks
        patterns_list: List of B pattern tensors
        max_queries: Maximum queries per network
        beta: Inhibition rate
        delta: Timestep
        noise_stddev: Noise std
        max_steps_per_query: Max steps per query
        match_threshold: Match threshold
        verbose: Print progress

    Returns:
        List of B result dicts
    """
    B = networks.batch_size
    device = networks.device
    networks.delta = delta
    networks.reset_inhibition()

    # Track state per network
    found_patterns = [set() for _ in range(B)]
    n_spurious = [0] * B
    n_found = [0] * B
    pattern_counts = [p.shape[0] for p in patterns_list]

    # Per-query results storage
    all_query_results = [[] for _ in range(B)]

    for q in range(max_queries):
        # Reset all networks to neutral
        networks.reset_to_neutral()

        # Run dynamics
        prev_v = networks.v.clone()
        converged = torch.zeros(B, dtype=torch.bool, device=device)

        for step in range(max_steps_per_query):
            networks.depressed_step(noise_stddev)

            if (step + 1) % 10 == 0:
                v_change = (networks.v - prev_v).abs().max(dim=1).values  # [B]
                converged = converged | (v_change < 1e-4)
                prev_v = networks.v.clone()

                if converged.all():
                    break

        # Check matches for each network
        for b in range(B):
            pattern_idx = detect_pattern_match(
                networks.v[b], patterns_list[b], match_threshold
            )

            result = {
                'query': q,
                'matched_pattern': pattern_idx,
                'converged': converged[b].item()
            }
            all_query_results[b].append(result)

            if pattern_idx >= 0:
                found_patterns[b].add(pattern_idx)
                n_found[b] = len(found_patterns[b])
            else:
                n_spurious[b] += 1

        # Potentiate inhibition for all networks
        networks.pot_inhib_diag(beta)

        if verbose and (q + 1) % 20 == 0:
            avg_found = sum(n_found) / B
            avg_spurious = sum(n_spurious) / B
            print(f"  Query {q+1}: avg found={avg_found:.1f}, avg spurious={avg_spurious:.1f}")

    # Compile results
    results = []
    for b in range(B):
        K = pattern_counts[b]
        all_before_spurious = (len(found_patterns[b]) == K and n_spurious[b] == 0)

        results.append({
            'nb_fnd_pat': len(found_patterns[b]),
            'nb_spurious': n_spurious[b],
            'all_recovered_before_spurious': int(all_before_spurious),
            'found_patterns': list(found_patterns[b]),
            'queries': all_query_results[b]
        })

    return results


def results_to_dataframe(
    results: SleepPhaseResults,
    params: Dict
) -> List[Dict]:
    """Convert sleep results to list of dicts for CSV output.

    Matches C++ output format with one row per query iteration.

    Args:
        results: Sleep phase results
        params: Simulation parameters

    Returns:
        List of row dicts
    """
    rows = []
    found_so_far = set()
    spurious_so_far = 0

    for i, query in enumerate(results.queries):
        if query.matched_pattern >= 0:
            found_so_far.add(query.matched_pattern)
        else:
            spurious_so_far += 1

        row = {
            'query_iter': i,
            'nb_fnd_pat': len(found_so_far),
            'nb_spurious': spurious_so_far,
            'all_recovered_before_spurious': int(results.all_recovered_before_spurious),
            **params
        }
        rows.append(row)

    return rows
