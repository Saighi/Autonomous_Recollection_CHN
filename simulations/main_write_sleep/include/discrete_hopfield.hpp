/**
 * Discrete Hopfield Network class with Hebbian and Storkey learning rules.
 *
 * This class implements a classic discrete Hopfield network for comparison
 * with Continuous Hopfield Networks (CHN) using Autonomous Retrieval (AR).
 *
 * Key differences from CHN:
 * - Binary activations: {-1, +1} instead of continuous [0, 1]
 * - Threshold activation: sign(h_i) instead of sigmoid
 * - One-shot learning (Hebbian/Storkey) instead of gradient descent
 *
 * Learning Rules Implemented:
 * 1. Hebbian: W_ij += (1/N) * xi_i * xi_j
 *    - Simple outer product rule
 *    - Theoretical capacity: ~0.138*N patterns
 *
 * 2. Storkey: W_ij += (1/N) * [xi_i*xi_j - xi_i*h_j - h_i*xi_j]
 *    - Local field correction reduces crosstalk
 *    - Theoretical capacity: ~0.42*N patterns
 *    - h_i = sum_{k!=i} W_ik * xi_k (local field before storing)
 */

#ifndef DISCRETE_HOPFIELD_HPP
#define DISCRETE_HOPFIELD_HPP

#include <vector>
#include <random>
#include <string>

/**
 * Discrete Hopfield Network with binary {-1, +1} activations.
 */
class DiscreteHopfield {
public:
    /**
     * Construct a network of given size.
     * Initializes weight matrix to zeros.
     *
     * @param size Number of neurons in the network
     */
    explicit DiscreteHopfield(int size);

    /**
     * Network size (number of neurons).
     */
    int size;

    /**
     * Weight matrix W[i][j] (symmetric, zero diagonal).
     */
    std::vector<std::vector<double>> weight_matrix;

    // ========== Training Methods ==========

    /**
     * Train using Hebbian (outer product) rule.
     *
     * Formula: W_ij += (1/N) * xi_i * xi_j for i != j
     *
     * @param patterns Vector of patterns, each pattern is {-1, +1} values
     */
    void trainHebbian(const std::vector<std::vector<double>>& patterns);

    /**
     * Train using Storkey rule with AVX optimization.
     *
     * Formula: W_ij += (1/N) * [xi_i*xi_j - xi_i*h_j - h_i*xi_j]
     * Where h_i = sum_{k!=i} W_ik * xi_k (local field before this pattern)
     *
     * @param patterns Vector of patterns, each pattern is {-1, +1} values
     */
    void trainStorkeyAVX(const std::vector<std::vector<double>>& patterns);

    /**
     * Train using Storkey rule (scalar version, for verification).
     *
     * @param patterns Vector of patterns, each pattern is {-1, +1} values
     */
    void trainStorkey(const std::vector<std::vector<double>>& patterns);

    // ========== Dynamics ==========

    /**
     * Run asynchronous dynamics until convergence or max steps.
     *
     * Each step updates all neurons in sequence:
     * s_i = sign(sum_{j!=i} W_ij * s_j)
     *
     * @param initial_state Initial state vector {-1, +1}
     * @param nb_steps Number of full network sweeps
     * @return Final state after convergence
     */
    std::vector<double> runAsynchronous(const std::vector<double>& initial_state, int nb_steps);

    /**
     * Run synchronous dynamics (all neurons update simultaneously).
     *
     * @param initial_state Initial state vector {-1, +1}
     * @param steps Number of synchronous updates
     * @return Final state after updates
     */
    std::vector<double> runSynchronous(const std::vector<double>& initial_state, int steps);

    /**
     * Run synchronous dynamics until convergence (no unit changes).
     *
     * @param initial_state Initial state vector {-1, +1}
     * @param max_steps Maximum number of synchronous updates
     * @param steps_taken Output: actual number of steps taken
     * @return Final state after convergence or max_steps
     */
    std::vector<double> runSynchronousUntilConvergence(
        const std::vector<double>& initial_state,
        int max_steps,
        int& steps_taken
    );

    // ========== Query Helpers ==========

    /**
     * Create a partial cue from a pattern.
     *
     * Keeps informed_fraction of units with their pattern values,
     * sets the remaining (1-informed_fraction) units to random {-1, +1}.
     *
     * @param pattern Original pattern {-1, +1}
     * @param informed_fraction Fraction of units to keep (0 to 1)
     * @param rng Random number generator
     * @return Partial cue vector
     */
    std::vector<double> createPartialCue(
        const std::vector<double>& pattern,
        double informed_fraction,
        std::mt19937& rng
    );

    /**
     * Check if state matches pattern (or its inverse).
     *
     * In Hopfield networks, patterns and their inverses are both attractors,
     * so we check for both.
     *
     * @param state State to check
     * @param pattern Reference pattern
     * @return True if state equals pattern or -pattern
     */
    bool matchesPattern(
        const std::vector<double>& state,
        const std::vector<double>& pattern
    );

    /**
     * Reset weights to zero.
     */
    void reset();

private:
    /**
     * AVX-optimized dot product.
     *
     * @param wRow Weight row pointer
     * @param vec Vector pointer
     * @param length Vector length
     * @return Dot product result
     */
    double avxDotProduct(const double* wRow, const double* vec, int length);

    /**
     * AVX-optimized dot product excluding diagonal.
     *
     * @param wRow Weight row pointer
     * @param vec Vector pointer
     * @param length Vector length
     * @param skip_idx Index to skip (diagonal)
     * @return Dot product result excluding W[skip_idx] * vec[skip_idx]
     */
    double avxDotProductNoDiag(const double* wRow, const double* vec, int length, int skip_idx);
};

#endif // DISCRETE_HOPFIELD_HPP
