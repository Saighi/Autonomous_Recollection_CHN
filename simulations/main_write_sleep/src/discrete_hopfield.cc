/**
 * Implementation of Discrete Hopfield Network with AVX-optimized learning rules.
 *
 * Contains implementations for:
 * - Hebbian learning rule
 * - Storkey learning rule (scalar and AVX versions)
 * - Asynchronous and synchronous dynamics
 * - Partial cue generation for query testing
 */

#include "discrete_hopfield.hpp"
#include <immintrin.h>  // AVX intrinsics
#include <cmath>
#include <algorithm>
#include <numeric>

// ============================================================================
// AVX Helper Functions
// ============================================================================

double DiscreteHopfield::avxDotProduct(const double* wRow, const double* vec, int length) {
    __m256d vsum = _mm256_setzero_pd();

    int idx = 0;
    // Process 4 doubles at a time
    for (; idx + 4 <= length; idx += 4) {
        __m256d vw = _mm256_loadu_pd(&wRow[idx]);
        __m256d vv = _mm256_loadu_pd(&vec[idx]);
        vsum = _mm256_fmadd_pd(vw, vv, vsum);  // vsum += vw * vv
    }

    // Horizontal sum
    alignas(32) double partial[4];
    _mm256_storeu_pd(partial, vsum);
    double sum = partial[0] + partial[1] + partial[2] + partial[3];

    // Scalar remainder
    for (; idx < length; idx++) {
        sum += wRow[idx] * vec[idx];
    }

    return sum;
}

double DiscreteHopfield::avxDotProductNoDiag(const double* wRow, const double* vec, int length, int skip_idx) {
    __m256d vsum = _mm256_setzero_pd();

    int idx = 0;
    // Process 4 doubles at a time
    for (; idx + 4 <= length; idx += 4) {
        __m256d vw = _mm256_loadu_pd(&wRow[idx]);
        __m256d vv = _mm256_loadu_pd(&vec[idx]);
        vsum = _mm256_fmadd_pd(vw, vv, vsum);
    }

    // Horizontal sum
    alignas(32) double partial[4];
    _mm256_storeu_pd(partial, vsum);
    double sum = partial[0] + partial[1] + partial[2] + partial[3];

    // Scalar remainder
    for (; idx < length; idx++) {
        sum += wRow[idx] * vec[idx];
    }

    // Subtract diagonal element (since we included it in the loop)
    if (skip_idx >= 0 && skip_idx < length) {
        sum -= wRow[skip_idx] * vec[skip_idx];
    }

    return sum;
}

// ============================================================================
// Constructor and Initialization
// ============================================================================

DiscreteHopfield::DiscreteHopfield(int size) : size(size) {
    weight_matrix = std::vector<std::vector<double>>(size, std::vector<double>(size, 0.0));
}

void DiscreteHopfield::reset() {
    for (int i = 0; i < size; ++i) {
        std::fill(weight_matrix[i].begin(), weight_matrix[i].end(), 0.0);
    }
}

// ============================================================================
// Hebbian Learning Rule
// ============================================================================

void DiscreteHopfield::trainHebbian(const std::vector<std::vector<double>>& patterns) {
    double inv_n = 1.0 / static_cast<double>(size);

    for (const auto& xi : patterns) {
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                if (i != j) {
                    weight_matrix[i][j] += inv_n * xi[i] * xi[j];
                }
            }
        }
    }
}

// ============================================================================
// Storkey Learning Rule (Scalar Version)
// ============================================================================

void DiscreteHopfield::trainStorkey(const std::vector<std::vector<double>>& patterns) {
    std::vector<double> h(size);  // local field buffer
    double inv_n = 1.0 / static_cast<double>(size);

    for (const auto& xi : patterns) {
        // Step 1: Compute local fields h_i = sum_{k!=i} W_ik * xi_k
        for (int i = 0; i < size; ++i) {
            double sum = 0.0;
            for (int k = 0; k < size; ++k) {
                if (k != i) {
                    sum += weight_matrix[i][k] * xi[k];
                }
            }
            h[i] = sum;
        }

        // Step 2: Update weights using Storkey rule
        // W_ij += (1/N) * [xi_i*xi_j - xi_i*h_j - h_i*xi_j]
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                if (i != j) {
                    double update = inv_n * (xi[i] * xi[j] - xi[i] * h[j] - h[i] * xi[j]);
                    weight_matrix[i][j] += update;
                }
            }
        }
    }
}

// ============================================================================
// Storkey Learning Rule (AVX Optimized)
// ============================================================================

void DiscreteHopfield::trainStorkeyAVX(const std::vector<std::vector<double>>& patterns) {
    std::vector<double> h(size);  // local field buffer
    double inv_n = 1.0 / static_cast<double>(size);
    __m256d v_inv_n = _mm256_set1_pd(inv_n);

    for (const auto& xi : patterns) {
        // Step 1: Compute local fields h_i = sum_{k!=i} W_ik * xi_k (AVX optimized)
        for (int i = 0; i < size; ++i) {
            h[i] = avxDotProductNoDiag(weight_matrix[i].data(), xi.data(), size, i);
        }

        // Step 2: Update weights with AVX
        // W_ij += (1/N) * [xi_i*xi_j - xi_i*h_j - h_i*xi_j]
        for (int i = 0; i < size; ++i) {
            double xi_i = xi[i];
            double h_i = h[i];
            __m256d v_xi_i = _mm256_set1_pd(xi_i);
            __m256d v_h_i = _mm256_set1_pd(h_i);

            int j = 0;
            // Process 4 weights at a time
            for (; j + 4 <= size; j += 4) {
                __m256d v_xi_j = _mm256_loadu_pd(&xi[j]);
                __m256d v_h_j = _mm256_loadu_pd(&h[j]);

                // term1 = xi_i * xi_j
                __m256d term1 = _mm256_mul_pd(v_xi_i, v_xi_j);

                // term2 = xi_i * h_j
                __m256d term2 = _mm256_mul_pd(v_xi_i, v_h_j);

                // term3 = h_i * xi_j
                __m256d term3 = _mm256_mul_pd(v_h_i, v_xi_j);

                // update = (term1 - term2 - term3) / N
                __m256d update = _mm256_sub_pd(term1, term2);
                update = _mm256_sub_pd(update, term3);
                update = _mm256_mul_pd(update, v_inv_n);

                // W[i][j] += update
                __m256d v_w = _mm256_loadu_pd(&weight_matrix[i][j]);
                v_w = _mm256_add_pd(v_w, update);
                _mm256_storeu_pd(&weight_matrix[i][j], v_w);
            }

            // Scalar remainder
            for (; j < size; ++j) {
                if (i != j) {
                    double update = inv_n * (xi_i * xi[j] - xi_i * h[j] - h_i * xi[j]);
                    weight_matrix[i][j] += update;
                }
            }

            // Ensure diagonal remains zero
            weight_matrix[i][i] = 0.0;
        }
    }
}

// ============================================================================
// Network Dynamics
// ============================================================================

std::vector<double> DiscreteHopfield::runAsynchronous(const std::vector<double>& initial_state, int nb_steps) {
    std::vector<double> state = initial_state;

    for (int step = 0; step < nb_steps; ++step) {
        for (int i = 0; i < size; ++i) {
            // Compute local field using AVX-optimized dot product
            double h_i = avxDotProductNoDiag(weight_matrix[i].data(), state.data(), size, i);

            // Threshold activation: s_i = sign(h_i)
            state[i] = (h_i >= 0.0) ? 1.0 : -1.0;
        }
    }

    return state;
}

std::vector<double> DiscreteHopfield::runSynchronous(const std::vector<double>& initial_state, int steps) {
    std::vector<double> state = initial_state;
    std::vector<double> new_state(size);

    for (int step = 0; step < steps; ++step) {
        // Compute all local fields first
        for (int i = 0; i < size; ++i) {
            double h_i = avxDotProductNoDiag(weight_matrix[i].data(), state.data(), size, i);
            new_state[i] = (h_i >= 0.0) ? 1.0 : -1.0;
        }
        // Then update all states simultaneously
        state = new_state;
    }

    return state;
}

std::vector<double> DiscreteHopfield::runSynchronousUntilConvergence(
    const std::vector<double>& initial_state,
    int max_steps,
    int& steps_taken
) {
    std::vector<double> state = initial_state;
    std::vector<double> new_state(size);

    for (int step = 0; step < max_steps; ++step) {
        // Compute all local fields first
        for (int i = 0; i < size; ++i) {
            double h_i = avxDotProductNoDiag(weight_matrix[i].data(), state.data(), size, i);
            new_state[i] = (h_i >= 0.0) ? 1.0 : -1.0;
        }

        // Check for convergence (no unit changed)
        bool converged = true;
        for (int i = 0; i < size; ++i) {
            if (new_state[i] != state[i]) {
                converged = false;
                break;
            }
        }

        // Update state
        state = new_state;
        steps_taken = step + 1;

        if (converged) {
            return state;
        }
    }

    return state;
}

// ============================================================================
// Query Helpers
// ============================================================================

std::vector<double> DiscreteHopfield::createPartialCue(
    const std::vector<double>& pattern,
    double informed_fraction,
    std::mt19937& rng
) {
    std::vector<double> cue = pattern;  // Start with full pattern

    // Determine which indices to randomize
    int num_to_randomize = static_cast<int>(size * (1.0 - informed_fraction));

    // Create index list and shuffle
    std::vector<int> indices(size);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);

    // Randomize the first num_to_randomize indices
    std::uniform_int_distribution<int> coin(0, 1);
    for (int k = 0; k < num_to_randomize; ++k) {
        int idx = indices[k];
        cue[idx] = coin(rng) ? 1.0 : -1.0;  // Random {-1, +1}
    }

    return cue;
}

bool DiscreteHopfield::matchesPattern(
    const std::vector<double>& state,
    const std::vector<double>& pattern
) {
    // Check if state equals pattern
    bool matches_direct = true;
    for (int i = 0; i < size && matches_direct; ++i) {
        if (state[i] != pattern[i]) {
            matches_direct = false;
        }
    }
    if (matches_direct) return true;

    // Check if state equals -pattern (inverse)
    bool matches_inverse = true;
    for (int i = 0; i < size && matches_inverse; ++i) {
        if (state[i] != -pattern[i]) {
            matches_inverse = false;
        }
    }
    return matches_inverse;
}
