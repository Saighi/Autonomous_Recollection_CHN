/**
 * McCallum Pseudorehearsal Algorithm for Discrete Hopfield Networks.
 *
 * Implements the pseudorehearsal method from McCallum's 2007 PhD thesis
 * "Catastrophic Forgetting and the Pseudorehearsal Solution in Hopfield Networks".
 *
 * Key features:
 * - Delta learning rule with asymmetric weights
 * - Probing phase to discover stable states (pseudoitems)
 * - Noise applied ONLY to new patterns (heteroassociative + input noise)
 * - Evaluation via 50% partial cue queries
 *
 * Usage: ./bin/mccallum <config.json>
 *
 * Config format:
 * {
 *   "type": "mccallum",
 *   "output_dir": "/path/to/output",
 *   "base_params": {
 *     "network_size": 100,
 *     "max_patterns": 50,
 *     "rho": 0.0,
 *     "seed": 0
 *   },
 *   "varying_params": {
 *     "network_size": [50, 100, 150, 200, 250],
 *     "rho": [0.0, 0.2, 0.4, 0.6],
 *     "seed": [0, 1, 2, ..., 9]
 *   }
 * }
 */

#include "discrete_hopfield.hpp"
#include "utils.hpp"
#include "json_config.hpp"

#include <iostream>
#include <fstream>
#include <filesystem>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <set>
#include <algorithm>
#include <cmath>
#include <random>
#include <tuple>

namespace fs = std::filesystem;

// =============================================================================
// McCallum Algorithm Parameters (from spec)
// =============================================================================
constexpr double ETA = 0.1;              // Delta learning rate
constexpr int E_MAX = 500;               // Max epochs per incorporation
constexpr double ERROR_CRITERION = 0.001; // Early stopping threshold
constexpr double ERROR_SMOOTHING = 0.9;  // Smoothing factor for error
constexpr double NU_H = 0.05;            // 5% heteroassociative noise
constexpr double SIGMA_INPUT = 0.5;      // Gaussian input noise std
constexpr int P_PROBES = 2000;           // Number of probes for pseudoitems
constexpr int P_ITEMS = 256;             // Max pseudoitems to collect

// =============================================================================
// Delta Learning Rule Implementation
// =============================================================================

/**
 * Apply heteroassociative noise: flip nu_h fraction of bits.
 */
std::vector<double> apply_heteroassociative_noise(
    const std::vector<double>& pattern,
    double nu_h,
    std::mt19937& rng
) {
    int N = pattern.size();
    int n_flip = static_cast<int>(std::round(nu_h * N));

    std::vector<double> noisy = pattern;

    // Create and shuffle indices
    std::vector<int> indices(N);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);

    // Flip first n_flip indices
    for (int k = 0; k < n_flip; ++k) {
        noisy[indices[k]] *= -1.0;
    }

    return noisy;
}

/**
 * Train network using delta learning with McCallum's noise scheme.
 *
 * @param net Network to train
 * @param training_set All patterns to train on (pseudoitems + new pattern)
 * @param new_pattern_idx Index of the new pattern in training_set (-1 if no new pattern)
 * @param rng Random number generator
 * @return Number of epochs trained
 */
int train_delta_learning(
    DiscreteHopfield& net,
    const std::vector<std::vector<double>>& training_set,
    int new_pattern_idx,
    std::mt19937& rng
) {
    int N = net.size;
    double smoothed_error = 1.0;
    std::normal_distribution<double> input_noise(0.0, SIGMA_INPUT);

    // Create shuffled indices for training
    std::vector<int> order(training_set.size());
    std::iota(order.begin(), order.end(), 0);

    for (int epoch = 0; epoch < E_MAX; ++epoch) {
        std::shuffle(order.begin(), order.end(), rng);
        double epoch_errors = 0.0;

        for (int idx : order) {
            const std::vector<double>& target = training_set[idx];
            bool is_new_pattern = (idx == new_pattern_idx);

            // Prepare input
            std::vector<double> input;
            if (is_new_pattern) {
                // Apply heteroassociative noise to new pattern only
                input = apply_heteroassociative_noise(target, NU_H, rng);
            } else {
                input = target;  // Pseudoitems: no noise
            }

            // Update each unit
            for (int i = 0; i < N; ++i) {
                // Compute local field
                double h_i = 0.0;
                for (int j = 0; j < N; ++j) {
                    if (i != j) {
                        h_i += net.weight_matrix[i][j] * input[j];
                    }
                }

                // Add input noise for new patterns only
                if (is_new_pattern) {
                    h_i += input_noise(rng);
                }

                // Compute output
                double psi_i = (h_i >= 0.0) ? 1.0 : -1.0;

                // Compute error
                double error_i = target[i] - psi_i;

                if (std::abs(error_i) > 0.5) {  // error is ±2 or 0
                    // Update weights using delta rule
                    for (int j = 0; j < N; ++j) {
                        net.weight_matrix[i][j] += ETA * error_i * input[j];
                    }
                    net.weight_matrix[i][i] = 0.0;  // Enforce no self-connection
                    epoch_errors += std::abs(error_i) / 2.0;
                }
            }
        }

        // Check early stopping with smoothed error
        smoothed_error = smoothed_error * ERROR_SMOOTHING + epoch_errors * (1.0 - ERROR_SMOOTHING);
        if (smoothed_error < ERROR_CRITERION) {
            return epoch + 1;
        }
    }

    return E_MAX;
}

// =============================================================================
// Probing Phase
// =============================================================================

/**
 * Relax network from initial state until convergence or max cycles.
 * One cycle = N random unit updates.
 *
 * @param net Network
 * @param initial Initial state
 * @param max_cycles Maximum number of cycles (each cycle = N updates)
 * @param rng Random number generator
 * @return Final stable state
 */
std::vector<double> relax_async(
    DiscreteHopfield& net,
    const std::vector<double>& initial,
    int max_cycles,
    std::mt19937& rng
) {
    int N = net.size;
    std::vector<double> state = initial;
    std::vector<int> update_order(N);
    std::iota(update_order.begin(), update_order.end(), 0);

    for (int cycle = 0; cycle < max_cycles; ++cycle) {
        std::shuffle(update_order.begin(), update_order.end(), rng);
        bool changed = false;

        for (int i : update_order) {
            // Compute local field
            double h_i = 0.0;
            for (int j = 0; j < N; ++j) {
                if (i != j) {
                    h_i += net.weight_matrix[i][j] * state[j];
                }
            }

            double new_val = (h_i >= 0.0) ? 1.0 : -1.0;
            if (new_val != state[i]) {
                state[i] = new_val;
                changed = true;
            }
        }

        if (!changed) {
            break;  // Converged
        }
    }

    return state;
}

/**
 * Probe network to find pseudoitems (unique stable states).
 *
 * @param net Network to probe
 * @param rng Random number generator
 * @return Vector of unique stable states
 */
std::vector<std::vector<double>> probe_for_pseudoitems(
    DiscreteHopfield& net,
    std::mt19937& rng
) {
    int N = net.size;
    std::vector<std::vector<double>> pseudoitems;
    std::set<std::vector<int>> seen;  // Use int for set comparison

    std::uniform_int_distribution<int> coin(0, 1);

    for (int probe = 0; probe < P_PROBES && static_cast<int>(pseudoitems.size()) < P_ITEMS; ++probe) {
        // Generate random probe
        std::vector<double> state(N);
        for (int i = 0; i < N; ++i) {
            state[i] = coin(rng) ? 1.0 : -1.0;
        }

        // Relax to stable state
        state = relax_async(net, state, 4 * N, rng);

        // Convert to int for comparison
        std::vector<int> key(N);
        std::vector<int> inv_key(N);
        for (int i = 0; i < N; ++i) {
            key[i] = static_cast<int>(state[i]);
            inv_key[i] = -key[i];
        }

        // Check if state (or its inverse) is already seen
        if (seen.find(key) == seen.end() && seen.find(inv_key) == seen.end()) {
            seen.insert(key);
            pseudoitems.push_back(state);
        }
    }

    return pseudoitems;
}

// =============================================================================
// Query (50% Partial Cue)
// =============================================================================

/**
 * Query pattern with 50% partial cue (single trial).
 * Consistent with AR evaluation: spurious = failure.
 *
 * @param net Network
 * @param pattern Target pattern
 * @param rng Random number generator
 * @return True if correct pattern retrieved
 */
bool query_pattern_50pct(
    DiscreteHopfield& net,
    const std::vector<double>& pattern,
    std::mt19937& rng
) {
    int N = net.size;
    int n_informed = N / 2;

    // Create partial cue
    std::vector<double> cue(N);
    std::uniform_int_distribution<int> coin(0, 1);
    for (int i = 0; i < N; ++i) {
        cue[i] = coin(rng) ? 1.0 : -1.0;
    }

    // Select informed indices
    std::vector<int> indices(N);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);

    for (int k = 0; k < n_informed; ++k) {
        cue[indices[k]] = pattern[indices[k]];
    }

    // Relax
    std::vector<double> result = relax_async(net, cue, 4 * N, rng);

    // Check if correct (exact match or inverse)
    return net.matchesPattern(result, pattern);
}

// =============================================================================
// Main McCallum Simulation
// =============================================================================

/**
 * Run full McCallum pseudorehearsal simulation.
 *
 * @return M* = maximum patterns successfully stored and retrieved
 */
int run_mccallum_simulation(
    int sim_number,
    const std::unordered_map<std::string, double>& params,
    const std::string& output_dir
) {
    // Extract parameters
    int N = static_cast<int>(params.at("network_size"));
    int M_max = params.count("max_patterns") ? static_cast<int>(params.at("max_patterns")) : 50;
    double rho = params.count("rho") ? params.at("rho") : 0.0;

    // Seed random
    unsigned int seed_value;
    if (params.count("seed")) {
        seed_value = static_cast<unsigned int>(params.at("seed"));
    } else {
        seed_value = static_cast<unsigned int>(sim_number);
    }
    std::mt19937 rng(seed_value);

    std::cout << "McCallum sim " << sim_number << ": N=" << N << ", rho=" << rho
              << ", seed=" << seed_value << std::endl;

    // Create simulation output directory
    std::string sim_dir = output_dir + "/sim_nb_" + std::to_string(sim_number);
    fs::create_directories(sim_dir);

    // Generate all patterns upfront (in {0,1} then convert to {+1,-1})
    std::vector<std::vector<bool>> bool_patterns = generatePatterns(M_max, N, 0.5, rho, false, rng);

    // Convert to bipolar {+1, -1}
    std::vector<std::vector<double>> patterns;
    patterns.reserve(M_max);
    for (const auto& bp : bool_patterns) {
        std::vector<double> pat(N);
        for (int i = 0; i < N; ++i) {
            pat[i] = bp[i] ? 1.0 : -1.0;
        }
        patterns.push_back(std::move(pat));
    }

    // Initialize network with zero weights
    DiscreteHopfield net(N);

    // Track results per incorporation
    std::ofstream results_file(sim_dir + "/results.data");
    results_file << "M,num_pseudoitems,epochs,all_queries_passed" << std::endl;

    int M_star = 0;

    // Incorporation loop
    for (int M = 1; M <= M_max; ++M) {
        // Build training set
        std::vector<std::vector<double>> training_set;
        int new_pattern_idx;

        if (M == 1) {
            // First pattern: train alone
            training_set.push_back(patterns[0]);
            new_pattern_idx = 0;
        } else {
            // Probe for pseudoitems
            std::vector<std::vector<double>> pseudoitems = probe_for_pseudoitems(net, rng);

            // Training set = pseudoitems + new pattern
            training_set = std::move(pseudoitems);
            training_set.push_back(patterns[M - 1]);
            new_pattern_idx = static_cast<int>(training_set.size()) - 1;
        }

        int num_pseudoitems = static_cast<int>(training_set.size()) - 1;

        // Train with delta learning
        int epochs = train_delta_learning(net, training_set, new_pattern_idx, rng);

        // Query all stored patterns with 50% partial cues
        bool all_passed = true;
        for (int mu = 0; mu < M && all_passed; ++mu) {
            if (!query_pattern_50pct(net, patterns[mu], rng)) {
                all_passed = false;
            }
        }

        // Log results
        results_file << M << "," << num_pseudoitems << "," << epochs << ","
                     << (all_passed ? 1 : 0) << std::endl;

        if (all_passed) {
            M_star = M;
        } else {
            // Failed: M* = M - 1
            break;
        }
    }

    results_file.close();

    // Save patterns (bool format for consistency)
    std::ofstream patterns_file(sim_dir + "/patterns.data");
    for (const auto& bp : bool_patterns) {
        writeBoolToCSV(patterns_file, bp);
    }
    patterns_file.close();

    // Save final weights
    writeMatrixToFile(net.weight_matrix, sim_dir + "/weights.data");

    // Save parameters including M*
    std::unordered_map<std::string, double> saved_params = params;
    saved_params["M_star"] = static_cast<double>(M_star);
    saved_params["patterns_attempted"] = static_cast<double>(std::min(M_star + 1, M_max));
    createParameterFile(sim_dir, saved_params);

    std::cout << "Sim " << sim_number << " complete: M* = " << M_star << std::endl;

    return M_star;
}

// =============================================================================
// Main Entry Point
// =============================================================================

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <config.json>" << std::endl;
        return 1;
    }

    // Load configuration
    JsonConfig config = JsonConfig::load(argv[1]);

    if (config.type != "mccallum") {
        std::cerr << "Error: Config type must be 'mccallum', got '" << config.type << "'" << std::endl;
        return 1;
    }

    // Validate required parameters
    std::vector<std::string> required = {"network_size"};
    for (const auto& param : required) {
        bool found = config.base_params.count(param) || config.varying_params.count(param);
        if (!found) {
            std::cerr << "Error: '" << param << "' required for McCallum simulation" << std::endl;
            return 1;
        }
    }

    // Create output directory
    if (fs::exists(config.output_dir)) {
        fs::remove_all(config.output_dir);
    }
    fs::create_directories(config.output_dir);

    // Generate all parameter combinations
    auto combinations = config.generateCombinations();
    std::cout << "Running " << combinations.size() << " McCallum simulations" << std::endl;

    // Thread pool execution
    const int max_threads = std::min(20, static_cast<int>(std::thread::hardware_concurrency()));
    int active_threads = 0;
    std::mutex mtx;
    std::condition_variable cv;
    std::vector<std::thread> threads;

    for (size_t sim_number = 0; sim_number < combinations.size(); ++sim_number) {
        {
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait(lock, [&] { return active_threads < max_threads; });
            ++active_threads;
        }

        threads.emplace_back([=, &mtx, &cv, &active_threads] {
            run_mccallum_simulation(sim_number, combinations[sim_number], config.output_dir);
            {
                std::lock_guard<std::mutex> lock(mtx);
                --active_threads;
            }
            cv.notify_all();
        });
    }

    for (auto& t : threads) {
        if (t.joinable()) t.join();
    }

    // Aggregate results
    collectSimulationData(config.output_dir);

    std::cout << "McCallum simulations complete. Results in: " << config.output_dir << std::endl;
    return 0;
}
