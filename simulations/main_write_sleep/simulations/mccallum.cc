/**
 * McCallum Pseudorehearsal Algorithm for Discrete Hopfield Networks.
 *
 * Implements the pseudorehearsal method from McCallum's 2007 PhD thesis
 * "Catastrophic Forgetting and the Pseudorehearsal Solution in Hopfield Networks".
 *
 * Supports three modes:
 *   mode=0: Pseudorehearsal (probe + retrain with pseudoitems)
 *   mode=1: Delta hetero (train new pattern with heteroassociative noise only)
 *   mode=2: Delta gaussian (train new pattern with Gaussian input noise only)
 *
 * All algorithm parameters are configurable via JSON config with defaults
 * matching the original hardcoded values for backward compatibility.
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
 *     "seed": 0,
 *     "mode": 0,
 *     "base_pop": 0,
 *     "max_pseudoitems": 256,
 *     "n_probes": 2000,
 *     "stop_on_failure": 1,
 *     "eta": 0.1,
 *     "max_epochs": 500,
 *     "nu_h": 0.05,
 *     "sigma_input": 0.5
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
// McCallum Algorithm Parameters
// =============================================================================

struct McCallumParams {
    double eta             = 0.1;
    int    max_epochs      = 500;
    double error_criterion = 0.001;
    double error_smoothing = 0.9;
    double nu_h            = 0.05;
    double sigma_input     = 0.5;
    int    n_probes        = 2000;
    int    max_pseudoitems = 256;
    int    base_pop        = 0;
    int    mode            = 0;      // 0=PR, 1=delta_hetero, 2=delta_gaussian
    bool   stop_on_failure = true;

    static McCallumParams from_config(
        const std::unordered_map<std::string, double>& p
    ) {
        McCallumParams mp;
        if (p.count("eta"))             mp.eta = p.at("eta");
        if (p.count("max_epochs"))      mp.max_epochs = static_cast<int>(p.at("max_epochs"));
        if (p.count("error_criterion")) mp.error_criterion = p.at("error_criterion");
        if (p.count("error_smoothing")) mp.error_smoothing = p.at("error_smoothing");
        if (p.count("nu_h"))            mp.nu_h = p.at("nu_h");
        if (p.count("sigma_input"))     mp.sigma_input = p.at("sigma_input");
        if (p.count("n_probes"))        mp.n_probes = static_cast<int>(p.at("n_probes"));
        if (p.count("max_pseudoitems")) mp.max_pseudoitems = static_cast<int>(p.at("max_pseudoitems"));
        if (p.count("base_pop"))        mp.base_pop = static_cast<int>(p.at("base_pop"));
        if (p.count("mode"))            mp.mode = static_cast<int>(p.at("mode"));
        if (p.count("stop_on_failure")) mp.stop_on_failure = static_cast<int>(p.at("stop_on_failure")) != 0;
        return mp;
    }
};

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
 * Train network using delta learning with configurable noise scheme.
 *
 * @param net             Network to train
 * @param training_set    All patterns to train on
 * @param new_pattern_idx Index of the new pattern (-1 = no new pattern, no noise)
 * @param rng             Random number generator
 * @param mp              Algorithm parameters
 * @param apply_hetero    Apply heteroassociative noise to new pattern
 * @param apply_gauss     Apply Gaussian input noise to new pattern
 * @return Number of epochs trained
 */
int train_delta_learning(
    DiscreteHopfield& net,
    const std::vector<std::vector<double>>& training_set,
    int new_pattern_idx,
    std::mt19937& rng,
    const McCallumParams& mp,
    bool apply_hetero = true,
    bool apply_gauss = true
) {
    int N = net.size;
    double smoothed_error = 1.0;
    std::normal_distribution<double> input_noise(0.0, mp.sigma_input);

    // Create shuffled indices for training
    std::vector<int> order(training_set.size());
    std::iota(order.begin(), order.end(), 0);

    for (int epoch = 0; epoch < mp.max_epochs; ++epoch) {
        std::shuffle(order.begin(), order.end(), rng);
        double epoch_errors = 0.0;

        for (int idx : order) {
            const std::vector<double>& target = training_set[idx];
            bool is_new = (new_pattern_idx >= 0 && idx == new_pattern_idx);

            // Prepare input
            std::vector<double> input;
            if (is_new && apply_hetero) {
                input = apply_heteroassociative_noise(target, mp.nu_h, rng);
            } else {
                input = target;
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
                if (is_new && apply_gauss) {
                    h_i += input_noise(rng);
                }

                // Compute output
                double psi_i = (h_i >= 0.0) ? 1.0 : -1.0;

                // Compute error
                double error_i = target[i] - psi_i;

                if (std::abs(error_i) > 0.5) {  // error is +/-2 or 0
                    // Update weights using delta rule
                    for (int j = 0; j < N; ++j) {
                        net.weight_matrix[i][j] += mp.eta * error_i * input[j];
                    }
                    net.weight_matrix[i][i] = 0.0;  // Enforce no self-connection
                    epoch_errors += std::abs(error_i) / 2.0;
                }
            }
        }

        // Check early stopping with smoothed error
        smoothed_error = smoothed_error * mp.error_smoothing
                       + epoch_errors * (1.0 - mp.error_smoothing);
        if (smoothed_error < mp.error_criterion) {
            return epoch + 1;
        }
    }

    return mp.max_epochs;
}

// =============================================================================
// Probing Phase
// =============================================================================

/**
 * Relax network from initial state until convergence or max cycles.
 * One cycle = N random unit updates.
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
 */
std::vector<std::vector<double>> probe_for_pseudoitems(
    DiscreteHopfield& net,
    std::mt19937& rng,
    const McCallumParams& mp
) {
    int N = net.size;
    std::vector<std::vector<double>> pseudoitems;
    std::set<std::vector<int>> seen;  // Use int for set comparison

    std::uniform_int_distribution<int> coin(0, 1);

    for (int probe = 0; probe < mp.n_probes
         && static_cast<int>(pseudoitems.size()) < mp.max_pseudoitems; ++probe) {
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
// Evaluation
// =============================================================================

/**
 * Query pattern with 50% partial cue (single trial).
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

/**
 * Count how many of the first M patterns are stable fixed points.
 * (Relax from pattern itself; check convergence to the same state.)
 */
int count_stable(
    DiscreteHopfield& net,
    const std::vector<std::vector<double>>& patterns,
    int M,
    std::mt19937& rng
) {
    int N = net.size;
    int max_cycles = 4 * N;
    int count = 0;
    for (int mu = 0; mu < M; ++mu) {
        std::vector<double> state = relax_async(net, patterns[mu], max_cycles, rng);
        bool matches = true;
        for (int i = 0; i < N; ++i) {
            if (state[i] != patterns[mu][i]) {
                matches = false;
                break;
            }
        }
        if (matches) ++count;
    }
    return count;
}

// =============================================================================
// Main McCallum Simulation
// =============================================================================

/**
 * Run full McCallum simulation.
 *
 * @return M* = maximum patterns successfully stored and retrieved via query
 */
int run_mccallum_simulation(
    int sim_number,
    const std::unordered_map<std::string, double>& params,
    const std::string& output_dir
) {
    // Extract basic parameters
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

    // Build algorithm parameters from config (defaults = original hardcoded values)
    McCallumParams mp = McCallumParams::from_config(params);

    std::cout << "McCallum sim " << sim_number << ": N=" << N << ", rho=" << rho
              << ", seed=" << seed_value << ", mode=" << mp.mode
              << ", base_pop=" << mp.base_pop
              << ", max_pi=" << mp.max_pseudoitems << std::endl;

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
    results_file << "M,num_pseudoitems,epochs,all_queries_passed,num_stable" << std::endl;

    int M_star = 0;
    int start_M = 1;

    // --- Base population training (if base_pop > 0) ---
    if (mp.base_pop > 0 && mp.base_pop <= M_max) {
        std::vector<std::vector<double>> base_set(
            patterns.begin(), patterns.begin() + mp.base_pop);
        int epochs = train_delta_learning(net, base_set, -1, rng, mp, false, false);

        int stable = count_stable(net, patterns, mp.base_pop, rng);

        bool queries_ok = true;
        for (int mu = 0; mu < mp.base_pop && queries_ok; ++mu) {
            if (!query_pattern_50pct(net, patterns[mu], rng))
                queries_ok = false;
        }

        results_file << mp.base_pop << ",0," << epochs << ","
                     << (queries_ok ? 1 : 0) << "," << stable << std::endl;

        if (queries_ok) M_star = mp.base_pop;
        start_M = mp.base_pop + 1;
    }

    // --- Incorporation loop ---
    for (int M = start_M; M <= M_max; ++M) {
        std::vector<std::vector<double>> training_set;
        int new_pattern_idx;
        int num_pseudoitems = 0;
        int epochs;

        if (mp.mode == 0) {
            // ---- Pseudorehearsal: probe + retrain ----
            if (M == 1 && mp.base_pop == 0) {
                // First pattern: train alone
                training_set.push_back(patterns[0]);
                new_pattern_idx = 0;
            } else {
                // Probe for pseudoitems
                auto pseudoitems = probe_for_pseudoitems(net, rng, mp);
                training_set = std::move(pseudoitems);
                training_set.push_back(patterns[M - 1]);
                new_pattern_idx = static_cast<int>(training_set.size()) - 1;
                num_pseudoitems = static_cast<int>(training_set.size()) - 1;
            }
            epochs = train_delta_learning(net, training_set, new_pattern_idx,
                                          rng, mp, true, true);

        } else if (mp.mode == 1) {
            // ---- Delta hetero: single pattern, hetero noise only ----
            training_set.push_back(patterns[M - 1]);
            new_pattern_idx = 0;
            epochs = train_delta_learning(net, training_set, new_pattern_idx,
                                          rng, mp, true, false);

        } else {
            // ---- Delta gaussian: single pattern, gaussian noise only ----
            training_set.push_back(patterns[M - 1]);
            new_pattern_idx = 0;
            epochs = train_delta_learning(net, training_set, new_pattern_idx,
                                          rng, mp, false, true);
        }

        // Evaluate: stability + partial cue query
        int stable = count_stable(net, patterns, M, rng);

        bool all_passed = true;
        for (int mu = 0; mu < M && all_passed; ++mu) {
            if (!query_pattern_50pct(net, patterns[mu], rng)) {
                all_passed = false;
            }
        }

        // Log results
        results_file << M << "," << num_pseudoitems << "," << epochs << ","
                     << (all_passed ? 1 : 0) << "," << stable << std::endl;

        if (all_passed) {
            M_star = M;
        }

        if (!all_passed && mp.stop_on_failure) {
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
