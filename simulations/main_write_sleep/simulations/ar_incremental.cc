/**
 * Autonomous Retrieval (AR) with Incremental Pattern Incorporation.
 *
 * Implements Continuous Incorporation for CHN: after each new pattern is added,
 * run sleep (autonomous retrieval) to consolidate memories. If a spurious state
 * is encountered during sleep, the incorporation fails.
 *
 * Key differences from McCallum:
 * - Uses CHN (continuous activations) instead of DHN (discrete)
 * - Uses GDA learning instead of delta rule
 * - Sleep = autonomous retrieval with self-inhibition (AR)
 * - Spurious during sleep = FAILURE (vs McCallum where spurious becomes pseudoitem)
 * - Spurious during query = FAILURE (same as McCallum)
 *
 * Usage: ./bin/ar_incremental <config.json>
 */

#include "network.hpp"
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

namespace fs = std::filesystem;

// =============================================================================
// AR Parameters
// =============================================================================
constexpr double DEFAULT_LEAK = 1.0;
constexpr double DEFAULT_DRIVE_TARGET = 6.0;
constexpr double DEFAULT_LEARNING_RATE = 0.0001;
constexpr double DEFAULT_MOMENTUM = 0.9;
constexpr double DEFAULT_DELTA = 0.01;
constexpr double DEFAULT_BETA = 0.1;
constexpr double DEFAULT_STDDEV_DYNAMICS = 0.01;
constexpr int DEFAULT_MAX_SLEEP_QUERIES = 100;

// =============================================================================
// Sleep (Autonomous Retrieval) Functions
// =============================================================================

/**
 * Run single retrieval cycle: biased phase (with inhibition) + free phase.
 * Returns the final pattern as a boolean vector after winner-take-all.
 */
std::vector<bool> run_single_retrieval(
    Network& net,
    int network_size,
    double delta,
    bool noise,
    double stddev,
    bool use_inhibition
) {
    SimulationConfig config;
    config.delta = delta;
    config.epsilon = delta / 1000.0;
    config.noise = noise;
    config.stddev = stddev;
    config.max_iter = static_cast<int>(50 / delta);
    config.use_full_inhibition = false;  // Diagonal only

    // Reset to neutral state
    net.set_state(std::vector<double>(network_size, 0.5));

    // Biased phase (with inhibition)
    if (use_inhibition) {
        config.depressed = true;
        run_net_sim_choice(net, config);
    }

    // Free phase (without inhibition)
    config.depressed = false;
    run_net_sim_choice(net, config);

    // Threshold to get binary pattern
    std::vector<bool> winners(network_size, false);
    for (int i = 0; i < network_size; ++i) {
        winners[i] = net.rate_list[i] > 0.5;
    }

    return winners;
}

/**
 * Check if a pattern matches any stored pattern.
 * Returns the index of the matching pattern, or -1 if spurious.
 */
int find_matching_pattern(
    const std::vector<bool>& state,
    const std::vector<std::vector<bool>>& patterns
) {
    for (size_t idx = 0; idx < patterns.size(); ++idx) {
        if (state == patterns[idx]) {
            return static_cast<int>(idx);
        }
    }
    return -1;  // Spurious
}

/**
 * Run sleep phase: autonomous retrieval cycles until all patterns found or spurious.
 *
 * @param net Network
 * @param patterns Currently stored patterns (to verify retrievals)
 * @param delta Integration timestep
 * @param beta Inhibition plasticity rate
 * @param max_queries Maximum retrieval cycles
 * @param noise_dynamics Whether to use noisy dynamics
 * @param stddev_dynamics Noise standard deviation
 * @return Pair: (success, set of retrieved pattern indices)
 *         success = true if all patterns retrieved without spurious
 */
std::pair<bool, std::set<int>> run_sleep_phase(
    Network& net,
    const std::vector<std::vector<bool>>& patterns,
    double delta,
    double beta,
    int max_queries,
    bool noise_dynamics,
    double stddev_dynamics
) {
    int network_size = net.size;
    std::set<int> found_indices;
    size_t num_patterns = patterns.size();

    for (int query = 0; query < max_queries; ++query) {
        // Check if all patterns found
        if (found_indices.size() == num_patterns) {
            return {true, found_indices};
        }

        // Run one retrieval cycle
        std::vector<bool> retrieved = run_single_retrieval(
            net, network_size, delta, noise_dynamics, stddev_dynamics, true
        );

        // Apply inhibitory plasticity
        net.pot_inhib_diag(beta);

        // Check what was retrieved
        int match_idx = find_matching_pattern(retrieved, patterns);

        if (match_idx >= 0) {
            found_indices.insert(match_idx);
        } else {
            // SPURIOUS STATE DURING SLEEP = FAILURE
            return {false, found_indices};
        }
    }

    // Max queries reached: check if all found
    return {found_indices.size() == num_patterns, found_indices};
}

// =============================================================================
// Query (50% Partial Cue for CHN)
// =============================================================================

/**
 * Query pattern with 50% partial cue (single trial).
 * Consistent with AR evaluation: spurious = failure.
 *
 * @param net Network
 * @param pattern Target pattern (bool)
 * @param delta Integration timestep
 * @param rng Random number generator
 * @return True if correct pattern retrieved
 */
bool query_pattern_50pct_chn(
    Network& net,
    const std::vector<bool>& pattern,
    double delta,
    std::mt19937& rng
) {
    int N = net.size;
    int n_informed = N / 2;

    // Create partial cue: uninformed units at neutral 0.5
    std::vector<double> cue(N, 0.5);

    // Select informed indices
    std::vector<int> indices(N);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);

    // Set informed units to pattern values
    double up_rate = 0.95;
    double down_rate = 0.05;

    for (int k = 0; k < n_informed; ++k) {
        int idx = indices[k];
        cue[idx] = pattern[idx] ? up_rate : down_rate;
    }

    // Set network state
    net.set_state(cue);

    // Run dynamics
    SimulationConfig config;
    config.delta = delta;
    config.epsilon = delta / 1000.0;
    config.noise = false;
    config.max_iter = static_cast<int>(50 / delta);
    config.depressed = false;

    run_net_sim_choice(net, config);

    // Threshold to get binary result
    std::vector<bool> result(N);
    for (int i = 0; i < N; ++i) {
        result[i] = net.rate_list[i] > 0.5;
    }

    // Check if correct
    return result == pattern;
}

// =============================================================================
// GDA Training
// =============================================================================

/**
 * Train network on a set of patterns using batch GDA.
 */
void train_gda(
    Network& net,
    const std::vector<std::vector<bool>>& patterns,
    double drive_target,
    double learning_rate,
    double momentum_coef,
    double leak,
    int max_iter
) {
    int N = net.size;

    // Convert patterns to target drives
    std::vector<std::vector<double>> target_drives;
    for (const auto& pat : patterns) {
        std::vector<double> drives(N);
        for (int i = 0; i < N; ++i) {
            drives[i] = pat[i] ? drive_target : -drive_target;
        }
        target_drives.push_back(std::move(drives));
    }

    // Initialize velocity matrices
    std::vector<std::vector<double>> velocity_matrix(N, std::vector<double>(N, 0.0));
    std::vector<double> velocity_bias(N, 0.0);
    std::vector<double> drive_errors(N, 0.0);

    double epsilon = learning_rate / 1000000.0;
    double max_error = 1000.0;
    int iter = 0;

    while (max_error > epsilon && iter < max_iter) {
        for (size_t p = 0; p < patterns.size(); ++p) {
            net.derivative_gradient_descent_with_bias_and_momentum_avx(
                target_drives[p],
                learning_rate,
                leak,
                drive_errors,
                velocity_matrix,
                velocity_bias,
                momentum_coef
            );
        }
        max_error = std::abs(*std::max_element(drive_errors.begin(), drive_errors.end()));
        iter++;
    }
}

// =============================================================================
// Main AR Incremental Simulation
// =============================================================================

int run_ar_incremental_simulation(
    int sim_number,
    const std::unordered_map<std::string, double>& params,
    const std::string& output_dir
) {
    // Extract parameters
    int N = static_cast<int>(params.at("network_size"));
    int M_max = params.count("max_patterns") ? static_cast<int>(params.at("max_patterns")) : 50;
    double rho = params.count("rho") ? params.at("rho") : 0.0;
    double leak = params.count("leak") ? params.at("leak") : DEFAULT_LEAK;
    double drive_target = params.count("drive_target") ? params.at("drive_target") : DEFAULT_DRIVE_TARGET;
    double learning_rate = params.count("learning_rate") ? params.at("learning_rate") : DEFAULT_LEARNING_RATE;
    double momentum = params.count("momentum_coef") ? params.at("momentum_coef") : DEFAULT_MOMENTUM;
    double delta = params.count("delta") ? params.at("delta") : DEFAULT_DELTA;
    double beta = params.count("beta") ? params.at("beta") : DEFAULT_BETA;
    double stddev_dynamics = params.count("stddev_dynamics") ? params.at("stddev_dynamics") : DEFAULT_STDDEV_DYNAMICS;
    int max_sleep_queries = params.count("max_sleep_queries") ? static_cast<int>(params.at("max_sleep_queries")) : DEFAULT_MAX_SLEEP_QUERIES;
    bool noise_dynamics = params.count("noise_dynamics") ? params.at("noise_dynamics") > 0.5 : true;
    int max_iter = params.count("max_iter") ? static_cast<int>(params.at("max_iter")) : static_cast<int>(10.0 / learning_rate);

    // Seed random
    unsigned int seed_value;
    if (params.count("seed")) {
        seed_value = static_cast<unsigned int>(params.at("seed"));
    } else {
        seed_value = static_cast<unsigned int>(sim_number);
    }
    std::mt19937 rng(seed_value);
    srand(seed_value);  // For pattern generation

    std::cout << "AR-incremental sim " << sim_number << ": N=" << N << ", rho=" << rho
              << ", seed=" << seed_value << std::endl;

    // Create simulation output directory
    std::string sim_dir = output_dir + "/sim_nb_" + std::to_string(sim_number);
    fs::create_directories(sim_dir);

    // Generate all patterns upfront
    std::vector<std::vector<bool>> all_patterns = generatePatterns(M_max, N, 0.5, rho, false, rng);

    // Build fully connected network
    std::vector<std::vector<bool>> connectivity(N, std::vector<bool>(N, false));
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (i != j) connectivity[i][j] = true;
        }
    }

    Network net(connectivity, N, leak);

    // Track results per incorporation
    std::ofstream results_file(sim_dir + "/results.data");
    results_file << "M,sleep_success,patterns_retrieved_in_sleep,all_queries_passed" << std::endl;

    int M_star = 0;
    std::vector<std::vector<bool>> current_patterns;

    // Incorporation loop
    for (int M = 1; M <= M_max; ++M) {
        bool sleep_success = true;
        int patterns_retrieved = 0;

        if (M > 1) {
            // Run sleep phase for consolidation BEFORE adding new pattern
            // Sleep checks against previously trained patterns only
            net.reset_inhib();

            auto [success, retrieved_indices] = run_sleep_phase(
                net, current_patterns, delta, beta, max_sleep_queries,
                noise_dynamics, stddev_dynamics
            );

            sleep_success = success;
            patterns_retrieved = static_cast<int>(retrieved_indices.size());

            if (!sleep_success) {
                // Spurious during sleep = FAILURE
                results_file << M << ",0," << patterns_retrieved << ",0" << std::endl;
                break;
            }
        }

        // NOW add the new pattern (after successful sleep)
        current_patterns.push_back(all_patterns[M - 1]);

        // Train on all current patterns (previously retrieved + new)
        train_gda(net, current_patterns, drive_target, learning_rate, momentum, leak, max_iter);

        // Query all stored patterns with 50% partial cues
        bool all_passed = true;
        for (int mu = 0; mu < M && all_passed; ++mu) {
            if (!query_pattern_50pct_chn(net, current_patterns[mu], delta, rng)) {
                all_passed = false;
            }
        }

        results_file << M << "," << (sleep_success ? 1 : 0) << ","
                     << (M == 1 ? 0 : patterns_retrieved) << ","
                     << (all_passed ? 1 : 0) << std::endl;

        if (all_passed) {
            M_star = M;
        } else {
            // Query failed
            break;
        }
    }

    results_file.close();

    // Save patterns (bool format)
    std::ofstream patterns_file(sim_dir + "/patterns.data");
    for (const auto& pat : all_patterns) {
        writeBoolToCSV(patterns_file, pat);
    }
    patterns_file.close();

    // Save final weights
    writeMatrixToFile(net.weight_matrix, sim_dir + "/weights.data");

    // Save connectivity
    writeBoolMatrixToFile(connectivity, sim_dir + "/connectivity.data");

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

    if (config.type != "ar_incremental") {
        std::cerr << "Error: Config type must be 'ar_incremental', got '" << config.type << "'" << std::endl;
        return 1;
    }

    // Validate required parameters
    std::vector<std::string> required = {"network_size"};
    for (const auto& param : required) {
        bool found = config.base_params.count(param) || config.varying_params.count(param);
        if (!found) {
            std::cerr << "Error: '" << param << "' required for AR incremental simulation" << std::endl;
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
    std::cout << "Running " << combinations.size() << " AR incremental simulations" << std::endl;

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
            run_ar_incremental_simulation(sim_number, combinations[sim_number], config.output_dir);
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

    std::cout << "AR incremental simulations complete. Results in: " << config.output_dir << std::endl;
    return 0;
}
