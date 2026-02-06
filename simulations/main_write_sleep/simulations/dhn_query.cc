/**
 * Discrete Hopfield Network partial cue query simulation.
 *
 * Tests trained DHN networks with partial cues to measure retrieval capacity.
 * For each stored pattern:
 * 1. Keep informed_fraction of units with pattern values
 * 2. Set remaining units to random {-1, +1}
 * 3. Run dynamics (synchronous or asynchronous) until convergence
 * 4. Check if network recovers the correct pattern (or its inverse)
 *
 * Usage: ./bin/dhn_query <config.json>
 *
 * Config format:
 * {
 *   "type": "dhn_query",
 *   "input_dir": "/path/to/trained_networks",
 *   "output_dir": "/path/to/query_results",
 *   "base_params": {
 *     "informed_fraction": 0.5,
 *     "nb_dynamics_steps": 10,
 *     "use_synchronous": 1
 *   },
 *   "varying_params": {
 *     "informed_fraction": [0.9, 0.5, 0.2, 0.1]
 *   }
 * }
 *
 * Parameters:
 *   - use_synchronous: 1 for synchronous updates (faster, with convergence detection),
 *                      0 for asynchronous updates (default)
 *   - nb_dynamics_steps: max steps before stopping (default 10)
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
#include <algorithm>
#include <random>

namespace fs = std::filesystem;

/**
 * Query simulation for a single trained DHN network.
 */
void run_dhn_query(
    int sim_number,
    const std::vector<std::vector<double>>& weights,
    const std::vector<std::vector<bool>>& bool_patterns,
    const std::unordered_map<std::string, double>& inherited_params,
    const std::unordered_map<std::string, double>& query_params,
    const std::string& output_dir
) {
    std::cout << "DHN query sim " << sim_number << std::endl;

    // Merge parameters (query params override inherited)
    std::unordered_map<std::string, double> params = inherited_params;
    for (const auto& p : query_params) {
        params[p.first] = p.second;
    }

    // Extract parameters
    int network_size = static_cast<int>(params.at("network_size"));
    double informed_fraction = params.count("informed_fraction") ? params.at("informed_fraction") : 0.5;
    int max_steps = params.count("nb_dynamics_steps") ? static_cast<int>(params.at("nb_dynamics_steps")) : 10;
    bool use_synchronous = params.count("use_synchronous") ? (params.at("use_synchronous") > 0.5) : false;
    int num_patterns = static_cast<int>(bool_patterns.size());

    // Create simulation output directory
    std::string sim_dir = output_dir + "/sim_nb_" + std::to_string(sim_number);
    fs::create_directories(sim_dir);

    // Create network and load weights
    DiscreteHopfield net(network_size);
    net.weight_matrix = weights;

    // Convert bool patterns to {-1, +1}
    std::vector<std::vector<double>> patterns;
    patterns.reserve(num_patterns);
    for (const auto& bool_pat : bool_patterns) {
        std::vector<double> pat(network_size);
        for (int i = 0; i < network_size; ++i) {
            pat[i] = bool_pat[i] ? 1.0 : -1.0;
        }
        patterns.push_back(std::move(pat));
    }

    // Save patterns for reference
    std::ofstream patterns_file(sim_dir + "/patterns.data");
    for (const auto& pat : bool_patterns) {
        writeBoolToCSV(patterns_file, pat);
    }
    patterns_file.close();

    // Results file
    std::string results_path = sim_dir + "/results.data";
    std::ofstream results_file(results_path, std::ios::trunc);
    results_file << "pattern_idx,informed_fraction,recovered,nb_steps" << std::endl;

    // Random number generator for creating partial cues
    std::random_device rd;
    std::mt19937 rng(rd());

    int total_success = 0;
    int total_steps = 0;

    // Test each pattern
    for (int pat_idx = 0; pat_idx < num_patterns; ++pat_idx) {
        // Create partial cue
        std::vector<double> cue = net.createPartialCue(patterns[pat_idx], informed_fraction, rng);

        // Run dynamics
        int steps_taken = max_steps;
        std::vector<double> final_state;
        if (use_synchronous) {
            final_state = net.runSynchronousUntilConvergence(cue, max_steps, steps_taken);
        } else {
            final_state = net.runAsynchronous(cue, max_steps);
        }
        total_steps += steps_taken;

        // Check if pattern was recovered
        bool recovered = net.matchesPattern(final_state, patterns[pat_idx]);
        if (recovered) {
            total_success++;
        }

        // Log result
        results_file << pat_idx << ","
                     << informed_fraction << ","
                     << (recovered ? 1 : 0) << ","
                     << steps_taken << std::endl;
    }

    results_file.close();

    // Compute success rate
    double success_rate = static_cast<double>(total_success) / static_cast<double>(num_patterns);
    double avg_steps = static_cast<double>(total_steps) / static_cast<double>(num_patterns);

    // Save parameters with success metrics
    params["num_patterns"] = static_cast<double>(num_patterns);
    params["informed_fraction"] = informed_fraction;
    params["query_success_rate"] = success_rate;
    params["query_total_success"] = static_cast<double>(total_success);
    params["query_avg_steps"] = avg_steps;
    params["use_synchronous"] = use_synchronous ? 1.0 : 0.0;
    createParameterFile(sim_dir, params);

    std::cout << "Sim " << sim_number << ": " << total_success << "/" << num_patterns
              << " patterns recovered (" << (success_rate * 100) << "%) "
              << "with " << (informed_fraction * 100) << "% informed"
              << (use_synchronous ? " [sync, avg " + std::to_string(avg_steps) + " steps]" : " [async]")
              << std::endl;
}


int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <config.json>" << std::endl;
        return 1;
    }

    // Load configuration
    JsonConfig config = JsonConfig::load(argv[1]);

    if (config.type != "dhn_query") {
        std::cerr << "Error: Config type must be 'dhn_query', got '" << config.type << "'" << std::endl;
        return 1;
    }

    // Find all trained network directories
    std::vector<std::string> network_paths;
    if (fs::exists(config.input_dir) && fs::is_directory(config.input_dir)) {
        for (const auto& entry : fs::directory_iterator(config.input_dir)) {
            if (fs::is_directory(entry.path()) && entry.path().filename().string().find("sim_nb_") == 0) {
                network_paths.push_back(entry.path().string());
            }
        }
    }

    if (network_paths.empty()) {
        std::cerr << "Error: No trained networks found in " << config.input_dir << std::endl;
        return 1;
    }

    std::sort(network_paths.begin(), network_paths.end());
    std::cout << "Found " << network_paths.size() << " trained DHN networks" << std::endl;

    // Create output directory
    if (fs::exists(config.output_dir)) {
        fs::remove_all(config.output_dir);
    }
    fs::create_directories(config.output_dir);

    // Generate all query parameter combinations
    auto query_combinations = config.generateCombinations();
    std::cout << "Running " << network_paths.size() * query_combinations.size()
              << " DHN query simulations" << std::endl;

    // Thread pool execution
    const int max_threads = std::min(20, static_cast<int>(std::thread::hardware_concurrency()));
    int active_threads = 0;
    std::mutex mtx;
    std::condition_variable cv;
    std::vector<std::thread> threads;

    int sim_number = 0;

    for (const auto& net_path : network_paths) {
        // Load trained network data
        auto weights = readMatrixFromFile(net_path + "/weights.data");
        auto patterns = loadPatterns(net_path + "/patterns.data");
        auto inherited_params = readParametersFile(net_path + "/parameters.data");

        for (const auto& query_params : query_combinations) {
            {
                std::unique_lock<std::mutex> lock(mtx);
                cv.wait(lock, [&] { return active_threads < max_threads; });
                ++active_threads;
            }

            threads.emplace_back([=, &mtx, &cv, &active_threads] {
                run_dhn_query(sim_number, weights, patterns,
                             inherited_params, query_params, config.output_dir);
                {
                    std::lock_guard<std::mutex> lock(mtx);
                    --active_threads;
                }
                cv.notify_all();
            });

            sim_number++;
        }
    }

    for (auto& t : threads) {
        if (t.joinable()) t.join();
    }

    // Aggregate results
    collectSimulationDataSeries(config.output_dir);

    std::cout << "DHN query simulations complete. Results in: " << config.output_dir << std::endl;
    return 0;
}
