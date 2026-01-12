/**
 * Partial cue query simulation.
 *
 * Tests trained networks with partial cues.
 * For each pattern, keeps informed_fraction of units with their pattern values
 * and sets the remaining (1 - informed_fraction) units to neutral (0.5).
 * Checks if the network can recover the full pattern.
 *
 * Usage: ./bin/query <config.json>
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
#include <algorithm>
#include <random>

namespace fs = std::filesystem;

// Query simulation for a single trained network
void run_query(
    int sim_number,
    const std::vector<std::vector<double>>& weights,
    const std::vector<std::vector<bool>>& connectivity,
    const std::vector<std::vector<bool>>& patterns,
    const std::unordered_map<std::string, double>& inherited_params,
    const std::unordered_map<std::string, double>& query_params,
    const std::string& output_dir
) {
    std::cout << "Query sim " << sim_number << std::endl;

    // Merge parameters (query params override inherited)
    std::unordered_map<std::string, double> params = inherited_params;
    for (const auto& p : query_params) {
        params[p.first] = p.second;
    }

    // Extract parameters
    int network_size = static_cast<int>(params.at("network_size"));
    double leak = params.at("leak");
    double delta = params.count("delta") ? params.at("delta") : 0.01;
    double drive_target = params.count("drive_target") ? params.at("drive_target") : 6.0;
    double stddev_dynamics = params.count("stddev_dynamics") ? params.at("stddev_dynamics") : 0.01;
    bool noise_dynamics = params.count("noise_dynamics") ? params.at("noise_dynamics") > 0 : true;
    double informed_fraction = params.count("informed_fraction") ? params.at("informed_fraction") : 0.1;
    int num_patterns = static_cast<int>(patterns.size());
    int nb_winners = params.count("nb_winners") ? static_cast<int>(params.at("nb_winners")) :
                     static_cast<int>(params.at("relative_nb_winner") * network_size);

    // Create simulation output directory
    std::string sim_dir = output_dir + "/sim_nb_" + std::to_string(sim_number);
    fs::create_directories(sim_dir);

    // Create network and load weights
    Network net(connectivity, network_size, leak);
    net.weight_matrix = weights;

    // Save patterns for reference
    std::ofstream patterns_file(sim_dir + "/patterns.data");
    for (const auto& pattern : patterns) {
        writeBoolToCSV(patterns_file, pattern);
    }
    patterns_file.close();

    // Save parameters
    params["num_patterns"] = static_cast<double>(num_patterns);
    params["informed_fraction"] = informed_fraction;
    createParameterFile(sim_dir, params);

    // Setup simulation config
    SimulationConfig config;
    config.delta = delta;
    config.epsilon = delta / 1000;
    config.noise = noise_dynamics;
    config.stddev = stddev_dynamics;
    config.max_iter = static_cast<int>(1.0 / delta);  // Run for 1 time unit
    config.depressed = false;  // No inhibition for query
    config.save = false;

    // Results file
    std::string results_path = sim_dir + "/results.data";
    std::ofstream results_file(results_path, std::ios::trunc);
    results_file << "pattern_idx,informed_fraction,recovered,nb_iter" << std::endl;

    // Compute transfer function values for pattern encoding
    double up_rate = net.transfer(drive_target);    // ~0.997 for drive_target=6
    double down_rate = net.transfer(-drive_target); // ~0.003

    int total_success = 0;

    // Test each pattern
    for (int pat_idx = 0; pat_idx < num_patterns; ++pat_idx) {
        // Build query state from pattern
        std::vector<double> query_state = pattern_as_states(up_rate, down_rate, patterns[pat_idx]);

        // Set (1 - informed_fraction) of units to neutral (0.5)
        // Only informed_fraction of units keep their pattern values
        int num_uninformed = static_cast<int>(network_size * (1.0 - informed_fraction));
        query_state = setToValueRandomElements(query_state, num_uninformed, 0.5);

        // Initialize network with partial cue
        net.set_state(query_state);

        // Run dynamics until convergence
        int nb_iter = run_net_sim_choice(net, config);

        // Determine retrieved pattern by winner-take-all
        std::vector<bool> winners = assignBoolToTopNValues(net.rate_list, nb_winners);

        // Check if pattern was recovered
        bool recovered = (winners == patterns[pat_idx]);
        if (recovered) {
            total_success++;
        }

        // Log result
        results_file << pat_idx << ","
                     << informed_fraction << ","
                     << (recovered ? 1 : 0) << ","
                     << nb_iter << std::endl;
    }

    results_file.close();

    // Update parameters with success rate
    double success_rate = static_cast<double>(total_success) / static_cast<double>(num_patterns);
    params["query_success_rate"] = success_rate;
    params["query_total_success"] = static_cast<double>(total_success);
    createParameterFile(sim_dir, params);

    std::cout << "Sim " << sim_number << ": " << total_success << "/" << num_patterns
              << " patterns recovered (" << (success_rate * 100) << "%) "
              << "with " << (informed_fraction * 100) << "% informed" << std::endl;
}


int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <config.json>" << std::endl;
        return 1;
    }

    // Load configuration
    JsonConfig config = JsonConfig::load(argv[1]);

    if (config.type != "query") {
        std::cerr << "Error: Config type must be 'query', got '" << config.type << "'" << std::endl;
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
    std::cout << "Found " << network_paths.size() << " trained networks" << std::endl;

    // Create output directory
    if (fs::exists(config.output_dir)) {
        fs::remove_all(config.output_dir);
    }
    fs::create_directories(config.output_dir);

    // Generate all query parameter combinations
    auto query_combinations = config.generateCombinations();
    std::cout << "Running " << network_paths.size() * query_combinations.size()
              << " query simulations" << std::endl;

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
        auto connectivity = readBoolMatrixFromFile(net_path + "/connectivity.data");
        auto patterns = loadPatterns(net_path + "/patterns.data");
        auto inherited_params = readParametersFile(net_path + "/parameters.data");

        for (const auto& query_params : query_combinations) {
            {
                std::unique_lock<std::mutex> lock(mtx);
                cv.wait(lock, [&] { return active_threads < max_threads; });
                ++active_threads;
            }

            threads.emplace_back([=, &mtx, &cv, &active_threads] {
                run_query(sim_number, weights, connectivity, patterns,
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

    std::cout << "Query simulations complete. Results in: " << config.output_dir << std::endl;
    return 0;
}
