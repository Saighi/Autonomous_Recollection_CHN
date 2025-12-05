/**
 * Generic sleep/retrieval simulation.
 *
 * Reads trained networks and runs autonomous retrieval cycles with
 * inhibitory plasticity. Tracks pattern retrieval and spurious attractors.
 *
 * Usage: ./bin/sleep <config.json>
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

namespace fs = std::filesystem;

// Sleep simulation for a single trained network
void run_sleep(
    int sim_number,
    const std::vector<std::vector<double>>& weights,
    const std::vector<std::vector<bool>>& connectivity,
    const std::vector<std::vector<bool>>& patterns,
    const std::unordered_map<std::string, double>& inherited_params,
    const std::unordered_map<std::string, double>& sleep_params,
    const std::string& output_dir
) {
    std::cout << "Sleep sim " << sim_number << std::endl;

    // Merge parameters (sleep params override inherited)
    std::unordered_map<std::string, double> params = inherited_params;
    for (const auto& p : sleep_params) {
        params[p.first] = p.second;
    }

    // Extract parameters
    int network_size = static_cast<int>(params.at("network_size"));
    int nb_winners = static_cast<int>(params.at("nb_winners"));
    double leak = params.at("leak");
    double delta = params.count("delta") ? params.at("delta") : 0.01;
    double beta = params.count("beta") ? params.at("beta") : 0.1;
    double init_drive = params.count("init_drive") ? params.at("init_drive") : 0.5;
    double stddev_dynamics = params.count("stddev_dynamics") ? params.at("stddev_dynamics") : 0.01;
    bool noise_dynamics = params.count("noise_dynamics") ? params.at("noise_dynamics") > 0 : true;
    int max_queries = params.count("max_queries") ? static_cast<int>(params.at("max_queries")) : 200;
    bool save_trajectories = params.count("save_trajectories") ? params.at("save_trajectories") > 0 : false;
    bool stop_on_spurious = params.count("stop_on_spurious") ? params.at("stop_on_spurious") > 0 : true;
    bool stop_on_all_found = params.count("stop_on_all_found") ? params.at("stop_on_all_found") > 0 : false;

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
    params["num_patterns"] = static_cast<double>(patterns.size());
    createParameterFile(sim_dir, params);

    // Setup simulation config
    SimulationConfig config;
    config.delta = delta;
    config.epsilon = delta / 1000;
    config.noise = noise_dynamics;
    config.stddev = stddev_dynamics;
    config.max_iter = static_cast<int>(100 / delta);

    // Results tracking
    std::string results_path = sim_dir + "/results.data";
    std::ofstream results_file(results_path, std::ios::trunc);
    results_file << "query_iter,nb_fnd_pat,nb_spurious,nb_iter_biased,nb_iter_free," << std::endl;

    std::set<std::vector<bool>> found_patterns;
    int nb_spurious = 0;
    int query_iter = 0;
    bool all_found = false;

    // Main retrieval loop
    // Continues until max_queries, unless stopped by:
    //   - stop_on_spurious: stops when a spurious pattern is encountered
    //   - stop_on_all_found: stops when all patterns have been retrieved
    while (query_iter < max_queries) {
        // Check stop conditions
        if (stop_on_spurious && nb_spurious > 0) break;
        if (stop_on_all_found && all_found) break;
        // Reset to neutral state
        net.set_state(std::vector<double>(network_size, init_drive));

        // Optional trajectory saving
        std::string traj_path = save_trajectories ?
            sim_dir + "/results_" + std::to_string(query_iter) + ".data" : "/dev/null";
        std::ofstream traj_file(traj_path, std::ios::trunc);
        config.output = &traj_file;
        config.save = save_trajectories;

        // Biased phase (with inhibition)
        config.depressed = true;
        int nb_iter_biased = run_net_sim_choice(net, config);

        // Free phase (without inhibition)
        config.depressed = false;
        int nb_iter_free = run_net_sim_choice(net, config);

        traj_file.close();

        // Determine winning pattern
        std::vector<bool> winners = assignBoolToTopNValues(net.activity_list, nb_winners);

        // Apply inhibitory plasticity
        net.pot_inhib_symmetric(beta);

        // Check if retrieved pattern is in target set
        auto it = std::find(patterns.begin(), patterns.end(), winners);
        if (it != patterns.end()) {
            found_patterns.insert(winners);
            if (found_patterns.size() == patterns.size()) {
                all_found = true;
            }
        } else {
            nb_spurious++;
        }

        // Log results
        results_file << query_iter << ","
                     << found_patterns.size() << ","
                     << nb_spurious << ","
                     << nb_iter_biased << ","
                     << nb_iter_free << "," << std::endl;

        query_iter++;
    }

    results_file.close();

    std::cout << "Sim " << sim_number << ": found " << found_patterns.size()
              << "/" << patterns.size() << " patterns, "
              << nb_spurious << " spurious" << std::endl;
}


int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <config.json>" << std::endl;
        return 1;
    }

    // Load configuration
    JsonConfig config = JsonConfig::load(argv[1]);

    if (config.type != "sleep") {
        std::cerr << "Error: Config type must be 'sleep', got '" << config.type << "'" << std::endl;
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

    // Generate all sleep parameter combinations
    auto sleep_combinations = config.generateCombinations();
    std::cout << "Running " << network_paths.size() * sleep_combinations.size()
              << " sleep simulations" << std::endl;

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

        for (const auto& sleep_params : sleep_combinations) {
            {
                std::unique_lock<std::mutex> lock(mtx);
                cv.wait(lock, [&] { return active_threads < max_threads; });
                ++active_threads;
            }

            threads.emplace_back([=, &mtx, &cv, &active_threads] {
                run_sleep(sim_number, weights, connectivity, patterns,
                         inherited_params, sleep_params, config.output_dir);
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

    std::cout << "Sleep simulations complete. Results in: " << config.output_dir << std::endl;
    return 0;
}
