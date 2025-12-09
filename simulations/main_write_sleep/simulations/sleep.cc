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
    const std::string& output_dir,
    const std::string& source_network_path  // Path to trained network (for metadata)
) {
    std::cout << "Sleep sim " << sim_number << std::endl;

    // Merge parameters (sleep params override inherited)
    std::unordered_map<std::string, double> params = inherited_params;
    for (const auto& p : sleep_params) {
        params[p.first] = p.second;
    }

    // Extract parameters
    int network_size = static_cast<int>(params.at("network_size"));
    double leak = params.at("leak");
    double delta = params.count("delta") ? params.at("delta") : 0.01;
    double beta = params.count("beta") ? params.at("beta") : 0.1;
    double stddev_dynamics = params.count("stddev_dynamics") ? params.at("stddev_dynamics") : 0.01;
    bool noise_dynamics = params.count("noise_dynamics") ? params.at("noise_dynamics") > 0 : true;
    int max_queries = params.count("max_queries") ? static_cast<int>(params.at("max_queries")) : 200;
    bool save_trajectories = params.count("save_trajectories") ? params.at("save_trajectories") > 0 : false;
    bool stop_on_spurious = params.count("stop_on_spurious") ? params.at("stop_on_spurious") > 0 : true;
    bool stop_on_all_found = params.count("stop_on_all_found") ? params.at("stop_on_all_found") > 0 : false;

    bool symmetric_transfer = params.count("symmetric_transfer") ? params.at("symmetric_transfer") > 0.5 : false;
    bool use_inhibition_plasticity = params.count("use_inhibition_plasticity") ? params.at("use_inhibition_plasticity") > 0.5 : true;
    bool use_full_inhibition = params.count("use_full_inhibition") ? params.at("use_full_inhibition") > 0.5 : false;

    // Create simulation output directory
    std::string sim_dir = output_dir + "/sim_nb_" + std::to_string(sim_number);
    fs::create_directories(sim_dir);

    // Create network and load weights
    Network net(connectivity, network_size, leak, symmetric_transfer);
    net.weight_matrix = weights;

    // Save patterns for reference
    std::ofstream patterns_file(sim_dir + "/patterns.data");
    for (const auto& pattern : patterns) {
        writeBoolToCSV(patterns_file, pattern);
    }
    patterns_file.close();

    // Copy pattern metadata if it exists (for heterogeneous pattern support)
    std::string metadata_src = source_network_path + "/pattern_metadata.json";
    std::string metadata_dst = sim_dir + "/pattern_metadata.json";
    if (fs::exists(metadata_src)) {
        fs::copy_file(metadata_src, metadata_dst, fs::copy_options::overwrite_existing);
    }

    // Save parameters (note: all_recovered_before_spurious will be updated later)
    params["num_patterns"] = static_cast<double>(patterns.size());
    params["all_recovered_before_spurious"] = 0.0;  // Will be updated after retrieval loop
    params["first_iter_all_found"] = -1.0;  // Will be updated after retrieval loop
    createParameterFile(sim_dir, params);

    // Setup simulation config
    SimulationConfig config;
    config.delta = delta;
    config.epsilon = delta / 1000;
    config.noise = noise_dynamics;
    config.stddev = stddev_dynamics;
    config.max_iter = static_cast<int>(50 / delta);
    config.use_full_inhibition = use_full_inhibition;

    // Results tracking
    std::string results_path = sim_dir + "/results.data";
    std::ofstream results_file(results_path, std::ios::trunc);
    results_file << "query_iter,nb_fnd_pat,nb_spurious,nb_iter_biased,nb_iter_free,all_recovered_before_spurious,recovered_pattern_idx" << std::endl;

    std::set<int> found_pattern_indices;  // Track which pattern indices were recovered
    int nb_spurious = 0;
    int query_iter = 0;
    bool all_found = false;
    bool all_recovered_before_spurious = false;  // Track if all found before any spurious
    int first_iter_all_found = -1;  // Track when all patterns first found

    // Main retrieval loop
    // Continues until max_queries, unless stopped by:
    //   - stop_on_spurious: stops when a spurious pattern is encountered
    //   - stop_on_all_found: stops when all patterns have been retrieved
    while (query_iter < max_queries) {
        // Check stop conditions
        if (stop_on_spurious && nb_spurious > 0) break;
        if (stop_on_all_found && all_found) break;
        // Reset to neutral state (depends on transfer type)
        double init_rate = symmetric_transfer ? 0.0 : 0.5;
        net.set_state(std::vector<double>(network_size, init_rate));

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

        // Determine retrieved pattern by thresholding final rates
        std::vector<bool> winners(network_size, false);
        double threshold = symmetric_transfer ? 0.0 : 0.5;
        for (int i = 0; i < network_size; ++i) {
            winners[i] = net.rate_list[i] > threshold;
        }

        // Apply inhibitory plasticity (if enabled)
        if (use_inhibition_plasticity) {
            if (use_full_inhibition) {
                net.pot_inhib_full_matrix(beta);
            } else {
                net.pot_inhib_diag(beta);
            }
        }

        // Check if retrieved pattern is in target set
        // For symmetric transfer, also accept the converse pattern (all bits flipped)
        bool pattern_found = false;
        int recovered_pattern_idx = -1;  // -1 = spurious (not in stored set)

        if (symmetric_transfer) {
            // Check for exact match or converse match
            for (size_t idx = 0; idx < patterns.size(); ++idx) {
                if (matchesPatternOrConverse(winners, patterns[idx])) {
                    pattern_found = true;
                    recovered_pattern_idx = static_cast<int>(idx);
                    found_pattern_indices.insert(static_cast<int>(idx));
                    break;
                }
            }
        } else {
            // Standard matching: exact match only
            for (size_t idx = 0; idx < patterns.size(); ++idx) {
                if (patterns[idx] == winners) {
                    pattern_found = true;
                    recovered_pattern_idx = static_cast<int>(idx);
                    found_pattern_indices.insert(static_cast<int>(idx));
                    break;
                }
            }
        }

        if (pattern_found) {
            if (found_pattern_indices.size() == patterns.size() && !all_found) {
                all_found = true;
                first_iter_all_found = query_iter;  // Record first time all found
                if (nb_spurious == 0) {
                    all_recovered_before_spurious = true;  // Success!
                }
            }
        } else {
            nb_spurious++;
        }

        // Log results
        results_file << query_iter << ","
                     << found_pattern_indices.size() << ","
                     << nb_spurious << ","
                     << nb_iter_biased << ","
                     << nb_iter_free << ","
                     << (all_recovered_before_spurious ? 1 : 0) << ","
                     << recovered_pattern_idx << std::endl;

        query_iter++;
    }

    results_file.close();

    // Update parameters file with final metrics
    params["all_recovered_before_spurious"] = all_recovered_before_spurious ? 1.0 : 0.0;
    params["first_iter_all_found"] = static_cast<double>(first_iter_all_found);
    createParameterFile(sim_dir, params);

    std::cout << "Sim " << sim_number << ": found " << found_pattern_indices.size()
              << "/" << patterns.size() << " patterns, "
              << nb_spurious << " spurious";
    if (all_recovered_before_spurious) {
        std::cout << " (all recovered before spurious at iter " << first_iter_all_found << ")";
    }
    std::cout << std::endl;
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
                         inherited_params, sleep_params, config.output_dir, net_path);
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
