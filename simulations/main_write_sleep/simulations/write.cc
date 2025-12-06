/**
 * Generic network training simulation.
 *
 * Reads configuration from JSON file, trains networks with gradient descent,
 * and saves weights/connectivity/patterns for later sleep simulations.
 *
 * Usage: ./bin/write <config.json>
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
#include <cmath>
#include <ctime>

namespace fs = std::filesystem;

// Training function for a single simulation
void run_training(
    int sim_number,
    const std::unordered_map<std::string, double>& params,
    const std::vector<std::vector<bool>>* shared_patterns,  // nullptr for native mode
    bool native_mode,
    const std::string& output_dir
) {
    std::cout << "Training sim " << sim_number << std::endl;

    // Determine patterns: generate or use shared
    std::vector<std::vector<bool>> patterns;
    if (native_mode) {
        // Seed random per-thread for unique patterns
        srand(static_cast<unsigned>(time(nullptr)) ^ (sim_number * 1000));

        int network_size = static_cast<int>(params.at("network_size"));
        int num_patterns = static_cast<int>(params.at("num_patterns"));
        double sparsity = params.at("sparsity");
        double rho = params.at("rho");

        bool use_old_patterns = params.count("use_old_patterns") ? params.at("use_old_patterns") > 0.5 : false;
        patterns = generatePatterns(num_patterns, network_size, sparsity, rho, use_old_patterns);
        std::cout << "Sim " << sim_number << ": Generated " << patterns.size()
                  << " patterns (N=" << network_size << ", sparsity=" << sparsity
                  << ", rho=" << rho << ")" << std::endl;
    } else {
        patterns = *shared_patterns;
    }

    // Extract parameters with defaults
    int network_size = static_cast<int>(params.at("network_size"));
    int nb_winners = static_cast<int>(params.count("nb_winners") ? params.at("nb_winners") : network_size / 2);
    double leak = params.count("leak") ? params.at("leak") : 1.0;
    double drive_target = params.count("drive_target") ? params.at("drive_target") : 6.0;
    double learning_rate = params.count("learning_rate") ? params.at("learning_rate") : 0.0001;
    double epsilon = params.count("epsilon_learning") ? params.at("epsilon_learning") : learning_rate / 1000000;
    double distance_noise = params.count("distance_noise_level") ? params.at("distance_noise_level") : 0.0;
    double momentum_coef = params.count("momentum_coef") ? params.at("momentum_coef") : 0.9;
    int max_iter = params.count("max_iter") ? static_cast<int>(params.at("max_iter")) : static_cast<int>(10.0 / learning_rate);

    // Create simulation output directory
    std::string sim_dir = output_dir + "/sim_nb_" + std::to_string(sim_number);
    fs::create_directories(sim_dir);

    // Build fully connected network
    std::vector<std::vector<bool>> connectivity(network_size, std::vector<bool>(network_size, false));
    for (int i = 0; i < network_size; i++) {
        for (int j = 0; j < network_size; j++) {
            if (i != j) connectivity[i][j] = true;
        }
    }

    bool symmetric_transfer = params.count("symmetric_transfer") ? params.at("symmetric_transfer") > 0.5 : false;

    Network net(connectivity, network_size, leak, symmetric_transfer);

    // Convert patterns to target drives
    std::vector<std::vector<double>> target_drives = patterns_as_states_with_distance_noise(
        drive_target, patterns, distance_noise, net
    );

    // Initialize velocity matrices for momentum
    std::vector<std::vector<double>> velocity_matrix(network_size, std::vector<double>(network_size, 0.0));
    std::vector<double> velocity_bias(network_size, 0.0);
    std::vector<double> drive_errors(network_size, 0.0);

    // Training loop
    double max_error = 1000.0;
    int iter = 0;

    while (max_error > epsilon && iter < max_iter) {
        for (size_t p = 0; p < patterns.size(); p++) {
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

    std::cout << "Sim " << sim_number << " converged in " << iter << " iterations" << std::endl;

    // Save outputs
    writeMatrixToFile(net.weight_matrix, sim_dir + "/weights.data");
    writeBoolMatrixToFile(net.connectivity_matrix, sim_dir + "/connectivity.data");

    std::ofstream patterns_file(sim_dir + "/patterns.data");
    for (const auto& pattern : patterns) {
        writeBoolToCSV(patterns_file, pattern);
    }
    patterns_file.close();

    // Save parameters (including computed ones)
    std::unordered_map<std::string, double> saved_params = params;
    saved_params["num_patterns"] = static_cast<double>(patterns.size());
    saved_params["nb_winners"] = static_cast<double>(nb_winners);
    saved_params["symmetric_transfer"] = symmetric_transfer ? 1.0 : 0.0;
    saved_params["training_iterations"] = static_cast<double>(iter);
    createParameterFile(sim_dir, saved_params);
}


int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <config.json>" << std::endl;
        return 1;
    }

    // Load configuration
    JsonConfig config = JsonConfig::load(argv[1]);

    if (config.type != "write") {
        std::cerr << "Error: Config type must be 'write', got '" << config.type << "'" << std::endl;
        return 1;
    }

    // Determine pattern generation mode
    std::vector<std::vector<bool>> shared_patterns;

    if (!config.native_pattern_generation) {
        // FILE MODE: Load patterns from file (existing behavior)
        if (config.patterns_file.empty()) {
            std::cerr << "Error: patterns_file required when native_pattern_generation is false" << std::endl;
            return 1;
        }
        shared_patterns = loadPatterns(config.patterns_file);
        if (shared_patterns.empty()) {
            std::cerr << "Error: No patterns loaded from " << config.patterns_file << std::endl;
            return 1;
        }
        std::cout << "Loaded " << shared_patterns.size() << " patterns of size "
                  << shared_patterns[0].size() << std::endl;
    } else {
        // NATIVE MODE: Validate required parameters exist
        std::vector<std::string> required = {"network_size", "num_patterns", "sparsity", "rho"};
        for (const auto& param : required) {
            bool found = config.base_params.count(param) || config.varying_params.count(param);
            if (!found) {
                std::cerr << "Error: '" << param << "' required for native pattern generation" << std::endl;
                return 1;
            }
        }
        std::cout << "Native pattern generation enabled" << std::endl;
    }

    // Create output directory
    if (fs::exists(config.output_dir)) {
        fs::remove_all(config.output_dir);
    }
    fs::create_directories(config.output_dir);

    // Generate all parameter combinations
    auto combinations = config.generateCombinations();
    std::cout << "Running " << combinations.size() << " training simulations" << std::endl;

    // Thread pool execution
    const int max_threads = std::min(20, static_cast<int>(std::thread::hardware_concurrency()));
    int active_threads = 0;
    std::mutex mtx;
    std::condition_variable cv;
    std::vector<std::thread> threads;

    // Prepare pointer for shared patterns (nullptr in native mode)
    const std::vector<std::vector<bool>>* patterns_ptr =
        config.native_pattern_generation ? nullptr : &shared_patterns;
    bool native_mode = config.native_pattern_generation;

    for (size_t sim_number = 0; sim_number < combinations.size(); ++sim_number) {
        {
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait(lock, [&] { return active_threads < max_threads; });
            ++active_threads;
        }

        threads.emplace_back([=, &mtx, &cv, &active_threads] {
            run_training(sim_number, combinations[sim_number], patterns_ptr,
                         native_mode, config.output_dir);
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

    std::cout << "Training complete. Results in: " << config.output_dir << std::endl;
    return 0;
}
