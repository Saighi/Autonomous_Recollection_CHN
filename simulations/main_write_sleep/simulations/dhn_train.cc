/**
 * Discrete Hopfield Network training simulation.
 *
 * Trains DHN networks using Hebbian or Storkey learning rules.
 * Supports parallelized parameter sweeps via JSON configuration.
 *
 * Learning Rules:
 * - learning_rule = 0: Hebbian (outer product rule)
 * - learning_rule = 1: Storkey (local field correction, higher capacity)
 *
 * Usage: ./bin/dhn_train <config.json>
 *
 * Config format:
 * {
 *   "type": "dhn_train",
 *   "output_dir": "/path/to/output",
 *   "native_pattern_generation": true,
 *   "base_params": {
 *     "network_size": 100,
 *     "num_patterns": 10,
 *     "sparsity": 0.5,
 *     "rho": 0.5,
 *     "learning_rule": 0  // 0=Hebbian, 1=Storkey
 *   },
 *   "varying_params": {
 *     "network_size": [100, 200, 300],
 *     "learning_rule": [0, 1]
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
#include <algorithm>
#include <cmath>
#include <ctime>

namespace fs = std::filesystem;

/**
 * Training function for a single DHN simulation.
 */
void run_dhn_training(
    int sim_number,
    const std::unordered_map<std::string, double>& params,
    const std::string& output_dir
) {
    std::cout << "DHN training sim " << sim_number << std::endl;

    // Extract parameters
    int network_size = static_cast<int>(params.at("network_size"));
    int num_patterns = static_cast<int>(params.at("num_patterns"));
    double rho = params.count("rho") ? params.at("rho") : 0.5;
    int learning_rule = params.count("learning_rule") ? static_cast<int>(params.at("learning_rule")) : 0;

    // Seed random for reproducible pattern generation
    unsigned int seed_value;
    if (params.count("seed")) {
        seed_value = static_cast<unsigned int>(params.at("seed"));
    } else {
        seed_value = static_cast<unsigned int>(sim_number);
    }
    srand(seed_value);

    // Generate patterns
    std::vector<std::vector<bool>> bool_patterns;
    PatternMetadata generated_metadata;
    bool has_metadata = false;

    bool use_heterogeneous = params.count("use_heterogeneous_sparsity") &&
                              params.at("use_heterogeneous_sparsity") > 0.5;

    if (use_heterogeneous) {
        double mean_sparsity = params.count("mean_sparsity") ? params.at("mean_sparsity") : 0.5;
        double sparsity_width = params.count("sparsity_width") ? params.at("sparsity_width") : 0.2;

        auto [gen_patterns, metadata] = generatePatternsHeterogeneous(
            num_patterns, network_size, mean_sparsity, sparsity_width, rho);
        bool_patterns = std::move(gen_patterns);
        generated_metadata = std::move(metadata);
        has_metadata = true;
    } else {
        double sparsity = params.count("sparsity") ? params.at("sparsity") : 0.5;
        bool use_old_patterns = params.count("use_old_patterns") ? params.at("use_old_patterns") > 0.5 : false;
        bool_patterns = generatePatterns(num_patterns, network_size, sparsity, rho, use_old_patterns);
    }

    // Convert bool patterns to {-1, +1} encoding for DHN
    std::vector<std::vector<double>> patterns;
    patterns.reserve(bool_patterns.size());
    for (const auto& bool_pat : bool_patterns) {
        std::vector<double> pat(network_size);
        for (int i = 0; i < network_size; ++i) {
            pat[i] = bool_pat[i] ? 1.0 : -1.0;
        }
        patterns.push_back(std::move(pat));
    }

    // Create simulation output directory
    std::string sim_dir = output_dir + "/sim_nb_" + std::to_string(sim_number);
    fs::create_directories(sim_dir);

    // Create and train network
    DiscreteHopfield net(network_size);

    std::string learning_rule_name;
    if (learning_rule == 0) {
        net.trainHebbian(patterns);
        learning_rule_name = "hebbian";
    } else if (learning_rule == 1) {
        net.trainStorkeyAVX(patterns);
        learning_rule_name = "storkey";
    } else {
        std::cerr << "Unknown learning rule: " << learning_rule << std::endl;
        net.trainHebbian(patterns);
        learning_rule_name = "hebbian";
    }

    std::cout << "Sim " << sim_number << ": Trained " << patterns.size()
              << " patterns (N=" << network_size << ", rule=" << learning_rule_name
              << ", rho=" << rho << ")" << std::endl;

    // Save weights
    writeMatrixToFile(net.weight_matrix, sim_dir + "/weights.data");

    // Save connectivity (fully connected except diagonal)
    std::vector<std::vector<bool>> connectivity(network_size, std::vector<bool>(network_size, false));
    for (int i = 0; i < network_size; ++i) {
        for (int j = 0; j < network_size; ++j) {
            if (i != j) connectivity[i][j] = true;
        }
    }
    writeBoolMatrixToFile(connectivity, sim_dir + "/connectivity.data");

    // Save patterns (as bool, consistent with CHN format)
    std::ofstream patterns_file(sim_dir + "/patterns.data");
    for (const auto& pat : bool_patterns) {
        writeBoolToCSV(patterns_file, pat);
    }
    patterns_file.close();

    // Save pattern metadata if heterogeneous
    if (has_metadata) {
        writePatternMetadata(generated_metadata, sim_dir + "/pattern_metadata.json");
    }

    // Save parameters
    std::unordered_map<std::string, double> saved_params = params;
    saved_params["num_patterns"] = static_cast<double>(patterns.size());
    saved_params["learning_rule"] = static_cast<double>(learning_rule);
    createParameterFile(sim_dir, saved_params);
}


int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <config.json>" << std::endl;
        return 1;
    }

    // Load configuration
    JsonConfig config = JsonConfig::load(argv[1]);

    if (config.type != "dhn_train") {
        std::cerr << "Error: Config type must be 'dhn_train', got '" << config.type << "'" << std::endl;
        return 1;
    }

    // Validate required parameters
    std::vector<std::string> required = {"network_size", "num_patterns"};
    bool use_heterogeneous = config.base_params.count("use_heterogeneous_sparsity") &&
                              config.base_params.at("use_heterogeneous_sparsity") > 0.5;

    if (!use_heterogeneous) {
        // Fixed sparsity mode needs sparsity parameter
        // (but it has a default, so not strictly required)
    }

    for (const auto& param : required) {
        bool found = config.base_params.count(param) || config.varying_params.count(param);
        if (!found) {
            std::cerr << "Error: '" << param << "' required for DHN training" << std::endl;
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
    std::cout << "Running " << combinations.size() << " DHN training simulations" << std::endl;

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
            run_dhn_training(sim_number, combinations[sim_number], config.output_dir);
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

    std::cout << "DHN training complete. Results in: " << config.output_dir << std::endl;
    return 0;
}
