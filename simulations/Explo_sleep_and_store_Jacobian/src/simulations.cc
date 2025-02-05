#include <algorithm>
#include <condition_variable>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <numeric>
#include <random>
#include <set>
#include <thread>
#include <unordered_map>
#include <vector>

#include "network.hpp"
#include "utils.hpp"

using namespace std;

namespace fs = std::filesystem;

void run_sleep(int sim_number, std::vector<std::vector<double>> net_weights,
               std::vector<std::vector<bool>> net_connectivity,
               const unordered_map<string, double> parameters,
               const string foldername_results, vector<vector<bool>> patterns) {
    srand(sim_number);
    std::cout << "sim_number :" << sim_number << std::endl;
    // Inherited
    int network_size = static_cast<int>(parameters.at("network_size"));
    int nb_winners =
        static_cast<int>(parameters.at("nb_winners"));  // number of 1's neurons
    double leak = parameters.at("leak");
    double delta = parameters.at("delta");
    double init_drive = parameters.at("init_drive");
    int num_patterns = static_cast<int>(parameters.at("num_patterns"));

    Network net = Network(net_connectivity, network_size, leak);
    net.weight_matrix = net_weights;

    string sim_data_foldername =
        foldername_results + "/sim_nb_" + to_string(sim_number);

    // Create directory if it doesn't exist
    if (!fs::exists(sim_data_foldername)) {
        if (!fs::create_directory(sim_data_foldername)) {
            std::cerr << "Error creating directory: " << sim_data_foldername
                      << std::endl;
            return;
        }
    }

    // Store the inherited patterns
    string patterns_file_name = sim_data_foldername + "/patterns.data";
    std::ofstream file(patterns_file_name, std::ios::trunc);
    for (int i = 0; i < num_patterns; i++) {
        writeBoolToCSV(file, patterns[i]);
        // show_vector_bool_grid(patterns[i], 10);
    }
    file.close();
    createParameterFile(sim_data_foldername, parameters);
    // SLEEPING SIMULATIONS

    string result_file_jacobian_name =
        sim_data_foldername + "/results_jacobian_1.data";
    std::ofstream result_file_jacobian(result_file_jacobian_name,
                                       std::ios::trunc);

    string result_file_trajectory_name =
        sim_data_foldername + "/results_trajectory_1.data";
    std::ofstream result_file_trajectory(result_file_trajectory_name,
                                         std::ios::trunc);

    // std::cout << "SLEEP PHASE" << std::endl;
    float sum_rates;

    // std::cout << "NEW ITER" << std::endl;
    net.set_state(vector<double>(network_size, init_drive));
    // std::cout << "Initial random state:" << std::endl;
    // show_state_grid(net, 3); // Show initial state
    SimulationConfig config;
    config.output = &result_file_trajectory;
    config.output_jacobian = &result_file_jacobian;
    config.delta = delta;
    config.epsilon = delta / 10000;
    config.depressed = true;
    config.save = true;
    config.save_jacobian = true;
    config.max_iter = 100 / delta;
    config.noise = false;
    config.patterns = patterns;
    int nb_iter_sim = 0;
    nb_iter_sim += run_net_sim_choice(net, config);

    result_file_trajectory.close();
    result_file_jacobian.close();

    std::cout << nb_iter_sim << std::endl;

    std::cout << "sim_number " << sim_number << std::endl;
}

int main(int argc, char **argv) {
    // string sim_name =
    // "Fig_load_SR_average_new_inh_plas_many_betta_larger_networks_2"; string
    // inputs_name =
    // "Fig_load_SR_average_new_inh_plas_many_betta_larger_networks_2";
    string sim_name = "explo_jacobian";
    string inputs_name = "explo_jacobian";
    // string inputs_name = "write_parameter_many_nb_iter_learning";
    string foldername_results =
        "../../../data/all_data_splited/sleep_simulations/" + sim_name;
    fs::path foldername_inputs =
        "../../../data/all_data_splited/trained_networks_fast/" + inputs_name;
    // Create directory if it doesn't exist
    if (fs::exists(foldername_results)) {
        fs::remove_all(foldername_results);
    }
    if (!fs::create_directory(foldername_results)) {
        std::cerr << "Error creating directory: " << foldername_results
                  << std::endl;
        return 1;
    }
    // vector<double> beta = linspace(0.00125/10, 0.00125, 3);
    // vector<double> beta = {0.00125};

    unordered_map<string, vector<double>> varying_params = {
        {"save", {0}},
        // {"beta", beta},
        {"delta", {0.01}},
        {"noise", {1}},
        {"stddev", {0.01}},
        {"nb_iter_mult_max", {20}}};

    unordered_map<string, double> inherited_params;
    vector<vector<bool>> patterns;
    vector<vector<double>> net_weights;
    vector<vector<bool>> net_connectivity;
    string patterns_file_name;

    vector<unordered_map<string, double>> combinations =
        generateCombinations(varying_params);
    unordered_map<string, double> fused_parameters;
    vector<string> all_paths;
    // Check if the path exists and is a directory
    if (fs::exists(foldername_inputs) && fs::is_directory(foldername_inputs)) {
        // Iterate over the directory entries
        for (const auto &entry : fs::directory_iterator(foldername_inputs)) {
            // Check if the entry is a directory
            if (fs::is_directory(entry.path())) {
                all_paths.push_back(entry.path().generic_string());
            }
        }
    }
    const int max_threads = 20;  // Set the maximum number of concurrent threads
    int active_threads = 0;
    std::mutex mtx;
    std::condition_variable cv;
    std::vector<std::thread> threads;
    int all_sim_number = 0;
    for (const auto &path : all_paths) {
        inherited_params = readParametersFile(path + "/parameters.data");
        net_weights = readMatrixFromFile(path + "/weights.data");
        net_connectivity = readBoolMatrixFromFile(path + "/connectivity.data");
        patterns_file_name = path + "/patterns.data";

        patterns = loadPatterns(patterns_file_name);

        for (int sim_number = 0; sim_number < combinations.size();
             ++sim_number) {
            fused_parameters =
                fuseMaps(inherited_params, combinations[sim_number]);

            {
                std::unique_lock<std::mutex> lock(mtx);
                cv.wait(lock, [&] { return active_threads < max_threads; });
                ++active_threads;
            }

            threads.emplace_back([=, &mtx, &cv, &active_threads] {
                run_sleep(all_sim_number, net_weights, net_connectivity,
                          fused_parameters, foldername_results, patterns);
                {
                    std::lock_guard<std::mutex> lock(mtx);
                    --active_threads;
                }
                cv.notify_all();
            });

            all_sim_number += 1;
        }
    }

    for (auto &t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }

    collectSimulationDataSeries(foldername_results);

    return 0;
}
