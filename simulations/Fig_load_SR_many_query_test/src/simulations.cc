#include "network.hpp"
#include "utils.hpp"
#include <numeric>
#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <filesystem>
#include <set>
#include <algorithm>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <cstdlib>

using namespace std;

namespace fs = std::filesystem;

void run_sleep(int sim_number, std::vector<std::vector<double>> net_weights, std::vector<std::vector<bool>> net_connectivity, const unordered_map<string, double> parameters, const string foldername_results, vector<vector<bool>> patterns)
{
    srand(sim_number);
    std::cout <<"sim_number :"<< sim_number<< std::endl;
    bool save_trajectories=false;
    if (parameters.at("save")){
        save_trajectories=true;
        std::cout << "is saving !" << std::endl;
    }
    // Inherited
    double drive_target = parameters.at("drive_target");
    double init_drive = parameters.at("init_drive");
    double learning_rate = parameters.at("learning_rate");
    int network_size = static_cast<int>(parameters.at("network_size"));
    int nb_winners = static_cast<int>(parameters.at("nb_winners")); // number of 1's neurons
    double leak = parameters.at("leak");
    double delta = parameters.at("delta");
    int num_patterns = static_cast<int>(parameters.at("num_patterns"));
    double ratio_flip_querying = parameters.at("ratio_flip_querying");

    // not Inherited
    bool noise = false;
    if(parameters.at("noise")==1){
        noise=true;
    }
    double stddev=parameters.at("stddev");
    int col_with = sqrt(network_size);

    Network net = Network(net_connectivity, network_size, leak);
    net.weight_matrix= net_weights;

    string sim_data_foldername = foldername_results + "/sim_nb_" + to_string(sim_number);

    // Create directory if it doesn't exist
    if (!fs::exists(sim_data_foldername))
    {
        if (!fs::create_directory(sim_data_foldername))
        {
            std::cerr << "Error creating directory: " << sim_data_foldername << std::endl;
            return;
        }
    }

    // Store the inherited patterns
    string patterns_file_name = sim_data_foldername + "/patterns.data";
    std::ofstream file(patterns_file_name, std::ios::trunc);
    for (int i = 0; i < num_patterns; i++)
    {
        writeBoolToCSV(file, patterns[i]);
        // show_vector_bool_grid(patterns[i], 10);
    }
    file.close();
    createParameterFile(sim_data_foldername, parameters);
    // SLEEPING SIMULATIONS

    string result_file_name;
    string result_file_name_retrieval;
    int iter_all_retrieved;
    int nb_spurious_patterns = 0;
    bool check = false;
    std::set<std::vector<bool>> foundVectors;
    vector<bool> winning_units;

    result_file_name_retrieval = sim_data_foldername + "/results.data";
    std::ofstream result_file_retrieval(result_file_name_retrieval, std::ios::trunc);
    // std::cout << "SLEEP PHASE" << std::endl;
    float sum_rates;

    std::cout << "Querying initial memories" << std::endl;
    vector<double> query_pattern;
    int succes = 0;
    for (int i = 0; i < num_patterns; i++) {
        // Build a query state from the binary pattern and add perturbations on
        // a subset of units
        query_pattern =
            pattern_as_states(net.transfer(drive_target),
                              net.transfer(-drive_target), patterns[i]);
        query_pattern = setToValueRandomElements(
            query_pattern, int(network_size * ratio_flip_querying), init_drive);

        net.set_state(query_pattern);
        run_net_sim(net, int(1.0 / delta), delta);
        winning_units = assignBoolToTopNValues(net.activity_list, nb_winners);
        if (comparestates(winning_units, patterns[i])) {
            succes += 1;
        }
    }

    // The number of patterns successfully recovered
    std::cout << "Number of vectors found: " << succes
              << " nb_patterns : " << num_patterns
              << " nb_winners : " << nb_winners
              << " nb_flip : " << int(network_size * ratio_flip_querying)
              << " Network size: " << network_size << std::endl;
    result_file_name = sim_data_foldername + "/results.data";
    std::ofstream result_file(result_file_name, std::ios::trunc);
    result_file << "nb_found_patterns,\n";
    result_file << (num_patterns > 0 ? (static_cast<double>(succes) /
                                        static_cast<double>(num_patterns))
                                     : 0.0)
                << ",";
    result_file.close();
}

int main(int argc, char **argv)
{
    // string sim_name = "Fig_load_SR_average_new_inh_plas_many_betta_larger_networks_2";
    // string inputs_name = "Fig_load_SR_average_new_inh_plas_many_betta_larger_networks_2";
    string sim_name = "Fig_load_SR_query_test";
    string inputs_name =
        "Fig_load_SR_average_new_inh_plas_big_simulations_2025_optimized";
    // string inputs_name = "write_parameter_many_nb_iter_learning";
    string foldername_results = "../../../data/all_data_splited/sleep_simulations/" + sim_name;
    fs::path foldername_inputs = "../../../data/all_data_splited/trained_networks_fast/" + inputs_name;
    // Create directory if it doesn't exist
    if (fs::exists(foldername_results))
    {
        fs::remove_all(foldername_results);
    }
    if (!fs::create_directory(foldername_results))
    {
        std::cerr << "Error creating directory: " << foldername_results << std::endl;
        return 1;
    }
    vector<double> ratio_flip_querying = {0.1,0.3,0.5,0.7,0.9};
    unordered_map<string, vector<double>> varying_params = {
        {"ratio_flip_querying", ratio_flip_querying},
        {"save", {0}},
        // {"beta", {0.00125}},
        {"delta", {0.01}},
        {"noise", {1}},
        {"stddev", {0.01}}};

    unordered_map<string, double> inherited_params;
    vector<vector<bool>> patterns;
    vector<vector<double>> net_weights;
    vector<vector<bool>> net_connectivity;
    string patterns_file_name;

    vector<unordered_map<string, double>> combinations = generateCombinations(varying_params);
    unordered_map<string,double> fused_parameters;
    vector<string> all_paths;
    // Check if the path exists and is a directory
    if (fs::exists(foldername_inputs) && fs::is_directory(foldername_inputs))
    {
        // Iterate over the directory entries
        for (const auto &entry : fs::directory_iterator(foldername_inputs))
        {
            // Check if the entry is a directory
            if (fs::is_directory(entry.path()))
            {
                all_paths.push_back(entry.path().generic_string());
           }
        }
    }
    const int max_threads = 20; // Set the maximum number of concurrent threads
    int active_threads = 0;
    std::mutex mtx;
    std::condition_variable cv;
    std::vector<std::thread> threads;
    int all_sim_number = 0;
    for (const auto &path : all_paths)
    {
        inherited_params = readParametersFile(path + "/parameters.data");
        net_weights = readMatrixFromFile(path + "/weights.data");
        net_connectivity = readBoolMatrixFromFile(path + "/connectivity.data");
        patterns_file_name = path + "/patterns.data";


        patterns = loadPatterns(patterns_file_name);

        for (int sim_number = 0; sim_number < combinations.size(); ++sim_number)
        {
            fused_parameters = fuseMaps(inherited_params, combinations[sim_number]);

            {
                std::unique_lock<std::mutex> lock(mtx);
                cv.wait(lock, [&]
                        { return active_threads < max_threads; });
                ++active_threads;
            }

            threads.emplace_back([=, &mtx, &cv, &active_threads]
                                 {
                run_sleep(all_sim_number, net_weights, net_connectivity, fused_parameters, foldername_results, patterns);
                {
                    std::lock_guard<std::mutex> lock(mtx);
                    --active_threads;
                }
                cv.notify_all(); });

            all_sim_number += 1;
        }
    }

    for (auto &t : threads)
    {
        if (t.joinable())
        {
            t.join();
        }
    }

    collectSimulationDataSeries(foldername_results);

    return 0;
}
