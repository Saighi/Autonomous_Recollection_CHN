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

using namespace std;

namespace fs = std::filesystem;

void run_sleep(int sim_number, std::vector<std::vector<double>> net_weights, std::vector<std::vector<bool>> net_connectivity, const unordered_map<string, double> parameters, const string foldername_results, vector<vector<bool>> patterns)
{
    // --- Inherited
    int nb_winners = parameters.at("nb_winners");
    double learning_rate = parameters.at("learning_rate");
    double drive_target = parameters.at("drive_target");
    double ratio_flip_writing = parameters.at("ratio_flip_writing");
    int network_size = static_cast<int>(parameters.at("network_size"));
    double noise_level = parameters.at("noise_level");
    double leak = parameters.at("leak");
    double delta = parameters.at("delta");
    int num_patterns = static_cast<int>(parameters.at("num_patterns"));
    int nb_repetition = static_cast<int>(parameters.at("nb_repetition"));
    // --- not inherited
    double ratio_flips_querying = parameters.at("ratio_flips_querying");
    int nb_converge = static_cast<int>(parameters.at("nb_converge"));
    float noise_stddev = parameters.at("noise_stddev");
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
        // show_vector_bool_grid(patterns[i], col_with);
    }
    file.close();

    createParameterFile(sim_data_foldername, parameters);
    // SLEEPING SIMULATIONS

    string result_file_name_retrieval;
    vector<bool> winning_units;

    result_file_name_retrieval = sim_data_foldername + "/results.data";
    std::ofstream result_file_retrieval(result_file_name_retrieval, std::ios::trunc);
    std::cout << "QUERYING PHASE" << std::endl;
    double nb_found_patterns = 0;
    vector<double> state_pat;
    int nb_iter;
    for(int i = 0; i< nb_repetition; i++)
    {
        for (size_t j = 0; j < patterns.size(); j++)
        {
            // std::cout << "NEW ITER" << std::endl;
            state_pat= pattern_as_states(net.transfer(drive_target), net.transfer(-drive_target), generateNoisyBalancedPattern(patterns[j], int(network_size * ratio_flips_querying)));

            net.set_state(state_pat);
            // std::cout << "Initial random state:" << std::endl;
            // show_state_grid(net, 3); // Show initial state

            // Let the network converge
            run_net_sim_noisy(net, nb_converge, delta, 0.0, 0.01); // Using utility function for noisy iterations
            // nb_iter = run_until_converg_net_sim_noisy(net, 0.001, delta, 0.0, noise_stddev); // Using utility function for noisy iterations
            std::cout << nb_iter << std::endl;
            winning_units = assignBoolToTopNValues(net.activity_list, nb_winners);
            // Check if the output vector is in the target set and hasn't been counted yet
            if (winning_units==patterns[j])
            {
                nb_found_patterns += 1.;
            }
        }
    }
    double ratio_found_patterns = nb_found_patterns/(nb_repetition*patterns.size());

    result_file_retrieval <<"ratio_found_patterns="<< ratio_found_patterns << std::endl;
    // The number of unique vectors found
    // std::cout << "ratio_found_patterns: " << ratio_found_patterns << " nb_patterns : " << num_patterns << std::endl;
    // std::cout << "mean nb iter convergence : "<<nb_iter/(nb_repetition*patterns.size()) << std::endl;

    result_file_retrieval.close();
    
}

int main(int argc, char **argv)
{
    string sim_name = "quere_many_drives_relative_num_patterns_new_writing_faster_convergence";
    string inputs_name = "write_various_drive_target";
    // string inputs_name = "write_parameter_many_nb_iter_learning";
    string foldername_results = "../../../data/all_data_splited/query_simulations/" + sim_name;
    fs::path foldername_inputs = "../../../data/all_data_splited/trained_networks_fast/" + inputs_name;
    // Create directory if it doesn't exist
    std::cout << "what !!" << std::endl;
    if (!fs::exists(foldername_results))
    {
        if (!fs::create_directory(foldername_results))
        {
            std::cerr << "Error creating directory: " << foldername_results << std::endl;
            return 1;
        }
    }
    double nb_repetition = 10;
    vector<double> ratios_flips_querying = {0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5};
    unordered_map<string, vector<double>> varying_params = {
        {"ratio_flips_querying",ratios_flips_querying},
        {"nb_repetition", {nb_repetition}},
        {"nb_converge", {400}},
        {"noise_stddev",{0.005}}};

    unordered_map<string, double> inherited_params;
    vector<vector<bool>> patterns;
    vector<vector<double>> net_weights;
    vector<vector<bool>> net_connectivity;
    string patterns_file_name;
    vector<unordered_map<string, double>> combinations = generateCombinations(varying_params);
    unordered_map<string,double> fused_parameters;
    int batch = 0;
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

    const int max_threads = 30; // Set the maximum number of concurrent threads
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

    return 0;
}
