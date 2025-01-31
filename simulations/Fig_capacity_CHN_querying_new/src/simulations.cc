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
    // Inherited
    double drive_target = parameters.at("drive_target");
    double init_drive = parameters.at("init_drive");
    double learning_rate = parameters.at("learning_rate");
    int network_size = static_cast<int>(parameters.at("network_size"));
    int nb_winners = static_cast<int>(parameters.at("nb_winners")); // number of 1's neurons
    std::cout << nb_winners << std::endl;
    double noise_level = parameters.at("noise_level");
    double leak = parameters.at("leak");
    double delta = parameters.at("delta");
    int num_patterns = static_cast<int>(parameters.at("num_patterns"));
    // not Inherited
    int col_with = sqrt(network_size);
    double ratio_flips_querying = parameters.at("ratio_flips_querying");


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
    vector<double> query_pattern;
    double mean_nb_found_patterns;
    string result_file_name_sleep;

    for (size_t j = 0; j < patterns.size(); j++)
    {
        // std::cout << "NEW ITER" << std::endl;
        query_pattern=pattern_as_states(net.transfer(drive_target),net.transfer(-drive_target),patterns[j]);
        query_pattern=setToValueRandomElements(query_pattern, int(network_size*ratio_flips_querying), init_drive);
        net.set_state(query_pattern);
        // std::cout << "Initial random state:" << std::endl;
        // show_state_grid(net, 3); // Show initial state

        // Let the network converge
        SimulationConfig config;
        result_file_name_sleep = "no_save.data";
        std::ofstream result_file_sleep(result_file_name_sleep, std::ios::trunc);
        config.output=&result_file_sleep;
        config.delta = delta;
        config.epsilon = delta/1000;
        config.depressed = false;
        config.save = false;
        config.max_iter = 100/delta;
        run_net_sim_choice(net, config); 
        result_file_sleep.close();

        // nb_iter = run_until_converg_net_sim_noisy(net, 0.001, delta, 0.0, noise_stddev); // Using utility function for noisy iterations
        winning_units = assignBoolToTopNValues(net.activity_list, nb_winners);
        // Check if the output vector is in the target set and hasn't been counted yet
        if (winning_units==patterns[j])
        {
            nb_found_patterns += 1.;
        }
    }

    result_file_retrieval <<"nb_found_patterns="<< nb_found_patterns << std::endl;
    // The number of unique vectors found
    std::cout << "ratio random bits query: " << ratio_flips_querying <<" network size: " << network_size << " mean_found_patterns: " << nb_found_patterns << " nb_patterns : " << num_patterns << std::endl;
    // std::cout << "mean nb iter convergence : "<<nb_iter/(nb_repetition*patterns.size()) << std::endl;

    result_file_retrieval.close();
    std::cout << sim_number << std::endl;
    
}


int main(int argc, char **argv)
{
    string sim_name = "Fig_capacity_CHN_new";
    string inputs_name = "Fig_capacity_CHN_new";
    // string inputs_name = "write_parameter_many_nb_iter_learning";
    string foldername_results = "../../../data/all_data_splited/query_simulations/" + sim_name;
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

    vector<double> ratios_flips_querying = linspace(0.01,0.15,15);

    unordered_map<string, vector<double>> varying_params = {
        {"ratio_flips_querying",ratios_flips_querying},
        };


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
