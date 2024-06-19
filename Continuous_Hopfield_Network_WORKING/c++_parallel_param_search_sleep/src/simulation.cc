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
    // Learning constants
    double target_up_rate = parameters.at("target_up_rate");
    double target_down_rate = parameters.at("target_down_rate");
    double learning_rate = parameters.at("learning_rate");
    int network_size = static_cast<int>(parameters.at("network_size"));
    int nb_winners = static_cast<int>(parameters.at("nb_winners")); // number of 1's neurons
    double noise_level = parameters.at("noise_level");
    double leak = parameters.at("leak");
    double delta = parameters.at("delta");
    int nb_iter_learning = static_cast<int>(parameters.at("nb_iter_learning"));
    int nb_iter_sim = static_cast<int>(parameters.at("nb_iter_sim"));
    float beta = parameters.at("beta");
    int num_patterns = static_cast<int>(parameters.at("num_patterns"));
    int nb_iter = static_cast<int>(parameters.at("nb_iter_mult") * num_patterns);
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

    string result_file_name_sleep;
    string result_file_name_retrieval;
    int iter_all_retrieved;
    int nb_spurious_patterns = 0;
    bool check = false;
    std::set<std::vector<bool>> foundVectors;
    vector<bool> winning_units;

    result_file_name_retrieval = sim_data_foldername + "/results.data";
    std::ofstream result_file_retrieval(result_file_name_retrieval, std::ios::trunc);
    std::cout << "SLEEP PHASE" << std::endl;
    for (int r = 0; r < nb_iter; r++)
    {
        result_file_name_sleep = sim_data_foldername + "/results_" + to_string(r) + ".data";
        std::ofstream result_file_sleep(result_file_name_sleep, std::ios::trunc);
        // std::cout << "NEW ITER" << std::endl;
        net.set_state(vector<double>(network_size, 0.5));
        // std::cout << "Initial random state:" << std::endl;
        // show_state_grid(net, 3); // Show initial state

        // Let the network converge
        run_net_sim_noisy_depressed_save(net, nb_converge, delta, 0.0, noise_stddev, result_file_sleep); // Using utility function for noisy iterations and saving
        run_net_sim_noisy_save(net, nb_converge, delta, 0.0, 0.01, result_file_sleep); // Using utility function for noisy iterations
        result_file_sleep.close();
        winning_units = assignBoolToTopNValues(net.activity_list, nb_winners);
        net.pot_inhib_bin(beta, winning_units); // works with 0.005
        // Check if the output vector is in the target set and hasn't been counted yet
        if (std::find(patterns.begin(), patterns.end(), winning_units) != patterns.end())
        {
            if (foundVectors.find(winning_units) == foundVectors.end())
            {
                // Count the vector
                foundVectors.insert(winning_units);
            }
        }
        else
        {
            nb_spurious_patterns += 1;
        }
        if (foundVectors.size() == patterns.size() && !check)
        {
            iter_all_retrieved = r;
            check = true;
        }

        result_file_retrieval << "iter=" << r << "\n";
        result_file_retrieval << "nb_found_patterns=" << foundVectors.size() << "\n";
        result_file_retrieval << "nb_spurious_patterns=" << nb_spurious_patterns << "\n";
    }

    result_file_retrieval << "general results :" << std::endl;
    result_file_retrieval << "iter_all_retrieved=" << iter_all_retrieved << std::endl;
    // The number of unique vectors found
    std::cout << "Number of unique vectors found: " << foundVectors.size() << " nb_patterns : " << num_patterns << " beta : " << beta << " nb_spurious : " << nb_spurious_patterns << std::endl
              << " nb_winers : " << nb_winners << std::endl;

    result_file_retrieval.close();
    
}

int main(int argc, char **argv)
{
    string sim_name = "sleep_parameter_test";
    string inputs_name = "write_parameter_test";
    string foldername_results = "../../all_data_splited/sleep_simulations/" + sim_name;
    fs::path foldername_inputs = "../../all_data_splited/trained_networks/" + inputs_name;
    // Create directory if it doesn't exist
    if (!fs::exists(foldername_results))
    {
        if (!fs::create_directory(foldername_results))
        {
            std::cerr << "Error creating directory: " << foldername_results << std::endl;
            return 1;
        }
    }
    vector<double> betas = linspace(0.001,0.1, 10);
    unordered_map<string, vector<double>> varying_params = {
        {"beta", betas },
        {"nb_iter_mult", {5}},
        {"nb_converge", {800}},
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

    return 0;
}
