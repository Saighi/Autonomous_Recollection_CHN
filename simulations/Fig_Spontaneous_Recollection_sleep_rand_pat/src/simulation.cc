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
#include "timer.hpp"

using namespace std;

namespace fs = std::filesystem;
const int IMAGE_HEIGHT = 20;
const int IMAGE_WIDTH = 16;

void run_sleep(int sim_number, std::vector<std::vector<double>> net_weights, std::vector<std::vector<bool>> net_connectivity, const unordered_map<string, double> parameters, const string foldername_results, vector<vector<bool>> patterns)
{
    // Learning constants
    double learning_rate = parameters.at("learning_rate");
    int network_size = static_cast<int>(parameters.at("network_size"));
    double leak = parameters.at("leak");
    double delta = parameters.at("delta");
    float beta = parameters.at("beta");
    int nb_iter = static_cast<int>(parameters.at("nb_iter"));
    float noise_stddev = parameters.at("noise_stddev");
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
    for (int i = 0; i < patterns.size(); i++)
    {
        writeBoolToCSV(file, patterns[i]);
        // show_vector_bool_grid(patterns[i], col_with);
    }
    file.close();

    createParameterFile(sim_data_foldername, parameters);
    // SLEEPING SIMULATIONS

    int iter_all_retrieved;
    int nb_spurious_patterns = 0;
    bool check = false;
    std::set<std::vector<bool>> foundVectors;
    vector<bool> is_found_patterns;

    vector<bool> winning_units;
    // std::cout << "SLEEP PHASE" << std::endl;
    int cpt = 0;
    while(cpt<nb_iter)
    {
        string result_file_name_sleep_depressed = sim_data_foldername + "/results_depressed_" + to_string(cpt) + ".data";
        std::ofstream result_file_sleep_depressed(result_file_name_sleep_depressed, std::ios::trunc);
        string result_file_name_sleep = sim_data_foldername + "/results_" + to_string(cpt) + ".data";
        std::ofstream result_file_sleep(result_file_name_sleep, std::ios::trunc);
        string inhib_matrix_file_name_sleep = sim_data_foldername + "/inhib_matrix_" + to_string(cpt) + ".data";

        writeMatrixToFile(net.inhib_matrix, inhib_matrix_file_name_sleep);
        // std::cout << "NEW ITER" << std::endl;
        net.set_state(vector<double>(network_size, 0.25));
        // std::cout << "Initial random state:" << std::endl;
        // show_state_grid(net, 3); // Show initial state

        // Let the network converge
        run_net_sim_noisy_depressed_convergence_check_save(net, delta/100, delta, 0.0, noise_stddev, result_file_sleep_depressed,100/delta); // Using utility function for noisy iterations and saving
        run_net_sim_noisy_convergence_check_save(net, delta/100, delta, 0.0, noise_stddev, result_file_sleep,100/delta); // Using utility function for noisy iterations
        result_file_sleep.close();
        winning_units = assignBoolThreshold(net.activity_list, 0.5);
        net.pot_inhib_bin(beta, winning_units); // works with 0.005
        // Check if the output vector is in the target set and hasn't been counted yet
        cpt+=1;
        show_vector_bool_grid(winning_units,IMAGE_HEIGHT);
        is_found_patterns.push_back(false);
        if (std::find(patterns.begin(), patterns.end(), winning_units) != patterns.end())
        {
            if (foundVectors.find(winning_units) == foundVectors.end())
            {
                // Count the vector
                foundVectors.insert(winning_units);
                is_found_patterns.back() = true;
            }
        }
        else
        {
            nb_spurious_patterns += 1;
        }
        if (foundVectors.size() == patterns.size() && !check)
        {
            iter_all_retrieved = is_found_patterns.size()-1;
            check = true;
        }

        cpt+=1;
        std::cout << "iter: "<< cpt<< "/" <<nb_iter<< std::endl;
    }
    std::cout <<"sim_number " << sim_number << std::endl;
    std::cout <<"nb_spurious :"<< nb_spurious_patterns <<" nb_found_patterns : "<< foundVectors.size() << " nb_patterns : " << patterns.size() << " beta : " <<" nb_flip : " << " Network size: " << network_size << std::endl;
    
}

int main(int argc, char **argv)
{   
    Timer timer;
    timer.start();
    string sim_name = "Fig_Spontaneous_Recollection_rand_pat";
    string inputs_name = "Fig_Spontaneous_Recollection_rand_pat";
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
    // vector<double> repetitions = {0};
    
    unordered_map<string, vector<double>> varying_params = {
        {"beta", {0.001} },
        {"nb_iter",{10}},
        {"noise_stddev",{0}}};   

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

    const int max_threads = 10; // Set the maximum number of concurrent threads
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
    timer.stop();
    timer.print_elapsed();
    return 0;
}