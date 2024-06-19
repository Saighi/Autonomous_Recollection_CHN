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

using namespace std;

namespace fs = std::filesystem;

void run_simulation(int sim_number, unordered_map<string, double> parameters, const string foldername_results)
{
    // Learning constants
    double target_up_rate = parameters.at("target_up_rate");
    double target_down_rate = parameters.at("target_down_rate");
    double learning_rate = parameters.at("learning_rate");
    int network_size = static_cast<int>(parameters.at("network_size"));
    // int nb_winners =static_cast<int>(parameters.at("nb_winners"));
    int nb_winners = max(2,static_cast<int>(parameters.at("relative_nb_winner")*network_size)); // number of 1's neurons
    parameters["nb_winners"] = static_cast<double>(nb_winners);
    double noise_level = parameters.at("noise_level");
    double leak = parameters.at("leak");
    double delta = parameters.at("delta");
    int nb_iter_learning = static_cast<int>(parameters.at("nb_iter_learning"));
    int nb_iter_sim = static_cast<int>(parameters.at("nb_iter_sim"));
    int num_flip = static_cast<int>(parameters.at("num_flip"));
    int num_patterns = max(1,static_cast<int>(parameters.at("relative_num_patterns")*network_size));
    parameters["num_patterns"] = static_cast<double>(num_patterns);
    int col_with = sqrt(network_size);
    string sim_data_foldername;
    string patterns_file_name;
    string result_file_name;
    string weights_file_name;
    string connectivity_file_name;
    vector<vector<bool>> initial_patterns;
    vector<vector<double>> initial_patterns_state_list;
    vector<bool> winning_units;

    sim_data_foldername = foldername_results + "/sim_nb_" + to_string(sim_number);

    // Create directory if it doesn't exist
    if (!fs::exists(sim_data_foldername))
    {
        if (!fs::create_directory(sim_data_foldername))
        {
            std::cerr << "Error creating directory: " << sim_data_foldername << std::endl;
            return;
        }
    }

    patterns_file_name = sim_data_foldername + "/patterns.data";
    std::ofstream file(patterns_file_name, std::ios::trunc);
    initial_patterns = generatePatterns(num_patterns, network_size, nb_winners, noise_level);

    for (int i = 0; i < num_patterns; i++)
    {
        writeBoolToCSV(file, initial_patterns[i]);
        // show_vector_bool_grid(initial_patterns[i], col_with);
    }
    file.close();

    createParameterFile(sim_data_foldername, parameters);

    // Loading training data
    initial_patterns = loadPatterns(patterns_file_name);
    initial_patterns_state_list = patterns_as_states(target_up_rate, target_down_rate, initial_patterns);

    // Build Fully connected network
    vector<vector<bool>> connectivity_matrix(network_size, vector<bool>(network_size, false));
    for (int i = 0; i < network_size; i++)
    {
        for (int j = 0; j < network_size; j++)
        {
            if (i != j)
            {
                connectivity_matrix[i][j] = true;
            }
        }
    }

    Network net = Network(connectivity_matrix, network_size, leak);

    // Training loop
    std::cout << "WRITING ATTRACTORS" << std::endl;
    for (int i = 0; i < nb_iter_learning; i++)
    {
        for (int j = 0; j < num_patterns; j++)
        {
            net.set_state(initial_patterns_state_list[j]);
            run_net_sim(net, nb_iter_sim, delta);
            net.reinforce_attractor(initial_patterns_state_list[j], learning_rate);
        }
    }

    // Querying
    std::cout << "Querying initial memories" << std::endl;
    vector<double> noisy_pattern;
    int succes = 0 ;
    for (int i = 0; i < num_patterns; i++)
    {
        noisy_pattern =pattern_as_states(target_up_rate,target_down_rate,generateNoisyBalancedPattern(initial_patterns[i],num_flip));
        net.set_state(noisy_pattern);
        run_net_sim_noisy(net, nb_iter_sim, delta, 0.0, 0.01);
        winning_units = assignBoolToTopNValues(net.activity_list, nb_winners);
        if (comparestates(winning_units,initial_patterns[i])){
            succes+=1;
        }
    }

    // The number of unique vectors found
    std::cout << "Number of vectors found: " << succes << " nb_patterns : " << num_patterns << " beta : " << "nb_winers : " << nb_winners << " nb_flip : " <<num_flip<<std::endl;

    result_file_name = sim_data_foldername + "/results.data";
    std::ofstream result_file(result_file_name, std::ios::trunc);
    result_file << "nb_found_patterns=" << succes;
    result_file.close();

    weights_file_name = sim_data_foldername + "/weights.data";
    writeMatrixToFile(net.weight_matrix, weights_file_name);

    connectivity_file_name = sim_data_foldername + "/connectivity.data";
    writeBoolMatrixToFile(net.connectivity_matrix, connectivity_file_name);
}

int main(int argc, char **argv)
{
    string sim_name = "write_parameter_test_0.25_num_pattern";
    string foldername_results = "../../all_data_splited/trained_networks/" + sim_name;

    // Create directory if it doesn't exist
    if (!fs::exists(foldername_results))
    {
        if (!fs::create_directory(foldername_results))
        {
            std::cerr << "Error creating directory: " << foldername_results << std::endl;
            return 1;
        }
    }
    vector<double> all_relative_num_patterns = linspace(0.1, 0.6,10);
    vector<double> network_sizes = {15,20,25,30,40,45,50,55,60,65};
    unordered_map<string, vector<double>> varying_params = {
        {"num_flip", {0}},
        {"relative_num_patterns", all_relative_num_patterns},
        {"target_up_rate", {0.95}},
        {"target_down_rate", {0.05}},
        {"learning_rate", {0.01}},  
        {"network_size", network_sizes},
        // {"nb_winners", {5,7,9,11,13}},
        {"relative_nb_winner", {0.5}},
        {"noise_level", {0.3}},
        {"leak", {1.3}},
        {"delta", {0.5}},
        {"nb_iter_learning", {2400}},
        {"nb_iter_sim", {250}}};


    vector<unordered_map<string, double>> combinations = generateCombinations(varying_params);
    // std::cout << combinations.size() << std::endl;
    // for(const auto &stuff: combinations){
    //     std::cout << "new params" << std::endl;
    //     for(const auto &params: stuff){
    //         std::cout << params.first << std::endl;
    //         std::cout << params.second << std::endl;
            
    //     }
    // }
    vector<thread> threads;

    for (int sim_number = 0; sim_number < combinations.size(); ++sim_number)
    {
        threads.emplace_back(run_simulation, sim_number, combinations[sim_number], foldername_results);
    }

    for (auto &t : threads)
    {
        t.join();
    }

    return 0;
}
