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
    double epsilon_learning=0.1;
    double drive_target = parameters.at("drive_target");
    double learning_rate = parameters.at("learning_rate");
    int network_size = static_cast<int>(parameters.at("network_size"));
    // int nb_winners =static_cast<int>(parameters.at("nb_winners"));
    int nb_winners = max(2,static_cast<int>(parameters.at("relative_nb_winner")*network_size)); // number of 1's neurons
    parameters["nb_winners"] = static_cast<double>(nb_winners);
    double noise_level = parameters.at("noise_level");
    double leak = parameters.at("leak");
    double steepness = parameters.at("steepness");
    double delta = parameters.at("delta");
    double ratio_flip_writing = parameters.at("ratio_flip_writing");
    int num_patterns = max(1,static_cast<int>(parameters.at("relative_num_patterns")*network_size));
    parameters["num_patterns"] = static_cast<double>(num_patterns);
    int col_with = sqrt(network_size);
    string sim_data_foldername;
    string patterns_file_name;
    string result_file_name;
    string weights_file_name;
    string connectivity_file_name;
    vector<vector<bool>> initial_patterns;
    vector<vector<double>> initial_patterns_rates;
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

    Network net = Network(connectivity_matrix, network_size, leak, steepness);
    // Loading training data
    initial_patterns = loadPatterns(patterns_file_name);
    initial_patterns_rates = patterns_as_states(net.transfer(drive_target), net.transfer(-drive_target), initial_patterns);
    vector<double> drive_errors;
    drive_errors.resize(network_size,0.0);
    double sum_errors;
    bool stop_learning = false;
    int cpt =0;
    // Training loop
    std::cout << "WRITING ATTRACTORS" << std::endl;
    while(!stop_learning)
    {
        sum_errors=0.0;
        for (int j = 0; j < num_patterns; j++)
        {
            net.derivative_gradient_descent(initial_patterns[j],initial_patterns_rates[j],drive_target,learning_rate, leak, drive_errors);
            sum_errors+=std::accumulate(drive_errors.begin(),drive_errors.end(),0.0);
        }
        if(abs(sum_errors)<epsilon_learning){
            stop_learning=true;
        }
        if(cpt==10000){
            stop_learning=true;
        }
        cpt+=1;
    }
    std::cout << "nombre d'iterations" << std::endl;
    std::cout << cpt << std::endl;
    // Querying
    std::cout << "Querying initial memories" << std::endl;
    vector<double> noisy_pattern;
    int succes = 0 ;
    // double strength_drive = 0.1;
    for (int i = 0; i < num_patterns; i++)
    {
        //TODO - change the pattern_as_states and link the target drive not magic number
        noisy_pattern=pattern_as_states(net.transfer(drive_target),net.transfer(-drive_target),generateNoisyBalancedPattern(initial_patterns[i],int(network_size*ratio_flip_writing)));
        net.set_state(noisy_pattern);
        // run_net_sim_query_drive(net, noisy_pattern, strength_drive, 1200, delta);
        // run_net_sim_noisy(net,500, delta,0.0,0.01);
        run_net_sim(net,100, delta);
        winning_units = assignBoolToTopNValues(net.activity_list, nb_winners);
        if (comparestates(winning_units,initial_patterns[i])){
            succes+=1;
        }
    }

    // The number of unique vectors found

    std::cout << "sim number: " <<sim_number<< std::endl;
    std::cout << "ratio flip querying: " << ratio_flip_writing<<std::endl;
    std::cout << "Number of vectors found: " << succes << " nb_patterns : " << num_patterns << " beta : " << "nb_winers : " << nb_winners << " nb_flip : " <<int(network_size*ratio_flip_writing)<<std::endl;

    // result_file_name = sim_data_foldername + "/results.data";
    // std::ofstream result_file(result_file_name, std::ios::trunc);
    // result_file << "nb_found_patterns=" << succes;
    // result_file.close();

    // weights_file_name = sim_data_foldername + "/weights.data";
    // writeMatrixToFile(net.weight_matrix, weights_file_name);

    // connectivity_file_name = sim_data_foldername + "/connectivity.data";
    // writeBoolMatrixToFile(net.connectivity_matrix, connectivity_file_name);
}

int main(int argc, char **argv)
{
    // string sim_name = "write_net_sizes_relative_num_patterns";
    string sim_name = "write_modifying_transfer_to_discrete";
    string foldername_results = "../../../data/all_data_splited/trained_networks_fast/" + sim_name;

    // Create directory if it doesn't exist
    if (!fs::exists(foldername_results))
    {
        if (!fs::create_directory(foldername_results))
        {
            std::cerr << "Error creating directory: " << foldername_results << std::endl;
            return 1;
        }
    }
    vector<double> all_relative_num_patterns = linspace(0.1,0.4,3);
    // vector<double> all_relative_num_patterns = {0.5};
    // vector<double> network_sizes = {10,20,30,40,50,60,70,80,90,100};
    vector<double> drive_targets = {6};
    vector<double> network_sizes = {45};
    vector<double> ratios_flip_writing = {0.1,0.2,0.3};
    vector<double> leak = {0.5,1,2};
    vector<double> steepness = {0.5,1,2};
    // vector<double> repetitions = {0,1,2,3,4,5,6,7,8,9};
    unordered_map<string, vector<double>> varying_params = {
        // {"repetitions", repetitions},
        {"ratio_flip_writing",ratios_flip_writing},
        {"steepness", steepness},
        {"drive_target", drive_targets},
        {"drive_target", {5}},
        {"relative_num_patterns", all_relative_num_patterns},
        {"learning_rate", {0.01}}, // REMOVED-target rates
        {"network_size", network_sizes},
        {"relative_nb_winner", {0.5}},
        {"noise_level", {0.5}},
        {"leak", leak},
        {"delta", {0.1}}};

    vector<unordered_map<string, double>> combinations = generateCombinations(varying_params);
    for (int sim_number = 0; sim_number < combinations.size(); ++sim_number)
    {
        run_simulation(sim_number, combinations[sim_number], foldername_results);
    }
    // vector<thread> threads;

    // for (int sim_number = 0; sim_number < combinations.size(); ++sim_number)
    // {
    //     threads.emplace_back(run_simulation, sim_number, combinations[sim_number], foldername_results);
    // }

    // for (auto &t : threads)
    // {
    //     t.join();
    // }

    return 0;
}
