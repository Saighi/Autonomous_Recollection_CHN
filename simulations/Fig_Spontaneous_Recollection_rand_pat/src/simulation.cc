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
#include <matio.h>

const int IMAGE_HEIGHT = 20;
const int IMAGE_WIDTH = 16;
using namespace std;

namespace fs = std::filesystem;


void run_simulation(int sim_number, unordered_map<string, double> parameters, const string foldername_results)
{
    // Learning constants
    int cpt=0;
    int num_patterns = parameters.at("num_patterns");
    double epsilon_learning=parameters.at("epsilon_learning");
    double drive_target = parameters.at("drive_target");
    double learning_rate = parameters.at("learning_rate");
    int network_size = parameters.at("network_size");
    int nb_winners;
    double leak = parameters.at("leak");
    double delta = parameters.at("delta");
    int col_with = IMAGE_HEIGHT;
    string sim_data_foldername;
    string result_file_name;
    string weights_file_name;
    string connectivity_file_name;
    vector<vector<bool>> initial_patterns;
    vector<vector<bool>> query_patterns;
    vector<vector<double>> initial_patterns_rates;
    vector<vector<double>> query_patterns_rates;
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

    std::string patterns_file_name = sim_data_foldername + "/patterns.data";
    std::ofstream file(patterns_file_name, std::ios::trunc);
    initial_patterns = generatePatterns(num_patterns, 20*16, (20*16)/4, 1);
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

    Network net = Network(connectivity_matrix, network_size, leak);
    // Loading training data
    initial_patterns_rates = patterns_as_states(net.transfer(drive_target), net.transfer(-drive_target), initial_patterns);
    // query_patterns_rates = patterns_as_states(net.transfer(drive_target), net.transfer(-drive_target), query_patterns);

    std::vector<double> drives_error(net.size,0.0);

    double max_error=1000;
    cpt=0;
    // Training loop
    std::cout << "WRITING ATTRACTORS" << std::endl;
    while (max_error > epsilon_learning && cpt <= 10000)
    {
        for (int j = 0; j < initial_patterns.size(); j++)
        {
            net.derivative_gradient_descent(initial_patterns[j],initial_patterns_rates[j],drive_target,learning_rate, leak, drives_error);
        }
        max_error = std::abs(*std::max_element(drives_error.begin(),drives_error.end()));
        cpt+=1;
        std::cout << cpt << std::endl;
    }
    std::cout << "nombre d'iterations" << std::endl;
    std::cout << cpt << std::endl;
    // Querying
    std::cout << "Querying initial memories" << std::endl;
    vector<double> query_pattern;
    int succes = 0 ;
    // double strength_drive = 0.1;
    for (int i = 0; i < initial_patterns_rates.size(); i++)
    {
        nb_winners=0;
        string result_file_traj_name = sim_data_foldername + "/results_" + to_string(i) + ".data";
        std::ofstream result_file_traj(result_file_traj_name, std::ios::trunc);
        query_pattern = pattern_as_states(net.transfer(drive_target), net.transfer(-drive_target), initial_patterns[i]);
        query_pattern = setToValueRandomElements(query_pattern, int(network_size/2), 0.5);
        net.set_state(query_pattern);
        // run_net_sim_query_drive(net, noisy_pattern, strength_drive, 1200, delta);
        // run_net_sim_noisy(net,2800, delta,0.0,0.01);
        run_net_sim_save(net,1/delta, delta, result_file_traj);
        // run_net_sim_noisy_save_display(net,10, 2800, delta,0,0.01, result_file_traj);// 
        // run_net_sim_noisy_save(net, 2800, delta,0,0.01, result_file_traj);
        for (size_t j = 0; j < initial_patterns[i].size(); j++)
        {
            nb_winners+=initial_patterns[i][j];   
        }
        winning_units = assignBoolThreshold(net.activity_list, 0.5);
        if (comparestates(winning_units,initial_patterns[i])){
            succes+=1;
        }
        std::cout << "writed pattern :" << std::endl;
        std::cout << i << std::endl;
        show_vector_double_grid(net.activity_list,IMAGE_HEIGHT);
    }
    std::cout << "success ?" << std::endl;
    std::cout << to_string(succes) << std::endl;

    weights_file_name = sim_data_foldername + "/weights.data";
    writeMatrixToFile(net.weight_matrix, weights_file_name);

    connectivity_file_name = sim_data_foldername + "/connectivity.data";
    writeBoolMatrixToFile(net.connectivity_matrix, connectivity_file_name);
    std::cout << nb_winners << std::endl;
}

int main(int argc, char **argv)
{
    // string sim_name = "write_net_sizes_relative_num_patterns";
    string sim_name = "Fig_Spontaneous_Recollection_rand_pat";
    string foldername_results = "../../../data/all_data_splited/trained_networks_fast/" + sim_name;

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
    // vector<double> all_relative_num_patterns = {0.5};
    // vector<double> network_sizes = {10,20,30,40,50,60,70,80,90,100};
    vector<double> drive_targets = {6};
    vector<double> network_sizes = {IMAGE_HEIGHT*IMAGE_WIDTH};
    // vector<double> repetitions = {0,1,2,3,4,5,6,7,8,9};
    unordered_map<string, vector<double>> varying_params = {
        // {"repetitions", repetitions},
        {"drive_target", drive_targets},
        {"learning_rate", {0.001}}, // REMOVED-target rates
        {"network_size", network_sizes},
        {"leak", {1.3}},
        {"num_patterns", {10}},
        {"epsilon_learning", {0.01}},
        {"delta", {0.01}}};
    

    lunchParalSim(foldername_results,varying_params,run_simulation);
    collectSimulationData(foldername_results);
    
    return 0;
}
