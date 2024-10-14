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
    double leak = parameters.at("leak");
    double delta = parameters.at("delta");
    int col_with = sqrt(network_size);
    string sim_data_foldername;
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
    int nb_winners = 0;
    initial_patterns = { {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 1, 1, 1, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0, 1, 0,
        0, 1, 0, 0, 0, 0, 0, 0, 1, 0,
        0, 1, 0, 0, 0, 0, 0, 0, 1, 0,
        0, 1, 0, 0, 0, 0, 0, 0, 1, 0,
        0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 1, 0, 0, 1, 0, 0, 0,
        0, 1, 1, 1, 0, 0, 1, 1, 1, 0} };  
    // Reverse the initial pattern:
    for (size_t i = 0; i < initial_patterns.size(); i++)
    {
        for (size_t j = 0; j < initial_patterns[i].size(); j++)
        {
            initial_patterns[i][j] = initial_patterns[i][j] == 1 ? 0 : 1; 
        }
    }
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
        for (int j = 0; j < initial_patterns.size(); j++)
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
    vector<double> query_pattern;
    int succes = 0 ;
    // double strength_drive = 0.1;
    for (int i = 0; i < initial_patterns_rates.size(); i++)
    {
        string result_file_traj_name = sim_data_foldername + "/results_" + to_string(i) + ".data";
        std::ofstream result_file_traj(result_file_traj_name, std::ios::trunc);
        //TODO - change the pattern_as_states and link the target drive not magic number
        query_pattern=pattern_as_states(net.transfer(drive_target),net.transfer(-drive_target),initial_patterns[i]);
        // query_pattern=setToValueRandomElements(query_pattern, int(network_size*1), 0.5); // fully random initialization
        query_pattern = std::vector<double>(network_size,0.5);
        net.set_state(query_pattern);
        // run_net_sim_query_drive(net, noisy_pattern, strength_drive, 1200, delta);
        // run_net_sim_noisy(net,2800, delta,0.0,0.01);
        run_net_sim_save(net,2800, delta, result_file_traj);
        // run_net_sim_noisy_save_display(net,10, 2800, delta,0,0.01, result_file_traj);// 
        // run_net_sim_noisy_save(net, 2800, delta,0,0.01, result_file_traj);
        for (size_t j = 0; j < initial_patterns[i].size(); j++)
        {
            nb_winners+=initial_patterns[i][j];   
        }
        winning_units = assignBoolToTopNValues(net.activity_list, nb_winners);
        if (comparestates(winning_units,initial_patterns[i])){
            succes+=1;
        }
        std::cout << "writed pattern :" << std::endl;
        std::cout << i << std::endl;
        show_vector_double_grid(net.activity_list,10);
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
    string sim_name = "CHN_convergence";
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
    // vector<double> all_relative_num_patterns = {0.5};
    // vector<double> network_sizes = {10,20,30,40,50,60,70,80,90,100};
    vector<double> drive_targets = {6};
    vector<double> network_sizes = {100};
    // vector<double> repetitions = {0,1,2,3,4,5,6,7,8,9};
    unordered_map<string, vector<double>> varying_params = {
        // {"repetitions", repetitions},
        {"drive_target", drive_targets},
        {"drive_target", {5}},
        {"learning_rate", {0.01}}, // REMOVED-target rates
        {"network_size", network_sizes},
        {"leak", {1.3}},
        {"delta", {0.5}}};
    

    lunchParalSim(foldername_results,varying_params,run_simulation);
    collectSimulationData(foldername_results);
    
    return 0;
}
