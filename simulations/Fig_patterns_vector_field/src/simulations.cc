#include <matio.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
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

void run_simulation(int sim_number, unordered_map<string, double> parameters,
                    const string foldername_results) {
    // Learning constants
    double epsilon_learning = 0.2;
    double drive_target = parameters.at("drive_target");
    double learning_rate = parameters.at("learning_rate");
    int network_size = 10;
    double leak = parameters.at("leak");
    double delta = parameters.at("delta");
    double noise_level = parameters.at("noise_level");

    string sim_data_foldername;
    string result_file_name;

    sim_data_foldername =
        foldername_results + "/sim_nb_" + to_string(sim_number);

    // Create directory if it doesn't exist
    if (!fs::exists(sim_data_foldername)) {
        if (!fs::create_directory(sim_data_foldername)) {
            std::cerr << "Error creating directory: " << sim_data_foldername
                      << std::endl;
            return;
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

    //---------------------------------------------------------- Generate data
    vector<vector<bool>> patterns = generatePatterns(2, network_size, network_size / 2, noise_level);
    vector <vector<double>> patterns_rates = patterns_as_states(
        net.transfer(drive_target), net.transfer(-drive_target), patterns);

    compute_and_save_rate_vector_field_two_pattern(net, sim_data_foldername, patterns_rates[0], patterns_rates[1], 3);

    // // Vector Field variable
    // double lower_potential = -6.0;
    // double higher_potential = 6.0;
    // double lower_rate = 0.05;
    // double higher_rate = 0.95;
    // double number_of_steps = 10;

    // //---------------------------------------------------------- Training
    // std::cout << "WRITING THE ATTRACTOR" << std::endl;
    // // Initialize velocity matrix for momentum
    // std::vector<std::vector<double>> velocity_matrix(
    //     network_size, std::vector<double>(network_size, 0.0));
    // double momentum_coef = 0.9;  // You can adjust this value
    // vector<double> drives_error;
    // drives_error.resize(network_size, 0.0);
    // // Training loop
    // double max_error = 1000;
    // int cpt = 0;
    // while (max_error > epsilon_learning && cpt <= 10 / learning_rate)
    // {
    //     for (int j = 0; j < initial_patterns.size(); j++)
    //     {
    //         // net.derivative_gradient_descent(initial_patterns[j],initial_patterns_rates[j],drive_target,learning_rate, leak, drives_error);
    //         net.derivative_gradient_descent_with_momentum(
    //             initial_patterns[j],
    //             initial_patterns_rates[j],
    //             drive_target,
    //             learning_rate,
    //             leak,
    //             drives_error,
    //             velocity_matrix,
    //             momentum_coef);
    //     }
    //     max_error = std::abs(*std::max_element(drives_error.begin(), drives_error.end()));
    //     cpt += 1;
    // }
    // std::cout << "nombre d'iterations : " << std::endl;
    // std::cout << cpt << std::endl;

    // //---------------------------------------------------------- Network evolution
    // std::cout << "Letting the network evolve from neutral state" << std::endl;
    // vector<double> query_pattern = {0.5,0.5};

    // string result_file_traj_name =
    //     sim_data_foldername + "/results_evolution.data";
    // std::ofstream result_file_traj(result_file_traj_name, std::ios::trunc);

    // net.set_state(query_pattern);
    // run_net_sim_save(net, 200, delta, result_file_traj);

    // std::string weights_file_name = sim_data_foldername + "/weights.data";
    // writeMatrixToFile(net.weight_matrix, weights_file_name);
}

int main(int argc, char **argv) {
    // string sim_name = "write_net_sizes_relative_num_patterns";
    string sim_name = "Fig_patterns_vector_field";
    string foldername_results =
        "../../../data/all_data_splited/trained_networks_fast/" + sim_name;

    // Create directory if it doesn't exist
    if (!fs::exists(foldername_results)) {
        if (!fs::create_directory(foldername_results)) {
            std::cerr << "Error creating directory: " << foldername_results
                      << std::endl;
            return 1;
        }
    }

    unordered_map<string, vector<double>> varying_params = {
        {"drive_target", {3.5}},
        {"learning_rate", {0.0001}},  
        {"leak", {1.3}},
        {"delta", {0.02}},
        {"noise_level",{1}}};

    lunchParalSim(foldername_results, varying_params, run_simulation);
    collectSimulationData(foldername_results);

    return 0;
}
