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
    int network_size = 2;
    double leak = parameters.at("leak");
    double delta = parameters.at("delta");

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
    vector<vector<bool>> connectivity_matrix(network_size,
                                             vector<bool>(network_size, false));
    connectivity_matrix[0][0]= false;
    connectivity_matrix[1][1]= true;
    connectivity_matrix[0][1]= true;
    connectivity_matrix[1][0] = true;

    Network net = Network(connectivity_matrix, network_size, leak);

    //---------------------------------------------------------- Generate data
    vector<bool> pattern = {0, 1};
    vector<double> pattern_rates = pattern_as_states(
        net.transfer(drive_target), net.transfer(-drive_target), pattern);
    // Vector Field variables
    double lower_potential = -6.0;
    double higher_potential = 6.0;
    double lower_rate = 0.05;
    double higher_rate = 0.95;
    double number_of_steps = 10;
    std::cout << net.transfer(lower_potential) << std::endl;
    std::cout << net.transfer(higher_potential) << std::endl;
    // ---------------------------------------------------------- BEFORE
    // TRAINING:
    {
        // Save vector field in activity space
        std::string act_field_pre =
            sim_data_foldername + "/vector_field_pre_pot.txt";
        compute_and_save_potential_vector_field(
            net, act_field_pre, lower_potential, higher_potential,
            (higher_potential - lower_potential) / number_of_steps);
        // Save vector field in rate space
        std::string rate_field_pre =
            sim_data_foldername + "/vector_field_pre_rate.txt";
        compute_and_save_rate_vector_field(
            net, rate_field_pre, lower_rate, higher_rate,
            (higher_rate - lower_rate) / number_of_steps);
    }

    //---------------------------------------------------------- Training
    std::cout << "WRITING THE ATTRACTOR" << std::endl;
    // Initialize velocity matrix for momentum
    std::vector<std::vector<double>> velocity_matrix(
        network_size, std::vector<double>(network_size, 0.0));
    double momentum_coef = 0.9;  // You can adjust this value
    vector<double> drives_error;
    drives_error.resize(network_size, 0.0);
    // Training loop
    double max_error = 1000;
    int cpt = 0;
    while (max_error > epsilon_learning && cpt <= 10 / learning_rate) {
        // net.derivative_gradient_descent(pattern, pattern_rates, drive_target,
        //                                 learning_rate, leak, drives_error);
        net.derivative_gradient_descent_with_momentum(
            pattern, pattern_rates, drive_target, learning_rate, leak,
            drives_error, velocity_matrix, momentum_coef);

        max_error = std::abs(
            *std::max_element(drives_error.begin(), drives_error.end()));
        cpt += 1;
    }
    std::cout << "nombre d'iterations : " << std::endl;
    std::cout << cpt << std::endl;

    // ---------------------------------------------------------- BEFORE
    // TRAINING:
    {
        // Save vector field in activity space
        std::string act_field_pre =
            sim_data_foldername + "/vector_field_post_pot.txt";
        compute_and_save_potential_vector_field(
            net, act_field_pre, lower_potential, higher_potential,
            (higher_potential - lower_potential) / number_of_steps);
        // Save vector field in rate space
        std::string rate_field_pre =
            sim_data_foldername + "/vector_field_post_rate.txt";
        compute_and_save_rate_vector_field(
            net, rate_field_pre, lower_rate, higher_rate,
            (higher_rate - lower_rate) / number_of_steps);
    }

    //---------------------------------------------------------- Network evolution
    std::cout << "Letting the network evolve from neutral state" << std::endl;
    vector<double> query_pattern = {0.5,0.5};

    string result_file_traj_name =
        sim_data_foldername + "/results_evolution.data";
    std::ofstream result_file_traj(result_file_traj_name, std::ios::trunc);

    net.set_state(query_pattern);
    run_net_sim_save(net, 200, delta, result_file_traj);

    std::string weights_file_name = sim_data_foldername + "/weights.data";
    writeMatrixToFile(net.weight_matrix, weights_file_name);
}

int main(int argc, char **argv) {
    // string sim_name = "write_net_sizes_relative_num_patterns";
    string sim_name = "Fig_pattern_vector_field";
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
        {"delta", {0.02}}};

    lunchParalSim(foldername_results, varying_params, run_simulation);
    collectSimulationData(foldername_results);

    return 0;
}
