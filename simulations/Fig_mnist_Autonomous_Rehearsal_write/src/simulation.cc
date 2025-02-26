#include <matio.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <numeric>
#include <random>
#include <set>
#include <thread>
#include <unordered_map>
#include <vector>

#include "network.hpp"
#include "utils.hpp"

using json = nlohmann::json;

const int IMAGE_HEIGHT = 28;
const int IMAGE_WIDTH = 28;
using namespace std;

namespace fs = std::filesystem;

std::vector<bool> intToBoolVector(const std::vector<int>& intVector) {
    std::vector<bool> boolVector(intVector.size());
    std::transform(intVector.begin(), intVector.end(), boolVector.begin(),
                   [](int val) { return val != 0; });
    return boolVector;
}

std::vector<std::vector<bool>> load_digit_patterns(
    const std::string& filename) {
    std::ifstream f(filename);
    if (!f.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    json j;
    f >> j;

    std::vector<std::vector<bool>> balanced_patterns;
    for (const auto& pattern : j) {
        std::vector<int> original = pattern.get<std::vector<int>>();
        balanced_patterns.push_back(intToBoolVector(original));
    }

    return balanced_patterns;
}

double max_abs_element(const std::vector<std::vector<double>>& matrix) {
    double max_val = 0.0;

    for (const auto& row : matrix) {
        for (const auto& element : row) {
            double abs_val = std::abs(element);
            max_val = std::max(max_val, abs_val);
        }
    }

    return max_val;
}

void run_simulation(int sim_number, unordered_map<string, double> parameters, const string foldername_results)
{
    // Learning constants
    int cpt=0;
    int nb_pat = 5;
    double epsilon_learning=0.00005;
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
    vector<vector<bool>> all_patterns;
    vector<vector<bool>> initial_patterns;
    vector<vector<bool>> query_patterns;
    vector<vector<double>> initial_patterns_rates;
    vector<vector<double>> query_patterns_rates;
    vector<bool> winning_units;

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

    all_patterns = load_digit_patterns(
        "/home/saighi/Desktop/data/binarized_mnist/mnist_patterns.json");
    initial_patterns = {all_patterns[1], all_patterns[2], all_patterns[3],
                        all_patterns[4]};
    query_patterns= initial_patterns;
        
    // Build Fully connected network
    std::vector<std::vector<bool>>
        connectivity_matrix(network_size,
                            std::vector<bool>(network_size, false));
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
    query_patterns_rates = patterns_as_states(net.transfer(drive_target), net.transfer(-drive_target), query_patterns);

    vector<double> drives_error;
    // Initialize velocity matrix for momentum
    std::vector<std::vector<double>> velocity_matrix(
        network_size, std::vector<double>(network_size, 0.0));
    double momentum_coef = 0.9;  // You can adjust this value
    drives_error.resize(network_size, 0.0);
    // Training loop
    double max_error = 1000;
    cpt = 0;
    std::vector<std::vector<double>> old_weights;
    std::vector<std::vector<double>> new_weights;
    old_weights = net.weight_matrix;
    std::vector<std::vector<double>> d_weights(
        net.weight_matrix.size(),std::vector<double>(net.weight_matrix[0].size()));

    while (max_error > epsilon_learning && cpt <= 10/learning_rate)
    {
        for (int j = 0; j < initial_patterns.size(); j++)
        {
            // net.derivative_gradient_descent(initial_patterns[j],initial_patterns_rates[j],drive_target,learning_rate, leak, drives_error);
          net.derivative_gradient_descent_with_momentum(
                                                        initial_patterns[j],
                                                        initial_patterns_rates[j],
                                                        drive_target,
                                                        learning_rate,
                                                        leak,
                                                        drives_error,
                                                        velocity_matrix,
                                                        momentum_coef
                                                        );
        }
        new_weights = net.weight_matrix;
        for (size_t row = 0; row < net.weight_matrix.size(); row++)
        {
            for (size_t collumn = 0; collumn < net.weight_matrix[row].size(); collumn++) {
                d_weights[row][collumn] = new_weights[row][collumn]-old_weights[row][collumn];
            }
        }
        old_weights=new_weights;
        max_error = max_abs_element(d_weights);
        cpt+=1;
        std::cout << max_error  << std::endl;
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
        //CHANGE
        query_pattern = setToValueRandomElements(query_patterns_rates[i],int(network_size/2),0.5);
        // for (size_t i = 0; i < query_pattern.size(); i++)
        // {
        //     std::cout << query_pattern[i] << std::endl;
        // }
        
        net.set_state(query_pattern);
        // run_net_sim_query_drive(net, noisy_pattern, strength_drive, 1200, delta);
        // run_net_sim_noisy(net,2800, delta,0.0,0.01);
        run_net_sim_save(net,300, delta, result_file_traj);
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
    createParameterFile(sim_data_foldername, parameters);
}

int main(int argc, char **argv)
{
    // string sim_name = "write_net_sizes_relative_num_patterns";
    string sim_name = "Fig_mnist_Autonomous_Rehearsal";
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
    vector<double> network_sizes = {IMAGE_HEIGHT*IMAGE_WIDTH};
    // vector<double> repetitions = {0,1,2,3,4,5,6,7,8,9};
    unordered_map<string, vector<double>> varying_params = {
        // {"repetitions", repetitions},
        {"drive_target", drive_targets},
        {"learning_rate", {0.0003}}, // REMOVED-target rates
        {"network_size", network_sizes},
        {"leak", {1.3}},
        {"delta", {0.02}}};
    

    lunchParalSim(foldername_results,varying_params,run_simulation);
    collectSimulationData(foldername_results);
    
    return 0;
}
