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

// Helper function to compare vectors of bool
bool areVectorsEqual(const std::vector<bool> &v1, const std::vector<bool> &v2)
{
    return v1 == v2;
}

using namespace std;

namespace fs = std::filesystem;

int main(int argc, char **argv)
{
    //WRITING RESULTS
    std::string foldername = "../../data/larger_network";
    std::string filename = foldername + "/output_sleep_";

    // Create directory if it doesn't exist
    if (!fs::exists(foldername)) {
        if (!fs::create_directory(foldername)) {
            std::cerr << "Error creating directory: " << foldername << std::endl;
            return 1;
        }
    }

    // Simulation constants
    // double value_convergence = 0.01;
    // int nb_iter_sim = 200;

    // Learning constants
    double target_up_rate = 0.75;
    double target_down_rate = 0.25;
    double learning_rate = 0.01;


    // Loading training data
    string patterns_file_name = "../../input_data/correlated_patterns/patterns.data";
    vector<vector<bool>> initial_patterns = loadPatterns(patterns_file_name);
    int num_patterns = initial_patterns.size();
    std::cout << num_patterns << std::endl;

    // Network constants
    int network_size = initial_patterns[0].size();
    int col_with = int(sqrt(network_size));
    double leak = 1.5;
    double delta = 0.5;
    int nb_winners = 10; // number of winning neurons

    for (int i = 0; i< num_patterns; i++){
        std::cout << "pattern"+ to_string(i) << std::endl;
        show_vector_bool_grid(initial_patterns[i], col_with);
    }

    // SIMULATION PARAMETERS
    int number_iter = 65;
    int nb_beta = 10;
    std::vector<double> betas = linspace(0.001, 0.01, nb_beta);
    // std::vector<double> betas = {0.0005,0.001};
    // int nb_beta = betas.size();
    // SIMULATION PARAMETERS
    int size_initial_patterns = initial_patterns.size();

    string end_filename;
    std::vector<std::vector<std::ofstream>> files;
    for (int h = 0; h < betas.size(); h++)
    {
        files.emplace_back();
        for (int i = 0; i < number_iter; i++)
        {
            end_filename = filename + to_string(h) + "_" + to_string(i) + ".data";
            std::ofstream file(end_filename, std::ios::trunc);
            files[h].emplace_back(end_filename);
            if (!files[h].back().is_open())
            {
                std::cerr << "Error opening file: " << end_filename << std::endl;
                return 1;
            }
        }
    }

    int nb_iter_learning = 2400;

    vector<double> state_input(network_size);

    vector<vector<double>> initial_patterns_state_list(size_initial_patterns);
    for (int i = 0; i < size_initial_patterns; i++)
    {
        for (int j = 0; j < network_size; j++){
            if (initial_patterns[i][j])
            {
                state_input[j] = target_up_rate;
            }
            else
            {
                state_input[j] = target_down_rate;
            }

        }
        initial_patterns_state_list[i] = state_input;
    }

    // WRITING the patterns to measure correlation later
    std::ofstream file_patterns(filename + "patterns.data", std::ios::trunc);
    for (int i = 0; i < size_initial_patterns; i++)
    {
        writeToCSV(file_patterns, initial_patterns_state_list[i]);
    }
    file_patterns.close();

    // Build Fully connected network
    vector<vector<bool>> connectivity_matrix(network_size, vector<bool>(network_size, false));

    for (int i = 0; i < network_size; i++)
    {
        for (int j = 0; j < network_size; j++)
        {
            if(i!=j){
                connectivity_matrix[i][j] = true;
            }
        }
    }

    Network net = Network(connectivity_matrix, network_size, leak);
    // starting here chat gpt, change all the calls to use noisy iterations
    // The parameters for the gaussian noise should allow the network to settle 
    // to a stable state during the learning and during the testing. The noise 
    // is only their to help the convergence. 

    // Training loop
    std::cout << "TRAINING THE NETWORK" << std::endl;

    for (int i = 0; i < nb_iter_learning; i++)
    {
        std::cout << to_string(i) << std::endl;
        for (int j = 0; j < size_initial_patterns; j++)
        {
            // std::cout << "NEW ITER" << std::endl;
            net.set_state(initial_patterns_state_list[j]);
            // show_state_grid(net, 3);
            run_net_sim(net, 250, delta);
            // std::cout << "after convergence" << std::endl;
            // show_state_grid(net, 3);
            net.reinforce_attractor(initial_patterns_state_list[j], learning_rate);
            // show_matrix(net.weight_matrix);
        }
    }

    // std::set<std::vector<bool>> foundVectors;
    // vector<bool> winning_units;
    // // Querying
    // std::cout << "Querying initial memories" << std::endl;
    // for (int i = 0; i < size_initial_patterns; i++)
    // {
    //     // std::cout << "NEW ITER" << std::endl;
    //     net.set_state(initial_patterns_state_list[i]);
    //     // show_state_grid(net, 3);
    //     run_net_sim_noisy(net, 250, delta, 0.0, 0.01);
    //     // std::cout << "after convergence" << std::endl;
    //     winning_units = assignBoolToTopNValues(net.activity_list, nb_winners);
    //     // appendToCSV(net.rate_list, filename);
    //     show_vector_bool_grid(winning_units, col_with);
    //     // show_state_grid(net, 5, 5);
    //     // show_matrix(net.weight_matrix);
    //     // Check if the output vector is in the target set and hasn't been counted yet
    //     if (std::find(initial_patterns.begin(), initial_patterns.end(), winning_units) != initial_patterns.end() &&
    //         foundVectors.find(winning_units) == foundVectors.end())
    //     {
    //         // Count the vector
    //         foundVectors.insert(winning_units);
    //     }
    // }

    // // The number of unique vectors found
    // std::cout << "Number of unique vectors found: " << foundVectors.size() << std::endl;

    // Reading Network with sleep
    vector<bool> winning_units;
    vector<double> new_target_state;
    vector<vector<double>> readed_state_list(number_iter);
                                 
    for(int h = 0; h<betas.size();h++){

        // THIS PART ADD THE SLEEP RELEARNING, TO REMOVE IN ORDER TO REMOVE SLEEP
        std::cout << "READING THE NETWORK ATTRACTORS" << std::endl;
        for (int r = 0; r < number_iter; r++)
        {
            // std::cout << "NEW ITER" << std::endl;
            net.set_state(vector<double>(network_size, 0.5));
            // std::cout << "Initial random state:" << std::endl;
            // show_state_grid(net, 3); // Show initial state

            // Let the network converge
            run_net_sim_noisy_depressed_save(net, 800, delta, 0.0, 0.01, files[h][r]); // Using utility function for noisy iterations and saving

            // run_net_sim_noisy(net, 500, delta, 0.0, 0.01);           // Using utility function for noisy iterations
            winning_units = assignBoolToTopNValues(net.activity_list, nb_winners);
            net.pot_inhib_bin(betas[h], winning_units); // works with 0.005

            // std::cout << "State after convergence:" << std::endl;
            // show_state_grid(net, col_with);
            // show_vector_bool_grid(winning_units, col_with);

            new_target_state = assignStateToTopNValues(net.activity_list, nb_winners, target_up_rate, target_down_rate);
            readed_state_list[r] = new_target_state;
        }

        net.reset_inhib();
    }
    for(int h = 0; h<betas.size();h++){
    std::cout << betas[h] << std::endl;
    }


    // // Querying
    // std::cout << "Querying initial memories" << std::endl;
    // for (int i = 0; i < size_initial_patterns; i++)
    // {
    //     // std::cout << "NEW ITER" << std::endl;
    //     net.set_state(initial_patterns_state_list[i]);
    //     // show_state_grid(net, 3);
    //     run_net_sim_noisy(net, 250, delta, 0.0, 0.01);
    //     // std::cout << "after convergence" << std::endl;
    //     winning_units = assignBoolToTopNValues(net.activity_list, nb_winners);
    //     // appendToCSV(net.rate_list, filename);
    //     show_vector_bool_grid(winning_units, col_with);
    //     // show_state_grid(net, 5, 5);
    //     // show_matrix(net.weight_matrix);
    // }
    //     // Querying
    // std::cout << "Querying added memories" << std::endl;
    // for (int i = 0; i < size_added_memory_list; i++)
    // {
    //     // std::cout << "NEW ITER" << std::endl;
    //     net.set_state(added_memory_state_list[i]);
    //     // show_state_grid(net, 3);
    //     run_net_sim_noisy(net, 250, delta, 0.0, 0.01);
    //     // std::cout << "after convergence" << std::endl;
    //     winning_units = assignBoolToTopNValues(net.activity_list, nb_winners);
    //     show_vector_bool_grid(winning_units, col_with);
    //     // show_state_grid(net, 5, 5);
    //     // show_matrix(net.weight_matrix);
    // }

    return 0;
}
