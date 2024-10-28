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

using namespace std;

namespace fs = std::filesystem;

int main(int argc, char **argv)
{
    std::string sim_name = "perceptron_parameter_test";
    //MAKING RESULTS FOlDER
    std::string foldername_results = "../../all_data/"+sim_name;

    // Create directory if it doesn't exist
    if (!fs::exists(foldername_results))
    {
        if (!fs::create_directory(foldername_results))
        {
            std::cerr << "Error creating directory: " << foldername_results << std::endl;
            std::cout << "Simulation folder file created at " << foldername_results << std::endl;
            return 1;
        }
    }
    
    // Varying parameters stored in a map
    unordered_map<string, vector<int>> varying_params = {
        {"num_patterns", {15}}};
    int sim_number = 0; // Will increment for each new sim, used for folder name

    // Learning constants
    double target_up_rate;
    double target_down_rate;
    double learning_rate;
    // Generating training data
    int network_size;
    int correlation;
    int nb_winners; // number of 1's neurons
    double noise_level;
    // network constants
    double leak;
    double delta;
    std::unordered_map<std::string, double> parameters;
    string sim_data_foldername;
    string patterns_file_name;
    string result_file_name;
    vector<vector<bool>> initial_patterns;
    int col_with;
    int number_iter;
    int nb_iter_learning;
    vector<vector<double>> initial_patterns_state_list;
    vector<bool> winning_units;

    for (int num_patterns : varying_params["num_patterns"])
    {
        // Learning constants
        target_up_rate = 0.75;
        target_down_rate = 0.25;
        // Generating training data
        network_size = 25;
        nb_winners = 10; // number of 1's neurons
        noise_level = 1;
        // network constants
        leak = 1.5;
        delta = 0.5;
        //Writing parameters
        nb_iter_learning = 2400;
        learning_rate = 0.01;
        //display parameters
        col_with = sqrt(network_size);

        parameters = {
            {"target_up_rate", target_up_rate},
            {"target_down_rate", target_down_rate},
            {"learning_rate", learning_rate},
            {"network_size", network_size},
            {"nb_winners", nb_winners},
            {"noise_level", noise_level},
            {"num_patterns", num_patterns},
            {"leak", leak},
            {"delta", delta},
        };

        sim_data_foldername = foldername_results + "/sim_nb_" + to_string(sim_number);

        // Create directory if it doesn't exist
        if (!fs::exists(sim_data_foldername))
        {
            if (!fs::create_directory(sim_data_foldername))
            {
                std::cerr << "Error creating directory: " << sim_data_foldername << std::endl;
                std::cout << "Simulation subfolder file created at " << sim_data_foldername << std::endl;  
                return 1;
            }
        }

        patterns_file_name = sim_data_foldername + "/patterns.data";
        std::ofstream file(patterns_file_name, std::ios::trunc);
        initial_patterns = generatePatterns(num_patterns, network_size, nb_winners, noise_level);

        for (int i = 0; i < num_patterns; i++)
        {
            writeBoolToCSV(file, initial_patterns[i]);
            show_vector_bool_grid(initial_patterns[i], col_with);
        }
        file.close();

        createParameterFile(sim_data_foldername, parameters);

        // Loading training data
        initial_patterns = loadPatterns(patterns_file_name);
        std::cout << num_patterns << std::endl;
        col_with = sqrt(network_size);
        for (int i = 0; i< num_patterns; i++){
            std::cout << "pattern"+ to_string(i) << std::endl;
            // show_vector_bool_grid(initial_patterns[i], col_with);
        }
        initial_patterns_state_list = patterns_as_states(target_up_rate, target_down_rate, initial_patterns);
        
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

        // Training loop
        std::cout << "TRAINING THE NETWORK" << std::endl;

        for (int i = 0; i < nb_iter_learning; i++)
        {
            // std::cout << to_string(i) << std::endl;
            for (int j = 0; j < num_patterns; j++)
            {
                net.set_state(initial_patterns_state_list[j]);
                run_net_sim(net, 250, delta);
                net.reinforce_attractor(initial_patterns_state_list[j], learning_rate);
            }
        }
        
        // Querying
        std::set<std::vector<bool>> foundVectors;
        std::cout << "Querying initial memories" << std::endl;
        for (int i = 0; i < num_patterns; i++)
        {
            net.set_state(initial_patterns_state_list[i]);
            run_net_sim_noisy(net, 250, delta, 0.0, 0.01);
            winning_units = assignBoolToTopNValues(net.activity_list, nb_winners);
            // show_vector_bool_grid(winning_units, col_with);

            // Check if the output vector is in the target set and hasn't been counted yet
            if (std::find(initial_patterns.begin(), initial_patterns.end(), winning_units) != initial_patterns.end() &&
                foundVectors.find(winning_units) == foundVectors.end())
            {
                // Count the vector
                foundVectors.insert(winning_units);
            }
        }

        // The number of unique vectors found
        std::cout << "Number of unique vectors found: " << foundVectors.size() << std::endl;

        result_file_name = sim_data_foldername + "/results.data";
        std::ofstream result_file(result_file_name, std::ios::trunc);
        result_file<<"nb_found_patterns="+foundVectors.size();
        result_file.close();

        sim_number += 1;
    }




    return 0;
}
