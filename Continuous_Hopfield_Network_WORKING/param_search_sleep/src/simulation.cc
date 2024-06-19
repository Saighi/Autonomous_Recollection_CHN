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
#include <sstream>

using namespace std;

namespace fs = std::filesystem;

int main(int argc, char **argv)
{
    std::string sim_name = "sleep_parameter_test";
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

    int nb_beta = 30;
    std::vector<double> betas = linspace(0.0005, 0.005, nb_beta);
    // Varying parameters stored in a map
    unordered_map<string, vector<int>> varying_params_int = {
        {"nb_pattern", {5,7,9,11,13}},
        {"nb_winners", {5,7,9,11,13}}};
    unordered_map<string, vector<double>> varying_params_double = {
        {"betas", betas}};

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
    string parameter_file_name;
    vector<vector<bool>> initial_patterns;
    vector<vector<vector<bool>>> all_initial_patterns;
    int col_with;
    int nb_iter;
    int nb_iter_learning;
    vector<vector<double>> initial_patterns_state_list;
    vector<bool> winning_units;

    // Should put all those all somewhere
    vector<Network> all_networks;
    vector<string> all_folder_names;
    vector<int> all_sim_nb_patterns;
    vector<int> all_number_of_winner;
    // vector<string> all_params_description;

    for (int nb_patterns : varying_params_int["nb_pattern"])
    {
        for (int nb_winners : varying_params_int["nb_winners"])
        {
            all_number_of_winner.push_back(nb_winners);
            all_sim_nb_patterns.push_back(nb_patterns);
            // Learning constants
            target_up_rate = 0.75;
            target_down_rate = 0.25;
            // Generating training data
            network_size = 25;
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
                {"nb_patterns", nb_patterns},
                {"leak", leak},
                {"delta", delta},
            };

            sim_data_foldername = foldername_results + "/sim_nb_" + to_string(sim_number);
            createParameterFile(sim_data_foldername, parameters);
            all_folder_names.push_back(sim_data_foldername);
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
            initial_patterns = generatePatterns(nb_patterns, network_size, nb_winners, noise_level);
            all_initial_patterns.push_back(initial_patterns);
            for (int i = 0; i < nb_patterns; i++)
            {
                writeBoolToCSV(file, initial_patterns[i]);
                show_vector_bool_grid(initial_patterns[i], col_with);
            }
            file.close();


            // Loading training data
            initial_patterns = loadPatterns(patterns_file_name);
            std::cout << nb_patterns << std::endl;
            col_with = sqrt(network_size);
            for (int i = 0; i< nb_patterns; i++){
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
                for (int j = 0; j < nb_patterns; j++)
                {
                    net.set_state(initial_patterns_state_list[j]);
                    run_net_sim(net, 250, delta);
                    net.reinforce_attractor(initial_patterns_state_list[j], learning_rate);
                }
            }
            
            all_networks.push_back(net);
            // Querying
            std::set<std::vector<bool>> foundVectors;
            std::cout << "Querying initial memories" << std::endl;
            for (int i = 0; i < nb_patterns; i++)
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
            result_file << "nb_found_patterns=" << foundVectors.size() << std::endl;
            result_file.close();

            sim_number += 1;
        }
    }

    // SLEEPING SIMULATIONS

    string sim_sleep_data_foldername;
    string full_traj_results_file_name_sleep;
    string scalar_results_file_name_retrieval;
    string iter_nb_retrieve_results_file_name;
    string iter_nb_spurious_results_file_name;
    int iter_all_retrieved;
    int nb_spurious_patterns;
    bool check;
    //actual sim is an index corresponding to the writted network on which sleep should be played
    for (int actual_sim = 0; actual_sim <sim_number; actual_sim++) 
    {
        Network net = all_networks[actual_sim];
        int sleep_sim_number = 0;

        //Static sleep parameters
        nb_winners = all_number_of_winner[actual_sim];
        nb_iter = all_sim_nb_patterns[actual_sim]*4; // enough iterations to recover most patterns even for fairly low betas

        for (float beta : varying_params_double["betas"]) //looping over varying sleep parameters
        {
            parameters = {
                {"beta", beta},
                {"nb_iter", nb_iter},
            };
            std::set<std::vector<bool>> foundVectors;
            sim_sleep_data_foldername = all_folder_names[actual_sim] + "/sleep_sim_nb_" + to_string(sleep_sim_number);
            createParameterFile(sim_sleep_data_foldername, parameters);

            // Create directory if it doesn't exist
            if (!fs::exists(sim_sleep_data_foldername))
            {
                if (!fs::create_directory(sim_sleep_data_foldername))
                {
                    std::cerr << "Error creating directory: " << sim_sleep_data_foldername << std::endl;
                    std::cout << "Sleep Simulation subfolder file created at " << sim_sleep_data_foldername << std::endl;
                    return 1;
                }
            }

            // result_file_name_sleep = sim_sleep_data_foldername + "/results.data";
            // std::ofstream result_file_sleep(result_file_name_sleep, std::ios::trunc);

            // THIS PART ADD THE SLEEP RELEARNING, TO REMOVE IN ORDER TO REMOVE SLEEP
            std::cout << "READING THE NETWORK ATTRACTORS" << std::endl;
            check = false;
            nb_spurious_patterns = 0;
            std::ostringstream iter_nb_retrieved;
            std::ostringstream iter_nb_spurious;
            for (int r = 0; r < nb_iter; r++)
            {
                // std::cout << "NEW ITER" << std::endl;
                net.set_state(vector<double>(network_size, 0.5));
                // std::cout << "Initial random state:" << std::endl;
                // show_state_grid(net, 3); // Show initial state

                // Let the network converge
                full_traj_results_file_name_sleep = sim_sleep_data_foldername + "/results_"+to_string(r)+".data";
                std::ofstream full_traj_results_file_sleep(full_traj_results_file_name_sleep, std::ios::trunc);
                run_net_sim_noisy_depressed_save(net, 800, delta, 0.0, 0.01, full_traj_results_file_sleep); // Using utility function for noisy iterations and saving
                full_traj_results_file_sleep.close();
                // run_net_sim_noisy(net, 500, delta, 0.0, 0.01);           // Using utility function for noisy iterations
                winning_units = assignBoolToTopNValues(net.activity_list, nb_winners);
                net.pot_inhib_bin(beta, winning_units); // works with 0.005

                // Check if the output vector is in the target set and hasn't been counted yet
                if (std::find(all_initial_patterns[actual_sim].begin(), all_initial_patterns[actual_sim].end(), winning_units) != all_initial_patterns[actual_sim].end())
                {
                    if (foundVectors.find(winning_units) == foundVectors.end())
                    {
                        // Count the vector
                        foundVectors.insert(winning_units);
                    }
                }else{
                    nb_spurious_patterns+=1;
                }

                if(foundVectors.size()==all_initial_patterns[actual_sim].size() && !check){
                    iter_all_retrieved = r;
                    check=true;
                }
                iter_nb_spurious<<r<<" "<<nb_spurious_patterns<< std::endl;
                iter_nb_retrieved<<r<<" "<<foundVectors.size()<< std::endl;
                // std::cout << "State after convergence:" << std::endl;
                // show_state_grid(net, col_with);
                // show_vector_bool_grid(winning_units, col_with);
            }

            // The number of unique vectors found
            std::cout << "Number of unique vectors found: " << foundVectors.size() << std::endl;

            scalar_results_file_name_retrieval = sim_sleep_data_foldername + "/scalar_results.data";
            std::ofstream scalar_results_file_retrieval(scalar_results_file_name_retrieval, std::ios::trunc);
            scalar_results_file_retrieval << "nb_found_patterns=" << foundVectors.size()<<"\n";
            scalar_results_file_retrieval << "nb_spurious_patterns=" << nb_spurious_patterns << "\n";
            scalar_results_file_retrieval << "iter_all_retrieved=" << iter_all_retrieved << std::endl;
            scalar_results_file_retrieval.close();

            iter_nb_retrieve_results_file_name = sim_sleep_data_foldername + "/iter_nb_retrieve_results.data";
            std::ofstream iter_nb_retrieve_results_file_retrieval(iter_nb_retrieve_results_file_name, std::ios::trunc);
            iter_nb_retrieve_results_file_retrieval << iter_nb_retrieved.str() ;
            iter_nb_retrieve_results_file_retrieval.close();

            iter_nb_spurious_results_file_name = sim_sleep_data_foldername + "/iter_nb_spurious_results.data";
            std::ofstream iter_nb_spurious_results_file_retrieval(iter_nb_spurious_results_file_name, std::ios::trunc);
            iter_nb_spurious_results_file_retrieval << iter_nb_spurious.str() ;
            iter_nb_spurious_results_file_retrieval.close();

            net.reset_inhib();
            sleep_sim_number+=1;

        }
    }
    return 0;
}
