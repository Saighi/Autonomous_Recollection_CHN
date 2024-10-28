#include "network.hpp"
#include "utils.hpp"
#include <numeric>
#include <iostream>
#include <vector>
#include <random>
#include <filesystem> // for filesystem operations

using namespace std;

namespace fs = std::filesystem;

int main(int argc, char **argv)
{
    //WRITING RESULTS
    std::string foldername = "../data";
    std::string filename = foldername + "/queries_output_no_sleep.data";

        // Create directory if it doesn't exist
    if (!fs::exists(foldername)) {
        if (!fs::create_directory(foldername)) {
            std::cerr << "Error creating directory: " << foldername << std::endl;
            return 1;
        }
    }

    // Network constants
    int network_size = 25;
    int col_with = int(sqrt(25));
    double leak = 1.5;
    double delta = 0.5;

    // Simulation constants
    // double value_convergence = 0.01;
    // int nb_iter_sim = 200;

    // Learning constants
    double target_up_rate = 0.8;
    double target_down_rate = 0.2;
    double learning_rate = 0.08;
    int nb_winners = col_with; // number of winning neurons

    // Building training data
    vector<vector<bool>> initial_patterns = {
        {
        0,0,0,0,1,
        0,0,0,0,0,
        0,0,1,0,0,
        0,0,1,0,0,
        0,1,0,1,0,
        }
    };

    vector<vector<bool>> added_memory_list = {
        {
            0,0,1,0,0,
            0,0,1,0,0,
            0,0,1,0,0,
            0,0,1,0,0,
            0,0,1,0,0,
            },
            {
            0,0,0,0,0,
            0,0,0,0,0,
            1,1,1,1,1,
            0,0,0,0,0,
            0,0,0,0,0,
            },
            {
            0,0,0,0,1,
            0,0,0,1,0,
            0,0,1,0,0,
            0,1,0,0,0,
            1,0,0,0,0,
            },
            {
            1,0,0,0,0,
            0,1,0,0,0,
            0,0,1,0,0,
            0,0,0,1,0,
            0,0,0,0,1,
            },
            {
            0,1,0,0,0,
            0,1,0,0,0,
            0,1,0,0,0,
            0,1,0,0,0,
            0,1,0,0,0,
            },
            {
            0,0,0,1,0,
            0,0,0,1,0,
            0,0,0,1,0,
            0,0,0,1,0,
            0,0,0,1,0,
            },
            {
            0,0,0,0,0,
            0,0,0,0,0,
            0,0,0,0,0,
            1,1,1,1,1,
            0,0,0,0,0,
            },
            {
            0,0,0,0,0,
            1,1,1,1,1,
            0,0,0,0,0,
            0,0,0,0,0,
            0,0,0,0,0,
            }   

    };

    int size_initial_patterns = initial_patterns.size();
    int size_added_memory_list = added_memory_list.size();
    int nb_iter_learning = 300;

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
    appendToCSV(initial_patterns_state_list[0], filename); // TO CHANGE just to mesure correlation
    vector<vector<double>> added_memory_state_list(size_added_memory_list);

    for (int i = 0; i < size_added_memory_list; i++)
    {
        for (int j = 0; j < network_size; j++)
        {
            if (added_memory_list[i][j])
            {
                state_input[j] = target_up_rate;
            }
            else
            {
                state_input[j] = target_down_rate;
            }
        }
        added_memory_state_list[i] = state_input;
    }

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
    std::cout << "TRAINING THE FIRST NETWORK" << std::endl;

    for (int i = 0; i < nb_iter_learning; i++)
    {
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

    int nb_writted_patterns = size_initial_patterns;
    // Reading Network with sleep
    vector<bool> winning_units;
    vector<double> new_target_state;
    vector<vector<double>> readed_state_list(size_initial_patterns+size_added_memory_list);

    for(int h = 0; h<size_added_memory_list;h++){


        // Training loop
        std::cout << "TRAINING THE SECONDE NETWORK" << std::endl;

        for (int i = 0; i < nb_iter_learning; i++)
        {
            net.set_state(added_memory_state_list[h]);
            run_net_sim(net, 250, delta);
            net.reinforce_attractor(added_memory_state_list[h], learning_rate);


            // Querying
            std::cout << "Querying initial memories in the loop of correlation" << std::endl;
            for (int i = 0; i < size_initial_patterns; i++)
            {
                // std::cout << "NEW ITER" << std::endl;
                net.set_state(initial_patterns_state_list[i]);
                // show_state_grid(net, 3);
                run_net_sim_noisy(net, 250, delta, 0.0, 0.01);
                // std::cout << "after convergence" << std::endl;
                winning_units = assignBoolToTopNValues(net.activity_list, nb_winners);
                appendToCSV(net.rate_list, filename);
                show_vector_bool_grid(winning_units, col_with);
                // show_state_grid(net, 5, 5);
                // show_matrix(net.weight_matrix);
            }
        }
        nb_writted_patterns+=1;

        // // Querying
        // std::cout << "Querying initial memories in the loop of correlation" << std::endl;
        // for (int i = 0; i < size_initial_patterns; i++)
        // {
        //     // std::cout << "NEW ITER" << std::endl;
        //     net.set_state(initial_patterns_state_list[i]);
        //     // show_state_grid(net, 3);
        //     run_net_sim_noisy(net, 250, delta, 0.0, 0.01);
        //     // std::cout << "after convergence" << std::endl;
        //     winning_units = assignBoolToTopNValues(net.activity_list, nb_winners);
        //     appendToCSV(net.rate_list, filename);
        //     show_vector_bool_grid(winning_units, col_with);
        //     // show_state_grid(net, 5, 5);
        //     // show_matrix(net.weight_matrix);
        // }
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
