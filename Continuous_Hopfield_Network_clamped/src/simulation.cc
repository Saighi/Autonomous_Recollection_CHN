#include "network.hpp"
#include "utils.hpp"
#include <numeric>
#include <iostream>
#include <vector>
#include <random>

using namespace std;

int main(int argc, char **argv)
{

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
    vector<vector<bool>> input_list = {
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

    vector<vector<bool>> target_list = input_list;

    vector<bool> to_add = 
    {
    0,0,0,0,1,
    0,0,0,0,0,
    0,0,1,0,0,
    0,0,1,0,0,
    0,1,0,1,0,
    };

    vector<double> to_add_state_list(network_size);
    for (int j = 0; j < network_size; j++)
    {
        if (to_add[j]){
            to_add_state_list[j] = target_up_rate;
        }
        else{
            to_add_state_list[j] = target_down_rate;
        }
    }

    int size_training_data = target_list.size();
    int nb_iter_learning = 300;

    vector<vector<double>> input_state_list(size_training_data);
    vector<vector<double>> target_state_list(size_training_data);

    for (int i = 0; i < size_training_data; i++)
    {
        vector<double> state_input(network_size);
        vector<double> state_target(network_size);
        for (int j = 0; j < network_size; j++)
        {
            if (input_list[i][j])
            {
                state_input[j] = target_up_rate;
            }
            else
            {
                state_input[j] = target_down_rate;
            }

            if (target_list[i][j])
            {
                state_target[j] = target_up_rate;
            }
            else
            {
                state_target[j] = target_down_rate;
            }
        }
        input_state_list[i] = state_input;
        target_state_list[i] = state_target;
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
    Network net_2 = Network(connectivity_matrix, network_size, leak);

    // starting here chat gpt, change all the calls to use noisy iterations
    // The parameters for the gaussian noise should allow the network to settle 
    // to a stable state during the learning and during the testing. The noise 
    // is only their to help the convergence. 

    // Training loop
    std::cout << "TRAINING THE FIRST NETWORK" << std::endl;

    for (int i = 0; i < nb_iter_learning; i++)
    {
        for (int j = 0; j < size_training_data; j++)
        {
            // std::cout << "NEW ITER" << std::endl;
            net.set_state(input_state_list[j]);
            // show_state_grid(net, 3);
            run_net_sim(net, 250, delta);
            // std::cout << "after convergence" << std::endl;
            // show_state_grid(net, 3);
            net.reinforce_attractor(input_state_list[j], learning_rate);
            // show_matrix(net.weight_matrix);
        }
    }

    // Reading Network with sleep
    vector<bool> winning_units;
    vector<double> new_target_state;
    vector<vector<double>> readed_state_list(size_training_data);
    std::cout << "READING THE NETWORK ATTRACTORS" << std::endl;
    for (int r = 0; r < size_training_data; r++)
    {
        // std::cout << "NEW ITER" << std::endl;

        net.set_state(vector<double>(network_size, 0.5));
        // std::cout << "Initial random state:" << std::endl;
        // show_state_grid(net, 3); // Show initial state

        // Let the network converge
        run_net_sim_noisy_depressed(net, 250, delta, 0.0, 0.01); // Using utility function for noisy iterations
        // run_net_sim_noisy(net, 500, delta, 0.0, 0.01);           // Using utility function for noisy iterations
        winning_units = assignBoolToTopNValues(net.activity_list, nb_winners);
        net.pot_inhib_bin(0.01, winning_units); // works with 0.005

        std::cout << "State after convergence:" << std::endl;
        show_state_grid(net, col_with);
        show_vector_bool_grid(winning_units, col_with);

        new_target_state = assignStateToTopNValues(net.activity_list, nb_winners, target_up_rate, target_down_rate);
        readed_state_list[r] = new_target_state;
    }

    // Training loop
    std::cout << "TRAINING THE SECONDE NETWORK" << std::endl;

    for (int i = 0; i < nb_iter_learning; i++)
    {
        net_2.set_state(to_add_state_list);
        run_net_sim(net_2, 250, delta);
        net_2.reinforce_attractor(to_add_state_list, learning_rate);

        for(int j = 0; j<size_training_data;j++){
            // std::cout << "NEW ITER" << std::endl;
            net_2.set_state(readed_state_list[j]);
            // if(i==1){
            //     winning_units = assignBoolToTopNValues(net_2.activity_list, nb_winners);
            //     show_vector_bool_grid(winning_units, col_with);
            // }
            run_net_sim(net_2, 250, delta);
            // std::cout << "after convergence" << std::endl;
            // show_state_grid(net, 3);
            net_2.reinforce_attractor(readed_state_list[j], learning_rate);
            // show_matrix(net.weight_matrix);

        }
    }

    // Querying
    std::cout << "Querying learned memories" << std::endl;
    for (int i = 0; i < size_training_data; i++)
    {
        // std::cout << "NEW ITER" << std::endl;
        net_2.set_state(input_state_list[i]);
        // show_state_grid(net, 3);
        run_net_sim_noisy(net_2, 250, delta, 0.0, 0.01);
        // std::cout << "after convergence" << std::endl;
        winning_units = assignBoolToTopNValues(net_2.activity_list, nb_winners);
        show_vector_bool_grid(winning_units, col_with);
        // show_state_grid(net, 5, 5);
        // show_matrix(net.weight_matrix);
    }
    net_2.set_state(to_add_state_list);
    // show_state_grid(net, 3);
    run_net_sim_noisy(net_2, 250, delta, 0.0, 0.01);
    // std::cout << "after convergence" << std::endl;
    winning_units = assignBoolToTopNValues(net_2.activity_list, nb_winners);
    show_vector_bool_grid(winning_units, col_with);

    return 0;
}
