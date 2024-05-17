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
    int network_size = 9;
    double leak = 1;
    double delta = 0.1;

    // Simulation constants
    // double value_convergence = 0.01;
    // int nb_iter_sim = 200;

    // Learning constants
    int nb_iter_learning = 1000;
    double target_up_rate = 0.75;
    double target_down_rate = 0.25;
    double learning_rate = 0.2;
    int size_training_data = 3;
    int nb_winners = 3; //number of winning neurons

    // Building training data
    vector<vector<bool>> input_list = {
            {
            0,1,0,
            0,1,0,
            0,1,0,
            },
            {
            0,0,0,
            1,1,1,
            0,0,0,
            },            
            {
            0,0,0,
            0,1,1,
            0,0,1,
            },
    };

    vector<vector<bool>> target_list = {
            {
            0,1,0,
            0,1,0,
            0,1,0,
            },
            {
            0,0,0,
            1,1,1,
            0,0,0,
            },            
            {
            0,0,0,
            0,1,1,
            0,0,1,
            },
    };

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

    // starting here chat gpt, change all the calls to use noisy iterations
    // The parameters for the gaussian noise should allow the network to settle 
    // to a stable state during the learning and during the testing. The noise 
    // is only their to help the convergence. 

    // Training loop
    for (int i = 0; i < nb_iter_learning; i++)
    {
        // std::cout << "NEW ITER" << std::endl;
        net.set_state(input_state_list[i % size_training_data]);
        //show_state_grid(net, 3);
        run_net_sim(net, 500, delta);
        // std::cout << "after convergence" << std::endl;
        // show_state_grid(net, 3);
        net.reinforce_attractor(target_state_list[i % size_training_data], learning_rate);
        //show_matrix(net.weight_matrix);

    }

    for (int r = 0; r < 10 ; r++){
        std::cout << "NEW ITER" << std::endl;

        net.set_state(vector<double>(network_size,0.5));
        // std::cout << "Initial random state:" << std::endl;
        // show_state_grid(net, 3); // Show initial state 

        // Let the network converge
        run_net_sim_noisy_depressed(net, 500, delta, 0.0, 0.01); // Using utility function for noisy iterations
        // run_net_sim_noisy(net, 500, delta, 0.0, 0.01); // Using utility function for noisy iterations
        std::cout << "State after convergence:" << std::endl;
        show_state_grid(net, 3); // Show state after convergence to check for attractor
        net.pot_inhib(0.01);
        // std::cout << "target sum inhib weights vector :" << std::endl;
        // show_vector(net.target_sum_each_inhib);
        // std::cout << "actual vector before norm :" << std::endl;
        // show_vector(net.actual_sum_each_inhib);
        //net.iterative_normalize(100, 0.01);
        // std::cout << "target sum inhib weights vector :" << std::endl;
        // show_vector(net.target_sum_each_inhib);
        // std::cout << "actual vector after norm :" << std::endl;
        // show_vector(net.actual_sum_each_inhib);
    }

    return 0;
}
