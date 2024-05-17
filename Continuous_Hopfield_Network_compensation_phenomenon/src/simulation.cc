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
    double delta = 0.05;

    // Simulation constants
    // double value_convergence = 0.01;
    int nb_iter_sim = 100;

    // Learning constants
    int nb_iter_learning = 500;
    double target_up_rate = 0.75;
    double target_down_rate = 0.25;
    double learning_rate = 0.05;
    int size_training_data = 2;

    // Building training data
    vector<vector<bool>> input_list = {
        {
            0,1,0,
            0,1,0,
            0,0,0,
            },
            {
            0,0,0,
            0,1,1,
            0,0,0,
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
            connectivity_matrix[i][j] = true;
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
        run_net_sim_noisy(net, nb_iter_sim, delta, 0.0, 0.05); // Example noise parameters
        // std::cout << "after convergence" << std::endl;
        //show_state_grid(net, 3);
        net.reinforce_attractor(target_state_list[i % size_training_data], learning_rate);
        //show_weight_matrix(net);

    }

    // initializing the network with
    // a random state using set_state such that neuron rate state is pickup randomly between
    // 0.25 and 0.75 with 0.5%, 0.5 % probabilities and then let the network
    // converge with run_net_sim to see if the network falls into an attractor.
    // Code addition starts here
    // Initialize the network with a random state
    vector<double> random_state(network_size);
    std::default_random_engine generator;
    std::discrete_distribution<int> distribution({50, 50}); // Probabilities for lower and upper bounds

    for (int r = 0; r < 10 ; r++){
        std::cout << "NEW ITER" << std::endl;
        for (int i = 0; i < network_size; ++i)
        {
            double rate = (distribution(generator) == 0) ? 0.25 : 0.75; // Assign rate based on probability
            random_state[i] = rate;
        }

        net.set_state(random_state);
        std::cout << "Initial random state:" << std::endl;
        show_state_grid(net, 3); // Show initial state

        // Let the network converge
        run_net_sim_noisy(net, nb_iter_sim+200, delta, 0.0, 0.1); // Using utility function for noisy iterations
        std::cout << "State after convergence:" << std::endl;
        show_state_grid(net, 3); // Show state after convergence to check for attractor
                                // Code addition ends here
    }


    // double added_state_prev = 0;
    // double added_state_post = 0;

    // for(int i = 0; i<nb_iter; i++){

    //     show_state_grid(net,3);
    //     net.iterate(delta);
    // }

    return 0;
}
