#include "network.hpp"
#include "utils.hpp"
#include <numeric>
#include <iostream>
#include <vector>
#include <random>
#include <cmath> // Include for sqrt

using namespace std;

int main(int argc, char **argv)
{

    // Network constants
    int network_size = 25;
    double leak = 1;
    double delta = 0.1;
    int colum_size = static_cast<int>(sqrt(network_size));
    // Simulation constants
    // double value_convergence = 0.01;
    int nb_iter_sim = 200;

    // Learning constants
    int nb_iter_learning = 1200;
    double target_up_rate = 0.75;
    double target_down_rate = 0.25;
    double learning_rate = 0.1;
    double learning_rate_heb = 0.002;
    int nb_winners = 5; //number of winning neurons

    // Building training data
    vector<vector<bool>> training_inputs = {
        {
            0,0,0,0,0,
            0,0,0,0,0,
            0,0,1,0,0,
            0,0,1,0,0,
            0,0,1,0,0,
            }
    };

    vector<vector<bool>> training_targets = {
        {
            0,0,1,0,0,
            0,0,1,0,0,
            0,0,1,0,0,
            0,0,1,0,0,
            0,0,1,0,0,
            }
 
    };

    vector<vector<bool>> added_memories_inputs = {
        {
            0,0,0,0,0,
            0,0,0,0,0,
            0,0,1,1,1,
            0,0,0,0,0,
            0,0,0,0,0,
            },
            {
            0,0,0,0,1,
            0,0,0,1,0,
            0,0,1,0,0,
            0,0,0,0,0,
            0,0,0,0,0,
            },
            {
            0,0,0,0,0,
            0,0,0,0,0,
            0,0,1,0,0,
            0,0,0,1,0,
            0,0,0,0,1,
            }
 
    };

    vector<vector<bool>> added_memories_targets = {
        {
            0,0,0,0,0,
            0,0,0,0,0,
            1,0,1,1,1,
            0,0,1,0,0,
            0,0,0,0,0,
            },
            {
            0,0,0,0,1,
            0,0,0,1,0,
            0,0,1,0,0,
            0,1,0,0,0,
            1,0,0,0,0
            },
            {
            1,0,0,0,0,
            0,1,0,0,0,
            0,0,1,0,0,
            0,0,0,1,0,
            0,0,0,0,1,
            }
 
    };


    int size_training_data = training_inputs.size();
    int size_added_data = added_memories_targets.size();
    int nb_sleep_episode = 110;

    vector<vector<double>> inputs_states(size_training_data);
    vector<vector<double>> targets_states(size_training_data);
    for (int i = 0; i < size_training_data; i++)
    {
        vector<double> state_input(network_size);
        vector<double> state_target(network_size);
        vector<double> added_memories_state_input(network_size);
        vector<double> added_memories_state_target(network_size);
        for (int j = 0; j < network_size; j++)
        {

            if (training_inputs[i][j])
            {
                state_input[j] = target_up_rate;
            }
            else
            {
                state_input[j] = target_down_rate;
            }

            if (training_targets[i][j])
            {
                state_target[j] = target_up_rate;
            }
            else
            {
                state_target[j] = target_down_rate;
            }
        }
        inputs_states[i] = state_input;
        targets_states[i] = state_target;
    }

    vector<vector<double>> added_memories_inputs_states(size_added_data);
    vector<vector<double>> added_memories_targets_states(size_added_data);
    for (int i = 0; i < size_added_data ; i++)
    {
        vector<double> state_input(network_size);
        vector<double> state_target(network_size);
        vector<double> added_memories_state_input(network_size);
        vector<double> added_memories_state_target(network_size);
        for (int j = 0; j < network_size; j++)
        {
            if (added_memories_inputs[i][j])
            {
                added_memories_state_input[j] = target_up_rate;
            }
            else
            {
                added_memories_state_input[j] = target_down_rate;
            }

            if (added_memories_targets[i][j])
            {
                added_memories_state_target[j] = target_up_rate;
            }
            else
            {
                added_memories_state_target[j] = target_down_rate;
            }
        }
        added_memories_inputs_states[i] = added_memories_state_input;
        added_memories_targets_states[i] = added_memories_state_target;
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

    // Training loop hopfield
    std::cout << "training initial set of memories" << std::endl;
    for (int i = 0; i < nb_iter_learning; i++)
    {
        net.reinforce_attractor_hebbian(training_targets[i % size_training_data], learning_rate_heb);
    }

    // Querying
    std::cout << "Querying learned memories" << std::endl;
    for (int i = 0; i < size_training_data; i++)
    {
        // std::cout << "NEW ITER" << std::endl;
        net.set_state(inputs_states[i]);
        // show_state_grid(net, 3);
        run_net_sim_noisy(net, nb_iter_sim, delta, 0.0, 0.01);
        // std::cout << "after convergence" << std::endl;
        show_state_grid(net, colum_size);
        // show_matrix(net.weight_matrix);
    }
    std::cout << "SLEEP" << std::endl;
    // Sleep to normalize
    std::vector<double> target_state;
    for (int i = 0; i < nb_sleep_episode ; i++){
        for (int r = 0; r < size_training_data*2; r++)
        {
            // std::cout << "dream number :" << std::endl;
            // std::cout << r << std::endl;

            net.set_state(vector<double>(network_size,0.5));
            run_net_sim_noisy_depressed(net, nb_iter_sim, delta, 0.0, 0.01); // Using utility function for noisy iterations
            target_state = assignStateToTopNValues(net.rate_list, nb_winners, target_up_rate, target_down_rate);
            net.pot_inhib(2);
            run_net_sim_noisy(net, nb_iter_sim, delta, 0.0, 0.01);           // Using utility function for noisy iterations
            // std::cout << "State after convergence:" << std::endl;
            // show_state_grid(net, 3); // Show state after convergence to check for attractor
            net.reinforce_attractor(target_state,learning_rate);

            // net.set_state(target_state);
            // show_state_grid(net, colum_size);
            // net.set_state(target_state);
            // show_state_grid(net, 3); 
            // std::cout << "target sum inhib weights vector :" << std::endl;
            // show_vector(net.target_sum_each_inhib);
            // std::cout << "actual vector before norm :" << std::endl;
            // show_vector(net.actual_sum_each_inhib);
            // net.iterative_normalize_inhib(100, 0.01);
            // std::cout << "target sum inhib weights vector :" << std::endl;
            // show_vector(net.target_sum_each_inhib);
            // std::cout << "actual vector after norm :" << std::endl;
            // show_vector(net.actual_sum_each_inhib);
        }
        net.reset_inhib();
    }
    // Querying
    std::cout << "Querying learned memories" << std::endl;
    for (int i = 0; i < size_training_data; i++)
    {
        // std::cout << "NEW ITER" << std::endl;
        net.set_state(inputs_states[i]);
        // show_state_grid(net, 3);
        run_net_sim_noisy(net, nb_iter_sim, delta, 0.0, 0.01);
        // std::cout << "after convergence" << std::endl;
        show_state_grid(net, colum_size);
        // show_matrix(net.weight_matrix);
    }

    std::cout << "training added set of memories" << std::endl;
    for (int added_i = 0; added_i < size_added_data; added_i++){

        for (int i = 0; i < nb_iter_learning; i++)
        {
            net.reinforce_attractor_hebbian(added_memories_targets[added_i], learning_rate_heb);
        }

        // // Querying
        // std::cout << "Querying added memories" << std::endl;
        // for (int i = 0; i < size_training_data; i++)
        // {
        //     // std::cout << "NEW ITER" << std::endl;
        //     net.set_state(added_memories_inputs_states[i]);
        //     // show_state_grid(net, 3);
        //     run_net_sim_noisy(net, nb_iter_sim, delta, 0.0, 0.01);
        //     // std::cout << "after convergence" << std::endl;
        //     show_state_grid(net, colum_size);
        //     // show_matrix(net.weight_matrix);
        // }

        // // Querying
        // std::cout << "Querying degraded initial memories" << std::endl;
        // for (int i = 0; i < size_training_data; i++)
        // {
        //     // std::cout << "NEW ITER" << std::endl;
        //     net.set_state(inputs_states[i]);
        //     // show_state_grid(net, 3);
        //     run_net_sim_noisy(net, nb_iter_sim, delta, 0.0, 0.01);
        //     // std::cout << "after convergence" << std::endl;
        //     show_state_grid(net, colum_size);
        //     // show_matrix(net.weight_matrix);
        // }


        std::cout << "SLEEP" << std::endl;
        // Sleep to normalize
        for (int i = 0; i < nb_sleep_episode ; i++){
            // std::cout << i << std::endl;
            for (int r = 0; r < (size_training_data+size_added_data)*2; r++)
            {
                // std::cout << "dream number :" << std::endl;
                // std::cout << r << std::endl;

                net.set_state(vector<double>(network_size,0.5));
                run_net_sim_noisy_depressed(net, nb_iter_sim, delta, 0.0, 0.01); // Using utility function for noisy iterations
                target_state = assignStateToTopNValues(net.rate_list, nb_winners, target_up_rate, target_down_rate);
                net.pot_inhib(2);
                run_net_sim_noisy(net, nb_iter_sim, delta, 0.0, 0.01);           // Using utility function for noisy iterations
                // std::cout << "State after convergence:" << std::endl;
                // show_state_grid(net, 3); // Show state after convergence to check for attractor
                net.reinforce_attractor(target_state,learning_rate);
                // if (i==0){
                //     net.set_state(target_state);
                //     show_state_grid(net, colum_size);
                // }
                // std::cout << "target sum inhib weights vector :" << std::endl;
                // show_vector(net.target_sum_each_inhib);
                // std::cout << "actual vector before norm :" << std::endl;
                // show_vector(net.actual_sum_each_inhib);
                // net.iterative_normalize_inhib(100, 0.01);
                // std::cout << "target sum inhib weights vector :" << std::endl;
                // show_vector(net.target_sum_each_inhib);
                // std::cout << "actual vector after norm :" << std::endl;
                // show_vector(net.actual_sum_each_inhib);
            }
            net.reset_inhib();
        }
    }

    // Querying
    std::cout << "Querying added memories" << std::endl;
    for (int i = 0; i < size_added_data; i++)
    {
        // std::cout << "NEW ITER" << std::endl;
        net.set_state(added_memories_inputs_states[i]);
        // show_state_grid(net, 3);
        run_net_sim_noisy(net, nb_iter_sim, delta, 0.0, 0.01);
        // std::cout << "after convergence" << std::endl;
        show_state_grid(net, colum_size);
        // show_matrix(net.weight_matrix);
    }

    // Querying
    std::cout << "Querying degraded initial memories" << std::endl;
    for (int i = 0; i < size_training_data; i++)
    {
        // std::cout << "NEW ITER" << std::endl;
        net.set_state(inputs_states[i]);
        // show_state_grid(net, 3);
        run_net_sim_noisy(net, nb_iter_sim, delta, 0.0, 0.01);
        // std::cout << "after convergence" << std::endl;
        show_state_grid(net, colum_size);
        // show_matrix(net.weight_matrix);
    }

    return 0;
}
