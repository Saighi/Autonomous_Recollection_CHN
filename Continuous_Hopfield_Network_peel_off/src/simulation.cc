#include "network.hpp"
#include "utils.hpp"
#include <numeric>
#include <iostream>
#include <vector>
#include <random>
#include <cmath> // Include for sqrt
#include <algorithm> // For std::nth_element

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
    int nb_iter_sim = 400;

    // Learning constants
    int nb_iter_learning = 1200;
    double target_up_rate = 0.75;
    double target_down_rate = 0.25;
    double learning_rate = 0.05;
    double learning_rate_heb = 0.001;
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
            // {
            // 0,0,0,0,1,
            // 0,0,0,1,0,
            // 0,0,1,0,0,
            // 0,0,0,0,0,
            // 0,0,0,0,0,
            // },
            // {
            // 0,0,0,0,0,
            // 0,0,0,0,0,
            // 0,0,1,0,0,
            // 0,0,0,1,0,
            // 0,0,0,0,1,
            // }
 
    };

    vector<vector<bool>> added_memories_targets = {
        {
            0,0,0,0,0,
            0,0,0,0,0,
            1,1,1,1,1,
            0,0,0,0,0,
            0,0,0,0,0,
            },
            // {
            // 0,0,0,0,1,
            // 0,0,0,1,0,
            // 0,0,1,0,0,
            // 0,1,0,0,0,
            // 1,0,0,0,0
            // },
            // {
            // 1,0,0,0,0,
            // 0,1,0,0,0,
            // 0,0,1,0,0,
            // 0,0,0,1,0, 
            // 0,0,0,0,1,
            // }
 
    };


    int size_training_data = training_inputs.size();
    int size_added_data = added_memories_targets.size();
    // int nb_sleep_episode = 2400;

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


    // adding engram
    // for (int i = 0; i < size_training_data; i++)
    // {
    //     net.add_engram(training_targets[i]);
    // }

    // Querying
    // std::cout << "Querying learned memories" << std::endl;
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
    int nb_sleep_episode = 100;
    std::vector<bool> explored_state;
    std::vector<bool> excited_units;
    std::vector<int> actual_overlay;
    std::vector<std::vector<int>> actual_overlay_synapses;
    std::vector<std::vector<bool>> depressed_synapses;
    int cpt = 0;

    for (int i = 0; i < size_training_data; i++)
    {
        cpt+=1;
        for (int j = 0; j < nb_sleep_episode; j++){
            actual_overlay = net.overlay_vec;
            actual_overlay_synapses = net.synapses_overlay_matrix;
            net.set_state(targets_states[i]);
            run_net_sim_noisy(net, nb_iter_sim, delta, 0.0, 0.01); // Using utility function for noisy iterations
            net.reinforce_attractor(targets_states[i], learning_rate);
            explored_state = training_targets[i];
            for (int k = 0; k<cpt; k++)
            {
                excited_units = take_peal(actual_overlay, explored_state);
                actual_overlay = peal_off(actual_overlay, excited_units);
                // show_vector_bool_grid(excited_units,5);
                // show_vector_int_grid(actual_overlay, 5);
                actual_overlay_synapses = peal_off_syn(actual_overlay_synapses, explored_state);
                depressed_synapses = toBinaryMatrix(actual_overlay_synapses);
                if(j==1){
                    show_matrix_bool(depressed_synapses);
                    std::cout << "the overlay" << std::endl;
                    show_matrix_int(actual_overlay_synapses);
                }
                // net.set_state(vector<double>(network_size, 0.5));
                // run_net_sim_noisy_depressed(net, nb_iter_sim, delta, 0.0, 0.01); // Using utility function for noisy iterations
                // target_state = assignStateToTopNValues(net.rate_list, nb_winners, target_up_rate, target_down_rate);
                // target_bool = assignBoolToTopNValues(net.rate_list, nb_winners, target_up_rate, target_down_rate);
                // run_net_sim_noisy(net, nb_iter_sim, delta, 0.0, 0.01); // Using utility function for noisy iterations
                // net.reinforce_attractor(target_state, learning_rate);
            }

        }
        net.add_overlay(training_targets[i]);
        net.add_synapse_overlay(training_targets[i]);
        std::cout << "new neuron overlay : " << std::endl;
        show_vector_int_grid(net.overlay_vec, 5);
        std::cout << "the syn overlay" << std::endl;
        show_matrix_int(net.synapses_overlay_matrix);
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
        show_state_grid_rate(net,5);
        // show_state_grid(net, 5, 5);
        // show_matrix(net.weight_matrix);
    }


    return 0;
}
