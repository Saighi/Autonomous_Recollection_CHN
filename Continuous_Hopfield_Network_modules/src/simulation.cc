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
    int network_size = 41;
    int col_with = int(sqrt(network_size));
    double leak = 1.5;
    double delta = 0.2;

    // Simulation constants
    // double value_convergence = 0.01;
    // int nb_iter_sim = 200;

    // Learning constants
    double target_up_rate = 0.8;
    double target_down_rate = 0.2;
    double learning_rate = 0.08;
    int nb_winners = col_with; // number of winning neurons

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

    vector<vector<int>> modules = { {0, 1},
                                    {2, 3, 4},
                                    {5, 6, 7, 8, 9, 10, 11, 12, 13},
                                    {14, 15, 16, 17, 18, 19, 20, 21, 22},
                                    {23, 24, 25, 26, 27, 28, 29, 30, 31},
                                    {32, 33, 34, 35, 36, 37, 38, 39, 40} };
    vector<int> nb_winner_per_module = {1,1,3,3,3,3};

    vector<double> new_target_state;
    int max_iter = 1;
    vector<bool> winning_units;
    for (int iter = 0; iter < max_iter; iter++)
    {
        std::cout << iter << std::endl;

        for (int r = 0; r < 10; r++)
        {
            // std::cout << "NEW ITER" << std::endl;
            net.set_state(vector<double>(network_size, 0.5));
            // std::cout << "Initial random state:" << std::endl;
            // show_state_grid(net, 3); // Show initial state
 
            // Let the network converge
            run_net_sim_noisy_depressed(net, 1000, delta, 0.0, 0.01); // Using utility function for noisy iterations
            //run_net_sim_noisy(net, 500, delta, 0.0, 0.01);           // Using utility function for noisy iterations
            
            winning_units = assignBoolToTopNValues(net.activity_list,nb_winners);
            if (iter == max_iter-1 || iter == max_iter-2)
            {
                std::cout << "State after convergence:" << std::endl;
                show_state_grid(net, col_with);
                show_vector_bool_grid(winning_units,col_with);
            }
            
            net.pot_inhib_exponential_bin(0.5,0,winning_units); // works with 0.005
            new_target_state = assignStateToTopNValues(net.activity_list, nb_winners, target_up_rate, target_down_rate);
            // if (iter == max_iter){
            //     show_vector(new_target_state);
            // }
            net.reinforce_attractor(new_target_state, learning_rate);
            // show_state_grid(net, col_with); // Show state after convergence to check for attractor
            // std::cout << "target sum inhib weights vector :" << std::endl;
            // show_vector(net.target_sum_each_inhib);
            // std::cout << "actual vector before norm :" << std::endl;
            // show_vector(net.actual_sum_each_inhib);
            // std::cout << "target sum inhib weights vector :" << std::endl;
            // show_vector(net.target_sum_each_inhib);
            // std::cout << "actual vector after norm :" << std::endl;
            // show_vector(net.actual_sum_each_inhib);
        }
        net.reset_inhib();
    }

    return 0;
}
