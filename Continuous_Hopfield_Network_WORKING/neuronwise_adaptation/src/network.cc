#include "network.hpp"
#include "utils.hpp"
#include <vector>
#include <cmath>
#include <iostream>
#include <random>

Network::Network(std::vector<std::vector<bool>> connect_mat, int size_network, double lk)
{
    leak = lk;
    connectivity_matrix = connect_mat;
    size = size_network;

    blank_init();
}

void Network::iterate(double delta)
{

    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            // if (connectivity_matrix[i][j]==true){ // Will have to verify that the weight stay 0 during the weight uptdate
            //  when connectivity_matrix is false
            derivative_activity_list[i] += weight_matrix[i][j] * rate_list[j];
            //}
        }
    }

    for (int i = 0; i < size; i++)
    {
        activity_list[i] += delta * (derivative_activity_list[i] - (leak * activity_list[i]));
        rate_list[i] = transfer(activity_list[i]);
    };

    std::fill(derivative_activity_list.begin(), derivative_activity_list.end(), 0);
}

// chatgpt add some code, the corresponding includes, the corresponding parameters to add some gaussian noise 
// to the derivative such that when we iterate over the network through this function it does some kind of annealing
// add the function definition to network.hpp
void Network::noisy_iterate(double delta, double mean, double stddev)
{
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(mean, stddev); // Gaussian noise with user-specified mean and stddev

    for (int i = 0; i < size; i++)
    {
        double noise = distribution(generator); // Generate Gaussian noise once per iteration for each element
        for (int j = 0; j < size; j++)
        {
            // Apply the noise outside the inner loop, directly influencing each derivative element only once
            derivative_activity_list[i] += (weight_matrix[i][j] * rate_list[j]);
        }
        derivative_activity_list[i] += noise; // Adding noise to each derivative element outside the inner loop
    }

    for (int i = 0; i < size; i++)
    {
        activity_list[i] += delta * (derivative_activity_list[i] - ((leak + depression[i]) * activity_list[i]));
        rate_list[i] = transfer(activity_list[i]);
    };

    std::fill(derivative_activity_list.begin(), derivative_activity_list.end(), 0);
}

void Network::noisy_depression_iterate(double delta, double mean, double stddev)
{
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(mean, stddev); // Gaussian noise with user-specified mean and stddev

    for (int i = 0; i < size; i++)
    {
        double noise = distribution(generator); // Generate Gaussian noise once per iteration for each element
        for (int j = 0; j < size; j++)
        {
            // Apply the noise outside the inner loop, directly influencing each derivative element only once
            // the weight_matrix is added the inhib_strenght to allow a lack of inhib from the inhib matrix to be excitatory.
            // there is no added inhibstrenght if connectivity is not (maybe a better way).
            derivative_activity_list[i] += weight_matrix[i][j] * rate_list[j];
        }
        derivative_activity_list[i] += noise; // Adding noise to each derivative element outside the inner loop
    }

    for (int i = 0; i < size; i++)
    {
        activity_list[i] += delta * (derivative_activity_list[i] - ((leak + depression[i]) * activity_list[i]));
        rate_list[i] = transfer(activity_list[i]);
    };

    std::fill(derivative_activity_list.begin(), derivative_activity_list.end(), 0);
}

double Network::transfer(double activation)
{
    return 1.0 / (1.0 + std::exp(-activation));
}

double Network::transfer_inverse(double activation)
{
    return -std::log(-1.0 + 1.0 / activation);
}

// blank initialisation of weight matrix
void Network::blank_init()
{
    sum_all_inhib = 0;
    activity_list = std::vector<double>(size, 0.0);
    rate_list = std::vector<double>(size, transfer(0.0));
    derivative_activity_list = std::vector<double>(size, 0.0);
    target_sum_each_inhib = std::vector<double>(size, 0.0);
    actual_sum_each_inhib = std::vector<double>(size, 0.0);
    depression = std:: vector<double>(size,0.0);

    weight_matrix = std::vector<std::vector<double>>(size, std::vector<double>(size));

    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            if (connectivity_matrix[i][j] == 1)
            {
                weight_matrix[i][j] = 0;
            }
        }
    }
}

void Network::set_state(std::vector<double> new_state){
    for (int i = 0; i < size; i++)
    {
        rate_list[i] = new_state[i];
        activity_list[i] = transfer_inverse(rate_list[i]);
    }
}

void Network::reinforce_attractor(std::vector<double> target_state, double learning_rate)
{
    for (int i = 0; i < size; i++)
    {
        for(int j = 0; j < size; j++){
            if (connectivity_matrix[i][j] == 1) {
                double update = (target_state[j] - rate_list[j]) * target_state[i] * learning_rate;
                weight_matrix[i][j] += update;
                weight_matrix[j][i] += update;
                }
        }
    }
}

void Network::pot_depress_bin(double pot_rate, std::vector<bool> winners)
{
    // Adjust weights
    for (int i = 0; i < size; ++i)
    {
        depression[i] += pot_rate* [i];
    }
}

void Network::reset_depression()
{
    depression = std::vector<double>(size, 0.0);
}
