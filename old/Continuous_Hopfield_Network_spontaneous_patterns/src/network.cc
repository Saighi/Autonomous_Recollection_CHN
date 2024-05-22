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
    inhib_strength = 10;

    random_init(-0.3,0.3);
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
        activity_list[i] += delta * (derivative_activity_list[i] - (leak * activity_list[i]));
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
            // the weight_matrix is added the inhib_strength to allow a lack of inhib from the inhib matrix to be excitatory.
            // there is no added inhibstrenght if connectivity is not (maybe a better way).
            derivative_activity_list[i] += ((weight_matrix[i][j] + (inhib_strength*connectivity_matrix[i][j]) - inhib_matrix[i][j]) * rate_list[j]);
        }
        derivative_activity_list[i] += noise; // Adding noise to each derivative element outside the inner loop
    }

    for (int i = 0; i < size; i++)
    {
        activity_list[i] += delta * (derivative_activity_list[i] - (leak * activity_list[i]));
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

    weight_matrix = std::vector<std::vector<double>>(size, std::vector<double>(size));
    inhib_matrix = std::vector<std::vector<double>>(size, std::vector<double>(size));
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            if (connectivity_matrix[i][j] == 1)
            {
                weight_matrix[i][j] = 0;
                inhib_matrix[i][j] = inhib_strength;
                sum_all_inhib += inhib_strength;
                target_sum_each_inhib[j] += inhib_strength;
                actual_sum_each_inhib[j] += inhib_strength;
            }
        }
    }
}

// random initialisation of weight matrix
void Network::random_init(double x, double y)
{
    std::random_device rd; // Non-deterministic random device for seeding
    std::mt19937 gen(rd()); // Mersenne Twister pseudo-random generator of 32-bit numbers

    // Ensure x < y to define a proper range [x, y] for uniform_real_distribution
    if (x >= y) {
        std::swap(x, y); // Swap x and y if x is not less than y
    }

    std::uniform_real_distribution<> dis(x, y); // Uniform real distribution in [x, y]

    sum_all_inhib = 0;
    activity_list = std::vector<double>(size, 0.0);
    rate_list = std::vector<double>(size, transfer(0.0));
    derivative_activity_list = std::vector<double>(size, 0.0);
    target_sum_each_inhib = std::vector<double>(size, 0.0);
    actual_sum_each_inhib = std::vector<double>(size, 0.0);

    weight_matrix = std::vector<std::vector<double>>(size, std::vector<double>(size));
    inhib_matrix = std::vector<std::vector<double>>(size, std::vector<double>(size));
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            if (connectivity_matrix[i][j] == 1)
            {
                weight_matrix[i][j] = dis(gen); // Assign a random value between x and y
                inhib_matrix[i][j] = inhib_strength;
                sum_all_inhib += inhib_strength;
                target_sum_each_inhib[j] += inhib_strength;
                actual_sum_each_inhib[j] += inhib_strength;
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

// No normalization, doesn't keep the sum of weight of synapses stable.
void Network::pot_inhib(double pot_rate)
{
    actual_sum_each_inhib = std::vector<double>(size,0);
    // Adjust weights
    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < size; ++j)
        {
            if (connectivity_matrix[i][j] == 1)
            {
                inhib_matrix[i][j] += pot_rate*(rate_list[j]*rate_list[i]);
                inhib_matrix[j][i] += pot_rate*(rate_list[j]*rate_list[i]);
            }
        }
    }
    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < size; ++j){
            actual_sum_each_inhib[j] += inhib_matrix[i][j];
        }
    }
}

void Network::pot_inhib_bin(double pot_rate, std::vector<bool> winners)
{
    actual_sum_each_inhib = std::vector<double>(size,0);
    // Adjust weights
    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < size; ++j)
        {
            if (connectivity_matrix[i][j] == 1)
            {
                inhib_matrix[i][j] += pot_rate*(winners[j]*winners[i]);
                inhib_matrix[j][i] += pot_rate*(winners[j]*winners[i]);
            }
        }
    }
    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < size; ++j){
            actual_sum_each_inhib[j] += inhib_matrix[i][j];
        }
    }
}

void Network::pot_inhib_exponential_bin(double pot_rate, double scaling, std::vector<bool> winners)
{
    actual_sum_each_inhib = std::vector<double>(size,0);
    // Adjust weights
    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < size; ++j)
        {
            if (connectivity_matrix[i][j] == 1)
            {
                inhib_matrix[i][j] += pot_rate*(winners[j]*winners[i])*(1+(inhib_matrix[i][j]-inhib_strength)*(scaling/pot_rate));
                inhib_matrix[j][i] += pot_rate*(winners[j]*winners[i])*(1+(inhib_matrix[i][j]-inhib_strength)*(scaling/pot_rate));
            }
        }
    }
    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < size; ++j){
            actual_sum_each_inhib[j] += inhib_matrix[i][j];
        }
    }
}

void Network::iterative_normalize(int nb_iter_normalize, double rate_normalize){
    std::vector<double> new_sum_each_inhib(size,0);
    for(int iter = 0; iter<nb_iter_normalize; iter++ ){
        for (int i = 0; i < size; ++i)
        {
            for (int j = 0; j < size; ++j)
            {
                if (connectivity_matrix[i][j] == 1)
                {
                    inhib_matrix[i][j] += (target_sum_each_inhib[j]-actual_sum_each_inhib[j])*rate_normalize;
                    inhib_matrix[j][i] += (target_sum_each_inhib[j] - actual_sum_each_inhib[j]) * rate_normalize;
                }
            }
        }
        for (int i = 0; i < size; ++i)
        {
            for (int j = 0; j < size; ++j)
            {
                new_sum_each_inhib[j] += inhib_matrix[i][j];
            }
        }
        for (int i = 0; i < size; ++i){
            actual_sum_each_inhib[i] = new_sum_each_inhib[i];
            new_sum_each_inhib[i] = 0;
        }
    }
}

void Network::reset_inhib()
{
    sum_all_inhib = 0;
    target_sum_each_inhib = std::vector<double>(size, 0.0);
    actual_sum_each_inhib = std::vector<double>(size, 0.0);
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            if (connectivity_matrix[i][j] == 1)
            {
                inhib_matrix[i][j] = inhib_strength;
                sum_all_inhib += inhib_strength;
                target_sum_each_inhib[j] += inhib_strength;
                actual_sum_each_inhib[j] += inhib_strength;
            }
        }
    }
}