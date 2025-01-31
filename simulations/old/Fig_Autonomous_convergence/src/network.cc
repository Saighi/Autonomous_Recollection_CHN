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
    inhib_strenght = 10;

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

void Network::iterate_query_drive(double delta, double strength_drive, std::vector<double>& query_drives)
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
        activity_list[i] += delta*(derivative_activity_list[i]-(leak * activity_list[i])+strength_drive*(query_drives[i]-activity_list[i]));
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
            // the weight_matrix is added the inhib_strenght to allow a lack of inhib from the inhib matrix to be excitatory.
            // there is no added inhibstrenght if connectivity is not (maybe a better way).
            derivative_activity_list[i] += ((weight_matrix[i][j] + (inhib_strenght*connectivity_matrix[i][j]) - inhib_matrix[i][j]) * rate_list[j]);
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
    scale_inhib = std::vector<std::vector<int>>(size, std::vector<int>(size));
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            if (connectivity_matrix[i][j] == 1)
            {
                weight_matrix[i][j] = 0;
                inhib_matrix[i][j] = inhib_strenght;
                scale_inhib[i][j] = 1;
                sum_all_inhib += inhib_strenght;
                target_sum_each_inhib[j] += inhib_strenght;
                actual_sum_each_inhib[j] += inhib_strenght;
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

void Network::derivative_gradient_descent(std::vector<bool>& target_bin_state,std::vector<double>& target_rates,double target_drive,double learning_rate, double leak, std::vector<double>& drive_errors)
{
    double input = 0;
    double ui;
    double update;
    double unit_target_drive;
    double diff;
    for (int i = 0; i < size; i++)
    {
        for(int j = 0; j < size; j++){
            if (connectivity_matrix[i][j] == 1) {
                input += weight_matrix[i][j] * target_rates[j];
                }
        }
        // std::cout << input << std::endl;
        ui = input/leak;
        unit_target_drive = ((target_bin_state[i]*2)-1)*target_drive; // target drive of the unit 
        diff = unit_target_drive - ui;
        drive_errors[i] = diff;
        for(int j = 0; j < size; j++){
            if (connectivity_matrix[i][j] == 1) {
                update = learning_rate*2*diff*target_rates[j];
                weight_matrix[i][j] += update;
                weight_matrix[j][i] += update;
                }
        }
        input = 0;
    }
}

void Network::derivative_gradient_descent_with_momentum(std::vector<bool>& target_bin_state,
                                                      std::vector<double>& target_rates,
                                                      double target_drive,
                                                      double learning_rate,
                                                      double leak,
                                                      std::vector<double>& drive_errors,
                                                      std::vector<std::vector<double>>& velocity_matrix,
                                                      double momentum_coef)
{
    double input = 0;
    double ui;
    double update;
    double unit_target_drive;
    double diff;

    for (int i = 0; i < size; i++) {
        for(int j = 0; j < size; j++) {
            if (connectivity_matrix[i][j] == 1) {
                input += weight_matrix[i][j] * target_rates[j];
            }
        }

        ui = input/leak;
        unit_target_drive = ((target_bin_state[i]*2)-1)*target_drive;
        diff = unit_target_drive - ui;
        drive_errors[i] = diff;

        for(int j = 0; j < size; j++) {
            if (connectivity_matrix[i][j] == 1) {
                // Calculate gradient update
                update = learning_rate * 2 * diff * target_rates[j];

                // Apply momentum update
                velocity_matrix[i][j] = momentum_coef * velocity_matrix[i][j] + update;
                velocity_matrix[j][i] = velocity_matrix[i][j]; // Maintain symmetry

                // Update weights with momentum
                weight_matrix[i][j] += velocity_matrix[i][j];
                weight_matrix[j][i] = weight_matrix[i][j]; // Maintain symmetry
            }
        }
        input = 0;
    }
}


void Network::rate_derivative_gradient_descent(std::vector<double> target_rate, double learning_rate, double leak)
{
    double input = 0;
    double ui;
    double vi;
    double update;
    for (int i = 0; i < size; i++)
    {
        for(int j = 0; j < size; j++){
            if (connectivity_matrix[i][j] == 1) {
                input += weight_matrix[i][j] * target_rate[j];
                }
        }
        ui = input/leak;
        vi = transfer(ui);
        for(int j = 0; j < size; j++){
            if (connectivity_matrix[i][j] == 1) {
                update = (target_rate[i]-vi) * vi * (1 - vi) * target_rate[i] * learning_rate;
                weight_matrix[i][j] += update;
                weight_matrix[j][i] += update;
                }
        }
        input = 0;
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
                inhib_matrix[i][j] = inhib_strenght;
                sum_all_inhib += inhib_strenght;
                target_sum_each_inhib[j] += inhib_strenght;
                actual_sum_each_inhib[j] += inhib_strenght;
            }
        }
    }
}

void Network::pot_inhib_bin(double pot_rate, std::vector<bool> winners)
{
    actual_sum_each_inhib = std::vector<double>(size, 0);
    // Adjust weights
    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < size; ++j)
        {
            if (connectivity_matrix[i][j] == 1)
            {
                inhib_matrix[i][j] += pot_rate * (winners[j] * winners[i]);
                inhib_matrix[j][i] += pot_rate * (winners[j] * winners[i]);
            }
        }
    }
    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < size; ++j)
        {
            actual_sum_each_inhib[j] += inhib_matrix[i][j];
        }
    }
}

void Network::pot_inhib_bin_scale(double pot_rate, std::vector<bool> winners)
{
    actual_sum_each_inhib = std::vector<double>(size, 0);
    // Adjust weights
    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < size; ++j)
        {
            if (connectivity_matrix[i][j] == 1)
            {
                if (winners[i] && winners[j])
                {
                    inhib_matrix[i][j] += pot_rate * (winners[j] * winners[i]) * scale_inhib[i][j];
                    inhib_matrix[j][i] += pot_rate * (winners[j] * winners[i]) * scale_inhib[i][j];
                    scale_inhib[i][j] += 1;
                }
            }
        }
    }
    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < size; ++j)
        {
            actual_sum_each_inhib[j] += inhib_matrix[i][j];
        }
    }
}
