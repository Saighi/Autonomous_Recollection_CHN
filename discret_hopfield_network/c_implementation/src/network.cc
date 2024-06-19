#include "network.hpp"
#include "utils.hpp"
#include <vector>
#include <cmath>
#include <iostream>
#include <random>

Network::Network(std::vector<std::vector<bool>> connect_mat, int size_network)
{
    connectivity_matrix = connect_mat;
    size = size_network;
    blank_init();
}

// blank initialisation of weight matrix
void Network::blank_init()
{

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

std::vector<double> Network::activate(const std::vector<double> &state)
{
    std::vector<double> new_state(state.size());
    for (size_t i = 0; i < state.size(); ++i)
    {
        new_state[i] = state[i] >= 0 ? 1 : -1;
    }
    return new_state;
}

std::vector<double> Network::run(const std::vector<double> &initial_state, int steps)
{
    std::vector<double> state = initial_state;
    for (int i = 0; i < steps; ++i)
    {
        std::vector<double> new_state(size, 0.0);
        for (int j = 0; j < size; ++j)
        {
            for (int k = 0; k < size; ++k)
            {
                new_state[j] += weight_matrix[j][k] * state[k];
            }
        }
        state = activate(new_state);
    }
    return state;
}

std::vector<double> Network::runReal(const std::vector<double> &initial_state, int nb_steps)
{
    std::vector<double> state = initial_state;
    
    for (int step = 0; step<nb_steps; step++){
        for (int i = 0; i < state.size(); ++i)
        {
            
            double sum = 0.0;
            for (int j = 0; j < size; ++j)
            {
                if (j != i)
                {
                    sum += weight_matrix[i][j] * state[j];
                }
            }

            state[i] = sum >= 0 ? 1 : -1;
        }

        return state;
    }

    }

void Network::trainPerceptron(const std::vector<std::vector<double>> &patterns, double step_size)
{
    for (const auto &pattern : patterns)
    {
        for (int i = 0; i < size; ++i)
        {
            double sum_in = 0.0;
            for (int j = 0; j < size; ++j)
            {
                if (j != i)
                {
                    sum_in += pattern[j] * weight_matrix[j][i];
                }
            }
            for (int j = 0; j < size; ++j)
            {
                if (j != i)
                {
                    double update = ((1 - (sum_in * pattern[i])) * pattern[i] * pattern[j]) * (1.0 / size);
                    weight_matrix[j][i] += update;
                    weight_matrix[i][j] += update;
                }
            }
        }
    }
}

void Network::trainPerceptronInput(const std::vector<std::vector<double>> &patterns, int index_input)
{
    for (const auto &pattern : patterns)
    {
        for (int i = index_input; i < size; ++i)
        {
            double sum_in = 0.0;
            for (int j = 0; j < index_input; ++j)
            {
                if (j != i)
                {
                    sum_in += pattern[j] * weight_matrix[j][i];
                }
            }
            for (int j = 0; j < index_input; ++j)
            {
                if (j != i)
                {
                    double update = ((1 - (sum_in * pattern[i])) * pattern[i] * pattern[j]) * (1.0 / size);
                    weight_matrix[j][i] += update;
                    weight_matrix[i][j] += update;
                }
            }
        }
    }
}

void Network::train(const std::vector<std::vector<double>> &patterns)
{
    for (const auto &p : patterns)
    {
        for (int i = 0; i < size; ++i)
        {
            for (int j = 0; j < size; ++j)
            {
                if (i != j)
                {
                    weight_matrix[i][j] += p[i] * p[j];
                    weight_matrix[j][i] += p[i] * p[j];
                }
            }
        }
    }
}

std::vector<double> Network::runAndClamp(const std::vector<double> &initial_states, int index_clamping, int steps)
{
    std::vector<double> all_state(size, 0.0);
    for (int i = 0; i < index_clamping; i++)
    {
        all_state[i] = initial_states[i];
    }

    for (int i = 0; i < steps; ++i)
    {
        for (int j = index_clamping; j < size; ++j)
        {
            for (int k = 0; k < index_clamping; ++k)
            {
                all_state[j] += weight_matrix[j][k] * all_state[k];
            }
        }
        all_state = activate(all_state);
    }
    return all_state;
}