#include "network.hpp"
#include <numeric>
#include <iostream>
#include <vector>

using namespace std;

void show_state(Network& net)
{
    std::cout << "activity :" << std::endl;
    for (const auto &element : net.activity_list)
    {
        std::cout << element << " ";
    }
    std::cout << "" << std::endl;
    std::cout << "rates :" << std::endl;
    for (const auto &element : net.rate_list)
    {
        std::cout << element << " ";
    }
    std::cout << "" << std::endl;
}

void show_state_grid(Network& net, int rows)
{
    int iter = 0;
    std::cout << "activity :" << std::endl;
    for (const auto &element : net.activity_list)
    {
        if ((iter % (net.size / rows) == 0) && iter != 0)
        {
            std::cout << "" << std::endl;
        }

        std::cout << element << " ";
        iter ++; 
    }
    iter = 0;
    std::cout << "" << std::endl;
    std::cout << "rates :" << std::endl;
    for (const auto &element : net.rate_list)
    {
        if ((iter % (net.size / rows) == 0.0) && iter != 0)
        {
            std::cout << "" << std::endl;
        }
        std::cout << element << " ";
        iter++;
    }
    std::cout << "" << std::endl;
}

void run_net_sim(Network& net, int nb_iter, double delta)
{
    for (int i = 0; i < nb_iter; i++)
    {
        net.iterate(delta);
    }
}

void run_net_sim_noisy(Network &net, int nb_iter, double delta, double mean, double stddev)
{
    for (int i = 0; i < nb_iter; i++)
    {
        net.noisy_iterate(delta, mean, stddev);
    }
}

void show_weight_matrix(Network& net)
{
    std::cout << "weight matrix :" << std::endl;
    for (const auto &row : net.weight_matrix)
    {
        for (const auto &element : row)
        {
            std::cout << element << " ";
        }
        std::cout << "" << std::endl;
    }
}