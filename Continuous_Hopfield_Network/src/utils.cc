#include "network.hpp"
#include <numeric>
#include <iostream>
#include <vector>
#include <queue>
#include <utility>
#include <algorithm>

using namespace std;

// VIZUALISATION

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

void show_matrix(std::vector<std::vector<double>> matrix)
{
    for (const auto &row : matrix)
    {
        for (const auto &element : row)
        {
            std::cout << element << " ";
        }
        std::cout << "" << std::endl;
    }
}

void show_vector(std::vector<double> vector)
{
    for (const auto &element : vector)
    {
        std::cout << element << " ";
    }
    std::cout << "" << std::endl;
}

// SIMULATION

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

void run_net_sim_noisy_depressed(Network &net, int nb_iter, double delta, double mean, double stddev)
{
    for (int i = 0; i < nb_iter; i++)
    {
        net.noisy_depression_iterate(delta, mean, stddev);
    }
}

// TOOLS

// Comparator for priority queue
struct Compare
{
    bool operator()(const std::pair<double, int> &a, const std::pair<double, int> &b)
    {
        return a.first < b.first; // Min Heap based on the value of the double
    }
};

std::vector<int> findTopNIndexes(const std::vector<double> &vec, int n)
{
    std::priority_queue<std::pair<double, int>, std::vector<std::pair<double, int>>, Compare> pq;

    // Iterate through the vector and maintain a heap of top 'n' elements
    for (int i = 0; i < vec.size(); ++i)
    {
        pq.push(std::make_pair(vec[i], i));
        // If the heap size exceeds 'n', pop the smallest element out
        if (pq.size() > n)
        {
            pq.pop();
        }
    }

    // Extract the indexes from the heap
    std::vector<int> indexes;
    while (!pq.empty())
    {
        indexes.push_back(pq.top().second);
        pq.pop();
    }

    // The indexes will be in reverse order because the last inserted element is at the top
    // If you need them in ascending order, reverse the vector
    std::reverse(indexes.begin(), indexes.end());

    return indexes;
}