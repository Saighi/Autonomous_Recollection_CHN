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
        return a.first > b.first; // Min Heap based on the value of the double
    }
};

std::vector<double> assignStateToTopNValues(std::vector<double> &vec, int n, double winner_state, double loser_state)
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
    // displayPriorityQueue(pq);
    //  Create a new vector to store the final states
    std::vector<double> state_vector(vec.size(), loser_state);
    std::vector<bool> bool_vector(vec.size(), false);

    // Extract the indexes from the heap and set 'winner_state' for top 'n' elements
    while (!pq.empty())
    {
        state_vector[pq.top().second] = winner_state;
        bool_vector[pq.top().second] = true;
        pq.pop();
    }

    return state_vector;
}

std::vector<bool> assignBoolToTopNValues(std::vector<double> &vec, int n)
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
    // displayPriorityQueue(pq);
    //  Create a new vector to store the final states
    std::vector<bool> bool_vector(vec.size(), false);

    // Extract the indexes from the heap and set 'winner_state' for top 'n' elements
    while (!pq.empty())
    {
        bool_vector[pq.top().second] = true;
        pq.pop();
    }

    return bool_vector;
}

void show_vector_bool_grid(std::vector<bool> vec, int rows)
{
    int iter = 0;
    // std::cout << "activity :" << std::endl;
    // for (const auto &element : net.activity_list)
    // {
    //     if ((iter % (net.size / rows) == 0) && iter != 0)
    //     {
    //         std::cout << "" << std::endl;
    //     }

    //     std::cout << std::fixed << std::setprecision(2) << element << " ";
    //     iter ++;
    // }
    // iter = 0;
    // std::cout << "" << std::endl;
    std::cout << "rates :" << std::endl;
    for (const auto &element : vec)
    {
        if ((iter % (vec.size() / rows) == 0.0) && iter != 0)
        {
            std::cout << "" << std::endl;
        }
        // std::cout << element << " ";
        std::cout << element << " ";
        iter++;
    }
    std::cout << "" << std::endl;
}