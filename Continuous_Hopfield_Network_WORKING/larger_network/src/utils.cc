#include "network.hpp"
#include <numeric>
#include <iostream>
#include <vector>
#include <queue>
#include <utility>
#include <algorithm>
#include <fstream>
#include <filesystem>
#include <string>
#include <unordered_set>
#include <cstdlib>
#include <ctime>
#include <random>
#include <fstream>
#include <sstream>

void writeToCSV(std::ofstream &file, const std::vector<double> &data)
{
    for (size_t i = 0; i < data.size(); ++i)
    {
        file << data[i];
        if (i != data.size() - 1)
        {
            file << " ";
        }
    }
    file << "\n"; // Add a new line after each vector
}

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

void run_net_sim_noisy_depressed_save(Network &net, int nb_iter, double delta, double mean, double stddev, std::ofstream &file)
{
    for (int i = 0; i < nb_iter; i++)
    {
        net.noisy_depression_iterate(delta, mean, stddev);
        writeToCSV(file, net.rate_list);
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

// Function to generate linspace equivalent
std::vector<double> linspace(double start, double end, int num)
{
    std::vector<double> result;
    if (num <= 0)
    {
        return result;
    }
    if (num == 1)
    {
        result.push_back(start);
        return result;
    }

    double step = (end - start) / (num - 1);
    for (int i = 0; i < num; ++i)
    {
        result.push_back(start + i * step);
    }
    return result;
}

// Function to generate a base pattern with a specified number of 1s
std::vector<bool> generateBasePattern(int N, int nb_winning_units)
{
    std::vector<bool> basePattern(N, false);
    for (int i = 0; i < nb_winning_units; ++i)
    {
        basePattern[i] = true;
    }

    return basePattern;
}

// Function to flip bits in a balanced pattern based on noise level
std::vector<bool> generateNoisyBalancedPattern(const std::vector<bool> &basePattern, int numFlips)
{
    std::vector<bool> noisyPattern = basePattern;
    int N = basePattern.size();
    std::unordered_set<int> flipIndices;

    // Ensure numFlips is valid
    int maxFlips = std::min(numFlips, N / 2);

    // Flip numFlips bits from 1 to 0
    while (flipIndices.size() < maxFlips)
    {
        int index = rand() % N;
        if (noisyPattern[index] == true && flipIndices.find(index) == flipIndices.end())
        {
            noisyPattern[index] = !noisyPattern[index];
            flipIndices.insert(index);
        }
    }

    // flipIndices.clear();

    // Flip numFlips bits from 0 to 1
    while (flipIndices.size() < maxFlips * 2)
    {
        int index = rand() % N;
        if (noisyPattern[index] == false && flipIndices.find(index) == flipIndices.end())
        {
            noisyPattern[index] = !noisyPattern[index];
            flipIndices.insert(index);
        }
    }

    return noisyPattern;
}

// Function to check if a pattern already exists in a vector of patterns
bool patternExists(const std::vector<std::vector<bool>> &patterns, const std::vector<bool> &pattern)
{
    for (const auto &p : patterns)
    {
        if (p == pattern)
        {
            return true;
        }
    }
    return false;
}

// Function to generate K unique noisy balanced patterns
std::vector<std::vector<bool>> generatePatterns(int K, int N, int nb_winning_units, double noiseLevel)
{
    std::vector<std::vector<bool>> patterns;
    std::vector<bool> basePattern = generateBasePattern(N, nb_winning_units);
    int numFlips = static_cast<int>(noiseLevel * (N / 2)); // Calculate number of flips based on noise level

    while (patterns.size() < K)
    {
        std::vector<bool> newPattern = generateNoisyBalancedPattern(basePattern, numFlips);
        if (!patternExists(patterns, newPattern))
        {
            patterns.push_back(newPattern);
        }
    }

    return patterns;
}

std::vector<std::vector<bool>> loadPatterns(const std::string &filename)
{
    std::ifstream file(filename);
    std::string line;
    std::vector<std::vector<bool>> boolVectors;

    if (file.is_open())
    {
        while (getline(file, line))
        {
            std::istringstream iss(line);
            std::vector<bool> boolVector;
            std::string value;

            while (iss >> value)
            {
                boolVector.push_back(value == "1");
            }
            boolVectors.push_back(boolVector);
        }
        file.close();
    }
    else
    {
        std::cerr << "Unable to open file" << std::endl;
    }

    return boolVectors;
}