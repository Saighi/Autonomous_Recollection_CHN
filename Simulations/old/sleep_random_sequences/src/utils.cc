#include "network.hpp"
#include <numeric>
#include <iostream>
#include <vector>
#include <queue>
#include <utility>
#include <algorithm>
#include <fstream>
#include <string>
#include <random>


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

void appendToCSV(const std::vector<double>& data, const std::string& filename) {
    std::ofstream file(filename, std::ios::app); // Open file in append mode
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }
    
    for (size_t i = 0; i < data.size(); ++i) {
        file << data[i];
        if (i != data.size() - 1) {
            file << " ";
        }
    }
    file << "\n"; // Add a new line after each vector
    
    file.close();
}

// Function to generate a random binary sequence of length n with x ones
std::vector<bool> generateRandomBinarySequenceWithOnes(int n, int x) {
    if (x > n) {
        std::cerr << "Error: Number of ones cannot exceed sequence length." << std::endl;
        exit(1);
    }

    std::vector<bool> sequence(n, false); // Initialize sequence with zeros
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // Set x random elements to true
    for (int i = 0; i < x; ++i) {
        sequence[i] = true;
    }

    // Shuffle the sequence to randomize the position of ones
    std::shuffle(sequence.begin(), sequence.end(), gen);

    return sequence;
}

// Function to check if two sequences have overlap less than or equal to O
bool hasOverlap(const std::vector<bool>& seq1, const std::vector<bool>& seq2, int O) {
    int overlapCount = 0;
    int n = seq1.size();

    for (int i = 0; i < n; ++i) {
        if (seq1[i] && seq2[i]) {
            overlapCount++;
            if (overlapCount > O) {
                return true; // Overlap exceeds threshold
            }
        }
    }

    return false; // Overlap is within threshold
}

// Define a type alias for a boolean sequence
using BoolSequence = std::vector<bool>;

// Function to check if two boolean sequences are equal
bool areSequencesEqual(const std::vector<bool>& seq1, const std::vector<bool>& seq2) {
    if (seq1.size() != seq2.size()) {
        return false;
    }
    for (size_t i = 0; i < seq1.size(); ++i) {
        if (seq1[i] != seq2[i]) {
            return false;
        }
    }
    return true;
}

// Function to count the number of common sequences between two vector of boolean sequences
int countCommonSequences(const std::vector<std::vector<bool>>& vec1, const std::vector<std::vector<bool>>& vec2) {
    int commonCount = 0;
    for (const auto& seq1 : vec1) {
        for (const auto& seq2 : vec2) {
            if (areSequencesEqual(seq1, seq2)) {
                ++commonCount;
                break; // Move to the next sequence in vec1
            }
        }
    }
    return commonCount;
}