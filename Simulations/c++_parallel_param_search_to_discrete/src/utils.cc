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
#include <unordered_map>

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

void writeBoolToCSV(std::ofstream &file, const std::vector<bool> &data)
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

void run_net_sim_query_drive(Network& net, std::vector<double>& query_drives, double strength_drive,int nb_iter, double delta)
{
    for (int i = 0; i < nb_iter; i++)
    {
        net.iterate_query_drive(delta, strength_drive, query_drives);
    }
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
    int index;
    int cpt = 0 ;
    // Flip numFlips bits from 1 to 0
    while (cpt < numFlips) // can loop a long time if not enough are 1s
    {
        index = rand() % N; // trying to find a 1 index
        if (noisyPattern[index] == true)
        {
            noisyPattern[index] = !noisyPattern[index]; // turning the 1 into 0
            index = rand() % N;
            while (noisyPattern[index] != false){ // can loop a long time if not enough are 0s
                index = rand() % N; // trying to find a 0 index
            }
            noisyPattern[index] = !noisyPattern[index]; // turning the 0 into 1
            cpt+=1;

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
    int numFlips = static_cast<int>(noiseLevel*nb_winning_units); // Calculate number of flips based on noise level

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

// Helper function to compare vectors of bool
bool areVectorsEqual(const std::vector<bool> &v1, const std::vector<bool> &v2)
{
    return v1 == v2;
}

// Function to create a text file with parameter values
void createParameterFile(const std::string &directory, const std::unordered_map<std::string, double> &parameters)
{
    // Create the directory if it doesn't exist
    std::filesystem::create_directories(directory);

    // Create the full path to the file
    std::string filePath = directory + "/parameters.data";

    // Open a file stream to write the parameters
    std::ofstream outFile(filePath);

    // Check if the file was opened successfully
    if (!outFile)
    {
        std::cerr << "Error: Could not create file " << filePath << std::endl;
        return;
    }

    // Write each parameter to the file
    for (const auto &param : parameters)
    {
        outFile << param.first << "=" << param.second << "\n";
    }

    // Close the file stream
    outFile.close();

}

std::vector<double> pattern_as_states(double up_rate, double down_rate, std::vector<bool> bin_pattern)
{
    std::vector<double> state_input(bin_pattern.size());
    for (int j = 0; j < state_input.size(); j++)
        {
            if (bin_pattern[j])
            {
                state_input[j] = up_rate;
            }
            else
            {
                state_input[j] = down_rate;
            }
        }
    return state_input;
}

std::vector<std::vector<double>> patterns_as_states(double up_rate, double down_rate, std::vector<std::vector<bool>> bin_patterns)
{
    std::vector<std::vector<double>> initial_patterns_state_list(bin_patterns.size());
    for (int i = 0; i < bin_patterns.size(); i++)
    {
        initial_patterns_state_list[i] = pattern_as_states(up_rate,down_rate,bin_patterns[i]);
    }
    return initial_patterns_state_list;
}


// Helper function to generate all combinations of parameters
// funny function generated by chatgpt, you can use division and module to do combinatorials tricks
std::vector<std::unordered_map<std::string, double>> generateCombinations(const std::unordered_map<std::string, std::vector<double>> &varying_params)
{
    std::vector<std::unordered_map<std::string, double>> combinations;

    // Calculate the total number of combinations
    size_t total_combinations = 1;
    for (const auto &param : varying_params)
    {
        total_combinations *= param.second.size();
    }

    // Generate all combinations
    for (size_t i = 0; i < total_combinations; ++i)
    {
        std::unordered_map<std::string, double> combination;
        size_t index = i;
        for (const auto &param : varying_params)
        {
            combination[param.first] = param.second[index % param.second.size()];
            index /= param.second.size();
        }
        combinations.push_back(combination);
    }

    return combinations;
}

void writeMatrixToFile(const std::vector<std::vector<double>> &matrix, const std::string &filePath)
{
    std::ofstream outFile(filePath);

    if (!outFile.is_open())
    {
        std::cerr << "Error opening file for writing: " << filePath << std::endl;
        return;
    }

    for (const auto &row : matrix)
    {
        for (const auto &element : row)
        {
            outFile << element << " ";
        }
        outFile << "\n";
    }

    outFile.close();
}

void writeBoolMatrixToFile(const std::vector<std::vector<bool>> &matrix, const std::string &filePath)
{
    std::ofstream outFile(filePath);

    if (!outFile.is_open())
    {
        std::cerr << "Error opening file for writing: " << filePath << std::endl;
        return;
    }

    for (const auto &row : matrix)
    {
        for (const auto &element : row)
        {
            outFile << element << " ";
        }
        outFile << "\n";
    }

    outFile.close();
}

std::vector<std::vector<double>> readMatrixFromFile(const std::string &filePath)
{
    std::vector<std::vector<double>> matrix;
    std::ifstream inFile(filePath);

    if (!inFile.is_open())
    {
        std::cerr << "Error opening file for reading: " << filePath << std::endl;
        return matrix;
    }

    std::string line;
    while (std::getline(inFile, line))
    {
        std::istringstream ss(line);
        std::vector<double> row;
        double value;

        while (ss >> value)
        {
            row.push_back(value);
        }

        matrix.push_back(row);
    }

    inFile.close();
    return matrix;
}

std::vector<std::vector<bool>> readBoolMatrixFromFile(const std::string &filePath)
{
    std::vector<std::vector<bool>> matrix;
    std::ifstream inFile(filePath);

    if (!inFile.is_open())
    {
        std::cerr << "Error opening file for reading: " << filePath << std::endl;
        return matrix;
    }

    std::string line;
    while (std::getline(inFile, line))
    {
        std::istringstream ss(line);
        std::vector<bool> row;
        double value;

        while (ss >> value)
        {
            row.push_back(value);
        }

        matrix.push_back(row);
    }

    inFile.close();
    return matrix;
}

// function to compare two states
bool comparestates(const std::vector<bool> &state1, const std::vector<bool> &state2)
{
    if (state1.size() != state2.size())
        return false;

    bool direct_match = true;
    bool inverse_match = true;

    for (size_t i = 0; i < state1.size(); ++i)
    {
        if (state1[i] != state2[i])
            direct_match = false;
        if (state1[i] != -state2[i])
            inverse_match = false;
    }

    return direct_match || inverse_match;
}