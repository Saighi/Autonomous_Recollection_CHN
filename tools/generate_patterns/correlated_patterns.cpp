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
#include <stdio.h>

using namespace std;

void writeToCSV(std::ofstream &file, const std::vector<bool> &data)
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

int main(int argc, char **argv)
{
    int num_patterns = 5;
    int network_size = 25;
    // int nb_winners = static_cast<int>(network_size/2);
    int nb_winners = 10;
    double noise_level = 0.5;


    std::string foldername = "../../input_data/correlated_patterns";
    std::string filename = foldername + "/patterns.data";
    std::ofstream file(filename, std::ios::trunc);

    vector<vector<bool>> initial_patterns = generatePatterns(num_patterns, network_size, nb_winners, noise_level);
    for(int i= 0; i<num_patterns; i++){
        std::cout << initial_patterns[i][0] << std::endl;
        writeToCSV(file, initial_patterns[i]);
    }
    file.close();
}
