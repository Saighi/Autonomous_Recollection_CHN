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

std::vector<std::vector<double>> patterns_as_states(double up_rate, double down_rate, std::vector<std::vector<bool>> bin_patterns)
{
    std::vector<double> state_input(bin_patterns[0].size());
    std::vector<std::vector<double>> initial_patterns_state_list(bin_patterns.size());
    for (int i = 0; i < bin_patterns.size(); i++)
    {
        for (int j = 0; j < bin_patterns[i].size(); j++)
        {
            if (bin_patterns[i][j])
            {
                state_input[j] = up_rate;
            }
            else
            {
                state_input[j] = down_rate;
            }
        }
        initial_patterns_state_list[i] = state_input;
    }
    return initial_patterns_state_list;
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

std::vector<std::vector<bool>> generateCombinedPatterns(std::vector<std::vector<bool>> patterns)
{
    std::vector<bool> combinedPattern;
    std::vector<std::vector<bool>> generated_patterns;
    for (int i = 0; i<(patterns.size())-1; i++)
    {
        combinedPattern = patterns[i];
        for(int j = 0; j<patterns[i+1].size(); j++){
            combinedPattern.emplace_back(patterns[i+1][j]);
        }
        generated_patterns.emplace_back(combinedPattern);
    }
    return generated_patterns;
}

std::vector<std::vector<double>> generateCombinedStates(std::vector<std::vector<double>> patterns)
{
    std::vector<double> combinedPattern;
    std::vector<std::vector<double>> generated_patterns;
    for (int i = 0; i<(patterns.size())-1; i++)
    {
        combinedPattern = patterns[i];
        for(int j = 0; j<patterns[i+1].size(); j++){
            combinedPattern.emplace_back(patterns[i+1][j]);
        }
        generated_patterns.emplace_back(combinedPattern);
    }
    return generated_patterns;
}

std::vector<double> pickFirstNElements(const std::vector<double> &elements, int N)
{
    std::vector<double> pickedElements;

    // Ensure N does not exceed the size of the elements vector
    N = std::min(N, static_cast<int>(elements.size()));

    // Add the first N elements to the pickedElements vector
    for (int i = 0; i < N; ++i)
    {
        pickedElements.push_back(elements[i]);
    }

    return pickedElements;
}

std::vector<double> pickLastNElements(const std::vector<double> &elements, int N)
{
    std::vector<double> pickedElements(N);
    if (N > elements.size())
    {
        N = elements.size();
    }
    std::copy(elements.end() - N, elements.end(), pickedElements.begin());
    return pickedElements;
}

// Function to randomize some elements of the initial state
std::vector<double> randomizeInitialState(const std::vector<double> &pattern, int num_random_elements)
{
    std::vector<double> randomized_state = pattern;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, pattern.size() - 1);

    for (int i = 0; i < num_random_elements; ++i)
    {
        int index = dis(gen);
        randomized_state[index] = (randomized_state[index] == 1) ? -1 : 1;
    }

    return randomized_state;
}

// function to compare two states
bool comparestates(const std::vector<double> &state1, const std::vector<double> &state2)
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

std::vector<double> randomBinaryVector(int size)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 1);
    std::vector<double> binaryVector(size);
    for (int i = 0; i < size; ++i)
    {
        binaryVector[i] = (dis(gen) == 1) ? -1 : 1;
    }

    return binaryVector;
}