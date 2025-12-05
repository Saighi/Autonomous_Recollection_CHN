#ifndef UTILS_HPP
#define UTILS_HPP

#include <vector>
#include <string>
#include "network.hpp"
#include <fstream>
#include <filesystem>
#include <unordered_map>
#include <unordered_set>

// Custom hash function for vector<bool>
struct VectorBoolHash {
    std::size_t operator()(const std::vector<bool>& vec) const;
};

// Custom hash function for vector<double>
struct VectorDoubleHash {
    std::size_t operator()(const std::vector<double>& vec) const;
};

// Custom hash function for vector<double> that treats pattern and its inverse as equal
// Used for Hopfield networks where +/- inversions are equivalent
struct VectorDoubleHashSymmetric {
    std::size_t operator()(const std::vector<double>& vec) const;
};

// Custom equality comparator for vector<double> that treats pattern and inverse as equal
struct VectorDoubleEqualSymmetric {
    bool operator()(const std::vector<double>& a, const std::vector<double>& b) const;
};


std::vector<double> pattern_as_states(double up_rate, double down_rate, const std::vector<bool>& bin_pattern);
std::vector<std::vector<double>> patterns_as_states(double up_rate, double down_rate, const std::vector<std::vector<bool>>& bin_patterns);
std::vector<std::vector<bool>> generateCorrelatedPatterns(int nbPattern, int networkSize, double noiseLevel);
std::vector<bool> generateRandomPattern(int N);
std::vector<bool> randomizePattern(const std::vector<bool> &basePattern, int numFlips);
void show_vector(std::vector<double> vector);
double ratio_diff_vectors(const std::vector<double> &state1, const std::vector<double> &state2);
std::vector<std::unordered_map<std::string, double>> generateCombinations(const std::unordered_map<std::string, std::vector<double>> &varying_params);
std::vector<double> linspace(double start, double end, int num);
void lunchParalSim(std::string foldername_results, std::unordered_map<std::string, std::vector<double>> varying_params, void (*run_simulation)(int, const std::unordered_map<std::string, double>&, const std::string&));
void collectSimulationData(const std::string &folderResultsPath);
void collectSimulationDataSeries(const std::string &folderResultsPath);
void createParameterFile(const std::string &directory, const std::unordered_map<std::string, double> &parameters);
double computeCorrelation(const std::vector<bool>& original, const std::vector<bool>& noisy);
void lunchParalSimThreadLimit(int nb_thread_max, std::string foldername_results, std::unordered_map<std::string, std::vector<double>> varying_params, void (*run_simulation)(int, const std::unordered_map<std::string, double>&, const std::string&));
std::vector<double> generateEvenlySpacedIntegers(int a, int b, int n);
#endif