#ifndef UTILS_HPP
#define UTILS_HPP

#include <vector>
#include <string>
#include "network.hpp"
#include <fstream>
#include <filesystem>
#include <unordered_map>
#include <iostream>
#include <cstdint>

// Configuration struct for simulation runs (from sleep version)
struct SimulationConfig {
    double delta;
    double epsilon;
    bool depressed = false;
    bool noise = false;
    bool save = false;
    double mean = 0.0;
    double stddev = 0.005;
    std::ostream *output = &std::cout;
    int max_iter = 10000;
    bool use_full_inhibition = false;  // false = diagonal only, true = full matrix
};

// Visualization functions
void show_state(Network& net);
void show_state_grid(Network& net, int rows);
void show_matrix(std::vector<std::vector<double>> matrix);
void show_vector(std::vector<double> vector);
void show_vector_bool_grid(std::vector<bool> vec, int rows);

// Simulation runners
void run_net_sim_noisy(Network& net, int nb_iter, double delta, double mean, double stddev);
void run_net_sim_noisy_depressed(Network& net, int nb_iter, double delta, double mean, double stddev);
void run_net_sim_noisy_depressed_save(Network& net, int nb_iter, double delta, double mean, double stddev, std::ofstream& file);
int run_net_sim_choice(Network& net, SimulationConfig& conf);

// Winner-take-all functions
std::vector<double> assignStateToTopNValues(std::vector<double>& vec, int n, double winner_state, double loser_state);
std::vector<bool> assignBoolToTopNValues(std::vector<double>& vec, int n);

// CSV/File I/O functions (overloads for compatibility)
void writeToCSV(std::ofstream& file, const std::vector<double>& data);
void writeToCSV(std::ostream* file, const std::vector<double>& data);
void writeBoolToCSV(std::ofstream& file, const std::vector<bool>& data);
void writeBoolToCSV(std::ostream& file, const std::vector<bool>& data);

// Matrix I/O (text format)
void writeMatrixToFile(const std::vector<std::vector<double>>& matrix, const std::string& filePath);
void writeBoolMatrixToFile(const std::vector<std::vector<bool>>& matrix, const std::string& filePath);
std::vector<std::vector<double>> readMatrixFromFile(const std::string& filePath);
std::vector<std::vector<bool>> readBoolMatrixFromFile(const std::string& filePath);

// Binary Matrix I/O (compact format for archiving)
void writeBinaryMatrix(const std::vector<std::vector<double>>& matrix, std::ostream& out);
void writeBinaryMatrix(const std::vector<std::vector<double>>& matrix, const std::string& filePath);
std::vector<std::vector<double>> readBinaryMatrix(std::istream& in);
std::vector<std::vector<double>> readBinaryMatrix(const std::string& filePath);

void writeBitpackedBoolMatrix(const std::vector<std::vector<bool>>& matrix, std::ostream& out);
void writeBitpackedBoolMatrix(const std::vector<std::vector<bool>>& matrix, const std::string& filePath);
std::vector<std::vector<bool>> readBitpackedBoolMatrix(std::istream& in);
std::vector<std::vector<bool>> readBitpackedBoolMatrix(const std::string& filePath);

// Binary blob conversion (for SQLite storage)
std::vector<uint8_t> matrixToBlob(const std::vector<std::vector<double>>& matrix);
std::vector<uint8_t> boolMatrixToBlob(const std::vector<std::vector<bool>>& matrix);
std::vector<std::vector<double>> blobToMatrix(const std::vector<uint8_t>& blob);
std::vector<std::vector<bool>> blobToBoolMatrix(const std::vector<uint8_t>& blob);

// Math utilities
std::vector<double> linspace(double start, double end, int num);
std::vector<double> generateEvenlySpacedIntegers(int a, int b, int n);

// Pattern generation and loading
// use_old_patterns: if true, use legacy balanced-flip generator (expects sparsity ~= 0.5)
//                   if false, use parent+redraw generator with sparsity = P(x_i = 0)
std::vector<std::vector<bool>> generatePatterns(int K, int N, double sparsity, double rho, bool use_old_patterns);
std::vector<std::vector<bool>> loadPatterns(const std::string& filename);
bool patternExists(const std::vector<std::vector<bool>>& patterns, const std::vector<bool>& pattern);
bool areVectorsEqual(const std::vector<bool>& v1, const std::vector<bool>& v2);
bool comparestates(const std::vector<bool>& state1, const std::vector<bool>& state2);
bool matchesPatternOrConverse(const std::vector<bool>& pattern1, const std::vector<bool>& pattern2);

// Heterogeneous pattern generation (patterns with varying sparsities)
// Per-pattern metadata
struct PatternInfo {
    int index;
    double sparsity;   // P(0) = fraction inactive
    int nb_active;
};

// Full metadata structure for heterogeneous patterns
struct PatternMetadata {
    int version = 1;
    int num_patterns = 0;
    int network_size = 0;
    std::string generation_method;
    double mean_sparsity = 0.5;
    double sparsity_width = 0.2;
    double rho = 0.5;
    std::vector<PatternInfo> patterns;
};

// Generate patterns with heterogeneous sparsities (each pattern has different sparsity)
// Uses C++-style parent/redraw algorithm:
// 1. Generate parent pattern with P(0) = mean_sparsity
// 2. For each pattern: sample sparsity_i ~ Uniform(mean - width/2, mean + width/2)
// 3. Redraw k = (1-rho)*N positions using P(0) = sparsity_i
std::pair<std::vector<std::vector<bool>>, PatternMetadata> generatePatternsHeterogeneous(
    int K, int N, double mean_sparsity, double sparsity_width, double rho);

// Write pattern metadata to JSON file
void writePatternMetadata(const PatternMetadata& metadata, const std::string& filepath);

// Pattern to state conversion
std::vector<double> pattern_as_states(double up_rate, double down_rate, std::vector<bool> bin_pattern);
std::vector<std::vector<double>> patterns_as_states(double up_rate, double down_rate, std::vector<std::vector<bool>> bin_patterns);
std::vector<double> pattern_as_states_with_distance_noise(double drive_target, std::vector<bool> bin_pattern, double distance_noise_level, Network& net);
std::vector<std::vector<double>> patterns_as_states_with_distance_noise(double drive_target, std::vector<std::vector<bool>> bin_patterns, double distance_noise_level, Network& net);

// Parameter handling
void createParameterFile(const std::string& directory, const std::unordered_map<std::string, double>& parameters);
std::unordered_map<std::string, double> readParametersFile(const std::string& filePath);
std::unordered_map<std::string, double> fuseMaps(std::unordered_map<std::string, double> map1, std::unordered_map<std::string, double> map2);
std::vector<std::unordered_map<std::string, double>> generateCombinations(const std::unordered_map<std::string, std::vector<double>>& varying_params);

// Utility functions
std::vector<double> setToValueRandomElements(const std::vector<double>& baseValues, int numFlips, double value);
void lunchParalSim(std::string foldername_results, std::unordered_map<std::string, std::vector<double>> varying_params, void (*run_simulation)(int, std::unordered_map<std::string, double>, const std::string));

// Data collection/aggregation
void collectSimulationData(const std::string& folderResultsPath);
void collectSimulationDataSeries(const std::string& folderResultsPath);

#endif
