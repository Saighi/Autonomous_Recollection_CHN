#ifndef UTILS_HPP
#define UTILS_HPP

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "network.hpp"

struct SimulationConfig {
    // Required parameters
    double delta;
    double epsilon;
    // Optional parameters
    bool depressed = false;
    bool noise = false;
    bool save = false;
    double mean = 0.0;
    double stddev = 0.005;
    std::ostream *output = &std::cout;
    int max_iter = 10000;
};

void show_state(Network &);
void show_state_grid(Network &, int);
void run_net_sim(Network &, int, double);
void show_matrix(std::vector<std::vector<double>>);
void show_vector(std::vector<double>);

int run_net_sim_choice(Network &net, SimulationConfig &conf);
std::vector<double> assignStateToTopNValues(std::vector<double> &, int, double,
                                            double);
std::vector<bool> assignBoolToTopNValues(std::vector<double> &, int);
void show_vector_bool_grid(std::vector<bool>, int);
struct Compare;
void writeToCSV(std::ostream *file, const std::vector<double> &data);
void writeBoolToCSV(std::ostream &file, const std::vector<bool> &data);
std::vector<double> linspace(double start, double end, int num);
std::vector<std::vector<bool>> generatePatterns(int K, int N,
                                                int nb_winning_units,
                                                double noiseLevel);
std::vector<std::vector<bool>> loadPatterns(const std::string &filename);
void createParameterFile(
    const std::string &directory,
    const std::unordered_map<std::string, double> &parameters);
std::vector<std::vector<double>> patterns_as_states(
    double up_rate, double down_rate,
    std::vector<std::vector<bool>> bin_patterns);
std::vector<std::unordered_map<std::string, double>> generateCombinations(
    const std::unordered_map<std::string, std::vector<double>> &varying_params);
std::unordered_map<std::string, double> fuseMaps(
    std::unordered_map<std::string, double> map1,
    std::unordered_map<std::string, double> map2);
std::unordered_map<std::string, double> readParametersFile(
    const std::string &filePath);
std::vector<std::vector<double>> readMatrixFromFile(
    const std::string &filePath);
std::vector<std::vector<bool>> readBoolMatrixFromFile(
    const std::string &filePath);
void collectSimulationDataSeries(const std::string &folderResultsPath);
double computeCorrelation(const std::vector<double> &vec1,
                          const std::vector<bool> &vec2);
void check_stable_states(Network net, std::vector<std::vector<bool>> patterns,
                         double init_drive, double drive_target,
                         SimulationConfig config, std::ostream *file);
std::vector<double> pattern_as_states(double up_rate, double down_rate,
                                      std::vector<bool> bin_pattern);
#endif
