#ifndef UTILS_HPP
#define UTILS_HPP

#include <vector>
#include <string>
#include "network.hpp"
#include <fstream>
#include <filesystem>
#include <unordered_map>

void show_state(Network&);
void show_state_grid(Network&, int);
void run_net_sim(Network&, int, double);
void show_matrix(std::vector<std::vector<double>>);
void show_vector(std::vector<double>);
void run_net_sim_noisy(Network &, int, double, double, double);
void run_net_sim_noisy_depressed(Network &, int, double, double, double);
void run_net_sim_noisy_depressed_save(Network &net, int nb_iter, double delta, double mean, double stddev, std::ofstream &file);
void run_net_sim_noisy_save(Network &net, int nb_iter, double delta, double mean, double stddev, std::ofstream &file);
std::vector<double> assignStateToTopNValues(std::vector<double> &, int, double, double);
std::vector<bool> assignBoolToTopNValues(std::vector<double> &, int);
void show_vector_bool_grid(std::vector<bool>, int);
struct Compare;
void writeToCSV(std::ofstream &file, const std::vector<double> &data);
void writeBoolToCSV(std::ofstream &file, const std::vector<bool> &data);
std::vector<double> linspace(double start, double end, int num);
std::vector<std::vector<bool>> generatePatterns(int K, int N, int nb_winning_units, double noiseLevel);
std::vector<std::vector<bool>> loadPatterns(const std::string &filename);
void createParameterFile(const std::string &directory, const std::unordered_map<std::string, double> &parameters);
std::vector<std::vector<double>> patterns_as_states(double up_rate, double down_rate, std::vector<std::vector<bool>> bin_patterns);
std::vector<std::unordered_map<std::string, double>> generateCombinations(const std::unordered_map<std::string, std::vector<double>> &varying_params);
std::unordered_map<std::string, double> fuseMaps(std::unordered_map<std::string, double> map1, std::unordered_map<std::string, double> map2);
std::unordered_map<std::string, double> readParametersFile(const std::string &filePath);
std::vector<std::vector<double>> readMatrixFromFile(const std::string &filePath);
std::vector<std::vector<bool>> readBoolMatrixFromFile(const std::string &filePath);
#endif