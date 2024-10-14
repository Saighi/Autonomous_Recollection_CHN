#ifndef UTILS_HPP
#define UTILS_HPP

#include <vector>
#include <string>
#include "network.hpp"
#include <fstream>
#include <filesystem>

void show_state(Network&);
void show_state_grid(Network&, int);
void run_net_sim(Network&, int, double);
void show_matrix(std::vector<std::vector<double>>);
void show_vector(std::vector<double>);
void run_net_sim_noisy(Network &, int, double, double, double);
void run_net_sim_noisy_depressed(Network &, int, double, double, double);
void run_net_sim_noisy_depressed_save(Network &net, int nb_iter, double delta, double mean, double stddev, std::ofstream &file);
std::vector<double> assignStateToTopNValues(std::vector<double> &, int, double, double);
std::vector<bool> assignBoolToTopNValues(std::vector<double> &, int);
void show_vector_bool_grid(std::vector<bool>, int);
struct Compare;
void writeToCSV(std::ofstream &file, const std::vector<double> &data);
std::vector<double> linspace(double start, double end, int num);

#endif