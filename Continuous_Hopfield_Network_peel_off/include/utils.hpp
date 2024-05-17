#ifndef UTILS_HPP
#define UTILS_HPP

#include <vector>
#include <queue>
#include "network.hpp"

struct Compare;
void displayPriorityQueue(std::priority_queue<std::pair<double, int>, std::vector<std::pair<double, int>>, Compare>);
void show_state(Network&);
void show_state_grid(Network&, int, int);
void show_state_grid_rate(Network &, int );
void show_vector_bool_grid(std::vector<bool>, int );
void show_vector_int_grid(std::vector<int>, int);
void run_net_sim(Network &, int, double);
void show_matrix_double(std::vector<std::vector<double>>);
void show_matrix_int(std::vector<std::vector<int>>);
void show_matrix_bool(std::vector<std::vector<bool>>);
void show_vector(std::vector<double>);
void run_net_sim_noisy(Network &, int, double, double, double);
// void run_net_sim_noisy_depressed(Network &, int, double, double, double);
std::vector<double> assignStateToTopNValues(std::vector<double> &, int, double, double);
std::vector<bool> assignBoolToTopNValues(std::vector<double> &, int);
std::vector<bool> take_peal(std::vector<int>&,std::vector<bool>&);
std::vector<int> peal_off(std::vector<int>&, std::vector<bool>&);

std::vector<std::vector<int>> peal_off_syn(std::vector<std::vector<int>>&, std::vector<bool>&);
std::vector<std::vector<bool>> toBinaryMatrix(std::vector<std::vector<int>>&);

#endif