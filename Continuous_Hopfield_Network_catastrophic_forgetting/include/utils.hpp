#ifndef UTILS_HPP
#define UTILS_HPP

#include <vector>
#include <queue>
#include "network.hpp"

struct Compare;
void displayPriorityQueue(std::priority_queue<std::pair<double, int>, std::vector<std::pair<double, int>>, Compare>);
void show_state(Network&);
void show_state_grid(Network&, int);
void run_net_sim(Network&, int, double);
void show_matrix(std::vector<std::vector<double>>);
void show_vector(std::vector<double>);
void run_net_sim_noisy(Network &, int, double, double, double);
void run_net_sim_noisy_depressed(Network &, int, double, double, double);
std::vector<double> assignStateToTopNValues(std::vector<double> &, int, double, double);

#endif