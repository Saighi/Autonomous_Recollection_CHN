#ifndef UTILS_HPP
#define UTILS_HPP

#include <vector>
#include "network.hpp"

void show_state(Network&);
void show_state_grid(Network&, int rows);
void run_net_sim(Network&, int, double);
void show_weight_matrix(Network&);
void run_net_sim_noisy(Network &, int, double, double, double);

#endif