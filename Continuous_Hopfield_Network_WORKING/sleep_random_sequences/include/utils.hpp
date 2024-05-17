#ifndef UTILS_HPP
#define UTILS_HPP

#include <vector>
#include <string>
#include "network.hpp"

void show_state(Network&);
void show_state_grid(Network&, int);
void run_net_sim(Network&, int, double);
void show_matrix(std::vector<std::vector<double>>);
void show_vector(std::vector<double>);
void run_net_sim_noisy(Network &, int, double, double, double);
void run_net_sim_noisy_depressed(Network &, int, double, double, double);
std::vector<double> assignStateToTopNValues(std::vector<double> &, int , double , double );
std::vector<bool> assignBoolToTopNValues(std::vector<double> &, int);
void show_vector_bool_grid(std::vector<bool>, int);
struct Compare;
void appendToCSV(const std::vector<double>& data, const std::string& filename);
std::vector<bool> generateRandomBinarySequenceWithOnes(int n, int x);
bool hasOverlap(const std::vector<bool>& seq1, const std::vector<bool>& seq2, int O);
bool areSequencesEqual(const std::vector<bool>& seq1, const std::vector<bool>& seq2);
int countCommonSequences(const std::vector<std::vector<bool>>& vec1, const std::vector<std::vector<bool>>& vec2);
#endif