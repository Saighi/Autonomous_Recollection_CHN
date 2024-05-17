#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <vector>

class Network{
    public:
        Network(std::vector<std::vector<bool>>, int, double);
        int size;
        double leak;
        std::vector<double> activity_list;
        std::vector<double> rate_list;
        std::vector<double> derivative_activity_list;
        std::vector<std::vector<bool>> connectivity_matrix;
        std::vector<int> overlay_vec;
        std::vector<std::vector<int>> synapses_overlay_matrix;
        std::vector<std::vector<double>>  weight_matrix;
        void iterate(double);
        void noisy_iterate(double, double, double);
        void noisy_depressed_iterate(double, double, double, std::vector<std::vector<bool>>);
        double transfer(double);
        double transfer_inverse(double);
        void blank_init();
        void set_state(std::vector<double>);
        void reinforce_attractor(std::vector<double>, double);
        void add_overlay(std::vector<bool>);
        void add_synapse_overlay(std::vector<bool>);

};      

#endif
