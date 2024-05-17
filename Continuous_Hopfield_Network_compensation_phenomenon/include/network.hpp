#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <vector>

class Network{
    public:
        Network(std::vector<std::vector<bool>>, int, double);
        int size;
        double leak;
        std::vector<std::vector<bool>> connectivity_matrix;
        std::vector<std::vector<double>>  weight_matrix;
        std::vector<double> activity_list;
        std::vector<double> rate_list;
        std::vector<double> derivative_activity_list;
        void iterate(double delta);
        double transfer(double);
        double transfer_inverse(double );
        void blank_init();
        void set_state(std::vector<double>);
        void reinforce_attractor(std::vector<double>, double);
        void noisy_iterate(double delta, double, double);
};      

#endif
