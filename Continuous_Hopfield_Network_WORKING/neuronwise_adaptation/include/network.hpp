#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <vector>

class Network{
    public:
        Network(std::vector<std::vector<bool>>, int, double);
        
        int size;
        double leak;
        double sum_all_inhib;

        std::vector<double> target_sum_each_inhib;
        std::vector<double> actual_sum_each_inhib;
        std::vector<double> activity_list;
        std::vector<double> rate_list;
        std::vector<double> derivative_activity_list;
        std::vector<double> depression;
        std::vector<std::vector<bool>> connectivity_matrix;
        std::vector<std::vector<double>>  weight_matrix;

        void iterate(double);
        void noisy_iterate(double, double, double);
        void noisy_depression_iterate(double, double, double);
        double transfer(double);
        double transfer_inverse(double );
        void blank_init();
        void set_state(std::vector<double>);
        void reinforce_attractor(std::vector<double>, double);
        void pot_depress_bin(double pot_rate, std::vector<bool> winners);
        void reset_depression();
};      

#endif
