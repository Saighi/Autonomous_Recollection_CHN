#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <vector>

class Network{
    public:

        Network(std::vector<std::vector<bool>>, int);
        void blank_init();

        int size;
        std::vector<std::vector<bool>> connectivity_matrix;
        std::vector<std::vector<double>>  weight_matrix;

        std::vector<double> activate(const std::vector<double> &state);
        std::vector<double> runSynchronous(const std::vector<double> &initial_state, int steps = 10);
        void trainPerceptron(const std::vector<std::vector<double>> &patterns, int nb_iter, double step_size);
        void trainHebbian(const std::vector<std::vector<double>> &patterns);
        std::vector<double> runAsynchronous(const std::vector<double> &initial_state, int nb_steps);
        std::vector<double> runStochastic(const std::vector<double> &initial_state, double temperature, double epsilon = 0.01);
        std::vector<double> runStochasticNbIter(const std::vector<double> &initial_state, double temperature, int nb_iter);
        std::vector<double> runStochasticNbIterSimulatedAnnealing(const std::vector<double> &initial_state, double temperature_start, int nb_iter);
        double ComputeMeanCrossTalk(std::vector<std::vector<bool>> patterns);
        // void trainPerceptronInput(const std::vector<std::vector<double>> &patterns, int index_input);
        // std::vector<double> runAndClamp(const std::vector<double> &initial_states, int index_clamping, int steps);

        // std::vector<std::vector<double>> inhib_matrix;
        // std::vector<std::vector<int>> scale_inhib;

        // void iterate(double);
        // void noisy_iterate(double, double, double);
        // void noisy_depression_iterate(double, double, double);
        // double transfer(double);
        // double transfer_inverse(double );
        // void set_state(std::vector<double>);
        // void reinforce_attractor(std::vector<double>, double);
        // void pot_inhib(double);
        // void pot_inhib_bin(double pot_rate, std::vector<bool> winners);
        // // void pot_inhib_normalize(double, int);
        // void iterative_normalize(int, double);
        // void reset_inhib();
        // void pot_inhib_bin_scale(double, std::vector<bool>);
};      

#endif
