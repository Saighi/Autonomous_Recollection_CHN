#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <immintrin.h>
#include <cmath>
#include <random>
#include <vector>

class Network {
public:
    Network(std::vector<std::vector<bool>>, int, double);

    // Network attributes
    int size;
    double leak;
    double inhib_strenght;
    double sum_all_inhib;
    std::vector<double> target_sum_each_inhib;
    std::vector<double> actual_sum_each_inhib;
    std::vector<double> activity_list;
    std::vector<double> rate_list;
    std::vector<double> derivative_activity_list;

    std::vector<std::vector<bool>> connectivity_matrix;
    std::vector<std::vector<double>> weight_matrix;
    std::vector<std::vector<double>> inhib_matrix;
    std::vector<std::vector<int>> scale_inhib;
    std::vector<double> bias;  // Per-neuron bias (plastic)

    // Core iteration methods
    void iterate(double delta);
    void iterate_query_drive(double delta, double strength_drive, std::vector<double>& query_drives);
    void noisy_iterate(double delta, double mean, double stddev);
    void depressed_iterate(double delta);
    void noisy_depressed_iterate(double delta, double mean, double stddev);

    // Transfer functions (public for external use)
    double transfer(double activation);
    double transfer_inverse(double activation);

    // State management
    void blank_init();
    void set_state(std::vector<double> new_state);
    void reinforce_attractor(std::vector<double> target_state, double learning_rate);

    // Inhibitory plasticity
    void pot_inhib(double pot_rate);
    void pot_inhib_symmetric(double pot_rate);
    void pot_inhib_bin(double pot_rate, std::vector<bool> winners);
    void pot_inhib_bin_scale(double pot_rate, std::vector<bool> winners);
    void iterative_normalize(int nb_iter_normalize, double rate_normalize);
    void reset_inhib();

    // Learning methods (gradient descent variants)
    void rate_derivative_gradient_descent(std::vector<double> target_state, double learning_rate, double leak);
    void derivative_gradient_descent(std::vector<bool>& target_bin_state, std::vector<double>& target_rates,
                                     double target_drive, double learning_rate, double leak,
                                     std::vector<double>& drive_errors);
    void derivative_gradient_descent_with_momentum(std::vector<bool>& target_bin_state,
                                                   std::vector<double>& target_rates,
                                                   double target_drive,
                                                   double learning_rate,
                                                   double leak,
                                                   std::vector<double>& drive_errors,
                                                   std::vector<std::vector<double>>& velocity_matrix,
                                                   double momentum_coef);
    void derivative_gradient_descent_with_bias(std::vector<double>& target_drives,
                                               double learning_rate,
                                               double leak,
                                               std::vector<double>& drive_errors);
    void derivative_gradient_descent_with_bias_and_momentum(std::vector<double>& target_drives,
                                                            double learning_rate,
                                                            double leak,
                                                            std::vector<double>& drive_errors,
                                                            std::vector<std::vector<double>>& velocity_matrix,
                                                            std::vector<double>& velocity_bias,
                                                            double momentum_coef);
    void derivative_gradient_descent_with_bias_and_momentum_avx(std::vector<double>& target_drives,
                                                                 double learning_rate,
                                                                 double leak,
                                                                 std::vector<double>& drive_errors,
                                                                 std::vector<std::vector<double>>& velocity_matrix,
                                                                 std::vector<double>& velocity_bias,
                                                                 double momentum_coef);

private:
    // Member RNG for noisy iterations (avoid re-initializing each call)
    std::default_random_engine generator;
};

#endif
