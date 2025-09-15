#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <immintrin.h>

#include <cmath>
#include <random>
#include <vector>

class Network {
public:
    Network(std::vector<std::vector<bool>>, int, double);
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

    // Added or updated methods that use AVX:
    void iterate(double delta);
    void noisy_iterate(double delta, double mean, double stddev);
    void depressed_iterate(double delta);
    void noisy_depressed_iterate(double delta, double mean, double stddev);

    void blank_init();
    void set_state(std::vector<double>);
    void reinforce_attractor(std::vector<double>, double);
    void reinforce_attractor_bin(
        std::vector<bool> binary_state,
        double learning_rate);
    void pot_inhib(double);
    void pot_inhib_symmetric(double);
    void pot_inhib_bin(double pot_rate, std::vector<bool> winners);
    // void pot_inhib_normalize(double, int);
    void iterative_normalize(int, double);
    void reset_inhib();
    void pot_inhib_bin_scale(double, std::vector<bool>);

    double transfer(double activation);
    double transfer_inverse(double activation);
    void derivative_gradient_descent(std::vector<double>& target_drives,
                                     double learning_rate,
                                     double leak,
                                     std::vector<double>& drive_errors);
    void derivative_gradient_descent_with_bias(std::vector<double>& target_drives,
                                               double learning_rate,
                                               double leak,
                                               std::vector<double>& drive_errors);
    
    void derivative_gradient_descent_arbitrary(std::vector<double>& target_rates,
                                          double learning_rate, double leak,
                                          std::vector<double>& drive_errors);
    void derivative_gradient_descent_arbitrary_with_bias(
        std::vector<double>& target_rates, double learning_rate, double leak,
        std::vector<double>& drive_errors);

    void derivative_gradient_descent_with_momentum(
        std::vector<bool>& target_bin_state, std::vector<double>& target_rates,
        double target_drive, double learning_rate, double leak,
        std::vector<double>& drive_errors,
        std::vector<std::vector<double>>& velocity_matrix,
        double momentum_coef);

    // null-sum training path removed

    double compute_energy();
    std::vector<double> give_derivative_u(double delta);
    std::vector<double> give_derivative_v(std::vector<double> derivative_u);

    // Decomposed derivative components (synaptic pushes only)
    // - Excitatory-only: d_i = (sum_j w_ij * v_j) * delta
    // - Inhibitory-only: d_i = -(sum_j inh_ij * v_j) * delta
    // Note: leak term is intentionally excluded to isolate synaptic contributions.
    std::vector<double> give_derivative_u_excit_only(double delta);
    std::vector<double> give_derivative_u_inhib_only(double delta);
    // Weights-only: exclude inhibition, leak, and bias entirely
    std::vector<double> give_derivative_u_W_only(double delta);

private:
    // ... your existing members ...

    // Example member to avoid re-initializing generator each call:
    std::default_random_engine generator;

    // Example: vectorized “dot-product-like” operations
    // might be placed here or inline in the .cpp.

};

#endif
