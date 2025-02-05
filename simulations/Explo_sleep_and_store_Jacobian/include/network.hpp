#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <immintrin.h>
#include <cmath>
#include <random>
#include <vector>

class Network{
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
        std::vector<std::vector<double>>  weight_matrix;
        std::vector<std::vector<double>> inhib_matrix;
        std::vector<std::vector<int>> scale_inhib;

        // Added or updated methods that use AVX:
        void iterate(double delta);
        void noisy_iterate(double delta, double mean, double stddev);
        void depressed_iterate(double delta);
        void noisy_depressed_iterate(double delta, double mean, double stddev);

        void blank_init();
        void set_state(std::vector<double>);
        void reinforce_attractor(std::vector<double>, double);
        void pot_inhib(double);
        void pot_inhib_symmetric(double);
        void pot_inhib_bin(double pot_rate, std::vector<bool> winners);
        // void pot_inhib_normalize(double, int);
        void iterative_normalize(int, double);
        void reset_inhib();
        void pot_inhib_bin_scale(double, std::vector<bool>);

    private:
        // ... your existing members ...

        // Example member to avoid re-initializing generator each call:
        std::default_random_engine generator;

        // Example: vectorized “dot-product-like” operations
        // might be placed here or inline in the .cpp.

        double transfer(double activation) const;
        double transfer_inverse(double activation) const;
        double transferDerivative(double activation) const;

        // Computes the Jacobian matrix at the current state.
        // J_{ij} = weight_matrix[i][j] * f'(activity_list[j]) - leak *
        // delta_{ij}
        std::vector<std::vector<double>> computeJacobian() const;

        // Computes the overlap (correlation) of a given Jacobian with a
        // pattern. Here the pattern is a vector of doubles (e.g., a stored
        // state). The measure is defined as: (p^T * J * p) / (p^T * p)
        double correlateJacobianWithPattern(
            const std::vector<std::vector<double>> &J,
            const std::vector<bool> &pattern) const;

        // Given a set of patterns, return the correlation of the Jacobian with
        // each pattern.
        std::vector<double> correlateJacobianWithPatterns(
            const std::vector<std::vector<double>> &patterns) const;
};

#endif
