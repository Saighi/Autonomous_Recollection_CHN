#include "network.hpp"
#include "utils.hpp"
#include <vector>
#include <cmath>
#include <iostream>
#include <random>

//
// AVX helper: Dot-product for double-precision arrays
// Summation of (weight[i][j] * rate_list[j]) ignoring connectivity for now.
// If connectivity matters, we can fold it in below.
//
static inline double avx_dot_product(const double* wRow, const double* rate,
                                     int length) {
    __m256d vsum = _mm256_setzero_pd();

    int idx = 0;
    for (; idx + 4 <= length; idx += 4) {
        __m256d vw = _mm256_loadu_pd(&wRow[idx]);
        __m256d vr = _mm256_loadu_pd(&rate[idx]);
        vsum = _mm256_fmadd_pd(vw, vr, vsum);  // vsum += vw * vr
    }

    alignas(32) double partial[4];
    _mm256_storeu_pd(partial, vsum);
    double sum = partial[0] + partial[1] + partial[2] + partial[3];

    // remainder
    for (; idx < length; idx++) {
        sum += wRow[idx] * rate[idx];
    }

    return sum;
}

Network::Network(std::vector<std::vector<bool>> connect_mat, int size_network,
                 double lk) {
    leak = lk;
    connectivity_matrix = connect_mat;  // a vector of vector<bool>
    size = size_network;
    inhib_strenght = 10;

    // Initialize the random engine once (avoid re-init each call)
    std::random_device rd;
    generator.seed(rd());

    blank_init();
}

// Basic iteration (no depression, no noise)
void Network::iterate(double delta) {
    // We zero out derivative_activity_list before accumulation
    std::fill(derivative_activity_list.begin(), derivative_activity_list.end(),
              0.0);

    // Compute derivative for each neuron i
    for (int i = 0; i < size; i++) {
        // Equivalent to:
        //   derivative_activity_list[i] += sum_j [ weight_matrix[i][j] *
        //   rate_list[j] ]
        // but using AVX for the dot-product:
        double sum =
            avx_dot_product(weight_matrix[i].data(), rate_list.data(), size);
        
        double d = sum - (leak * activity_list[i]);
        activity_list[i] += delta * d;
        rate_list[i] = transfer(activity_list[i]);
    }

    // We do not necessarily need to zero out derivative_activity_list here
    // again, but you can if you want to keep the same logic each step.
}

// chatgpt add some code, the corresponding includes, the corresponding
// parameters to add some gaussian noise to the derivative such that when we
// iterate over the network through this function it does some kind of annealing
// add the function definition to network.hpp
// Noisy iteration: add Gaussian noise to derivative
void Network::noisy_iterate(double delta, double mean, double stddev) {
    std::normal_distribution<double> distribution(mean, stddev);

    // Accumulate derivative
    for (int i = 0; i < size; i++) {
        double sum =
            avx_dot_product(weight_matrix[i].data(), rate_list.data(), size);
        // Add random Gaussian noise
        double noise = distribution(generator);
        double d = sum - (leak * activity_list[i]) + noise;
        activity_list[i] += delta * d;
        rate_list[i] = transfer(activity_list[i]);
    }

}

void Network::depressed_iterate(double delta) {

    // 3) For each neuron i, do a dot product for weights and for inhib
    for (int i = 0; i < size; i++) {
        // sum of excitatory or "normal" synapses
        double sum_weight =
            avx_dot_product(weight_matrix[i].data(), rate_list.data(), size);

        // sum of inhibitory synapses
        double sum_inhib =
            avx_dot_product(inhib_matrix[i].data(), rate_list.data(), size);

        double d = sum_weight - sum_inhib - (leak * activity_list[i]);
        activity_list[i] += delta * d;
        rate_list[i] = transfer(activity_list[i]);
    }
}

void Network::noisy_depressed_iterate(double delta, double mean,
                                      double stddev) {
    std::normal_distribution<double> distribution(mean, stddev);

    // 3) For each neuron i, do a dot product for weights and for inhib
    for (int i = 0; i < size; i++) {
        double noise = distribution(generator);
        // sum of excitatory or "normal" synapses
        double sum_weight =
            avx_dot_product(weight_matrix[i].data(), rate_list.data(), size);

        // sum of inhibitory synapses
        double sum_inhib =
            avx_dot_product(inhib_matrix[i].data(), rate_list.data(), size);

        double d = sum_weight - sum_inhib - (leak * activity_list[i])+noise;
        activity_list[i] += delta * d;
        rate_list[i] = transfer(activity_list[i]);
    }
}

double Network::transfer(double activation) const {
    return 1.0 / (1.0 + std::exp(-activation));
}

double Network::transfer_inverse(double activation) const {
    return -std::log(-1.0 + 1.0 / activation);
}

// Helper: Compute the derivative of the transfer function.
// Since transfer(x)=1/(1+exp(-x)), its derivative is f(x)*(1-f(x)).
// (You can also use rate_list since rate_list[i]=transfer(activity_list[i]).)
double Network::transferDerivative(double activation) const {
    double fx = transfer(activation);
    return fx * (1.0 - fx);
}

// blank initialisation of weight matrix
void Network::blank_init() {
    sum_all_inhib = 0.0;
    activity_list.resize(size, 0.0);
    rate_list.resize(size, transfer(0.0));
    derivative_activity_list.resize(size, 0.0);
    target_sum_each_inhib.resize(size, 0.0);
    actual_sum_each_inhib.resize(size, 0.0);

    weight_matrix.assign(size, std::vector<double>(size, 0.0));
    inhib_matrix.assign(size, std::vector<double>(size, 0.0));
    scale_inhib.assign(size, std::vector<int>(size, 0));

    // Initialize weights/inhib if connectivity is used,
    // but you said we assume "always connected" for depressed iteration.
    // We'll keep the code so that if connectivity_matrix[i][j] == true,
    // we set them. If you truly want everything connected, you can skip this
    // check.
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (connectivity_matrix.empty() || connectivity_matrix[i][j]) {
                weight_matrix[i][j] = 0.0;
                inhib_matrix[i][j] = 0.0;
                scale_inhib[i][j] = 1;
            }
        }
    }
}

void Network::set_state(std::vector<double> new_state){
    for (int i = 0; i < size; i++)
    {
        rate_list[i] = new_state[i];
        activity_list[i] = transfer_inverse(rate_list[i]);
    }
}

void Network::reinforce_attractor(std::vector<double> target_state, double learning_rate)
{
    for (int i = 0; i < size; i++)
    {
        for(int j = 0; j < size; j++){
            if (connectivity_matrix[i][j] == 1) {
                double update = (target_state[j] - rate_list[j]) * target_state[i] * learning_rate;
                weight_matrix[i][j] += update;
                weight_matrix[j][i] += update;
                }
        }
    }
}


// No normalization, doesn't keep the sum of weight of synapses stable.
void Network::pot_inhib(double pot_rate)
{
    actual_sum_each_inhib = std::vector<double>(size,0);
    // Adjust weights
    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < size; ++j)
        {
            if (connectivity_matrix[i][j] == 1)
            {
                inhib_matrix[i][j] += pot_rate*(rate_list[j]*rate_list[i]);
                inhib_matrix[j][i] += pot_rate*(rate_list[j]*rate_list[i]);
            }
        }
    }
    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < size; ++j){
            actual_sum_each_inhib[j] += inhib_matrix[i][j];
        }
    }
}
// TODO inhibitory synapses can be negative so dont need inhib strenght ?
void Network::pot_inhib_symmetric(double pot_rate)
{
    actual_sum_each_inhib = std::vector<double>(size,0);
    // Adjust weights
    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < size; ++j)
        {
            if (connectivity_matrix[i][j] == 1)
            {
                inhib_matrix[i][j] += pot_rate*(rate_list[j]*(rate_list[i]-0.5));
                inhib_matrix[j][i] += pot_rate*(rate_list[j]*(rate_list[i]-0.5));
            }
        }
    }
    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < size; ++j){
            actual_sum_each_inhib[j] += inhib_matrix[i][j];
        }
    }
}
// TODO Not used
void Network::iterative_normalize(int nb_iter_normalize, double rate_normalize){
    std::vector<double> new_sum_each_inhib(size,0);
    for(int iter = 0; iter<nb_iter_normalize; iter++ ){
        for (int i = 0; i < size; ++i)
        {
            for (int j = 0; j < size; ++j)
            {
                if (connectivity_matrix[i][j] == 1)
                {
                    inhib_matrix[i][j] += (target_sum_each_inhib[j]-actual_sum_each_inhib[j])*rate_normalize;
                    inhib_matrix[j][i] += (target_sum_each_inhib[j] - actual_sum_each_inhib[j]) * rate_normalize;
                }
            }
        }
        for (int i = 0; i < size; ++i)
        {
            for (int j = 0; j < size; ++j)
            {
                new_sum_each_inhib[j] += inhib_matrix[i][j];
            }
        }
        for (int i = 0; i < size; ++i){
            actual_sum_each_inhib[i] = new_sum_each_inhib[i];
            new_sum_each_inhib[i] = 0;
        }
    }
}

void Network::reset_inhib()
{
    sum_all_inhib = 0;
    target_sum_each_inhib = std::vector<double>(size, 0.0);
    actual_sum_each_inhib = std::vector<double>(size, 0.0);
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            if (connectivity_matrix[i][j] == 1)
            {
                inhib_matrix[i][j] = inhib_strenght;
                sum_all_inhib += inhib_strenght;
                target_sum_each_inhib[j] += inhib_strenght;
                actual_sum_each_inhib[j] += inhib_strenght;
            }
        }
    }
}

void Network::pot_inhib_bin(double pot_rate, std::vector<bool> winners)
{
    actual_sum_each_inhib = std::vector<double>(size, 0);
    // Adjust weights
    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < size; ++j)
        {
            if (connectivity_matrix[i][j] == 1)
            {
                inhib_matrix[i][j] += pot_rate * (winners[j] * winners[i]);
                inhib_matrix[j][i] += pot_rate * (winners[j] * winners[i]);
            }
        }
    }
    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < size; ++j)
        {
            actual_sum_each_inhib[j] += inhib_matrix[i][j];
        }
    }
}

void Network::pot_inhib_bin_scale(double pot_rate, std::vector<bool> winners)
{
    actual_sum_each_inhib = std::vector<double>(size, 0);
    // Adjust weights
    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < size; ++j)
        {
            if (connectivity_matrix[i][j] == 1)
            {
                if (winners[i] && winners[j])
                {
                    inhib_matrix[i][j] += pot_rate * (winners[j] * winners[i]) * scale_inhib[i][j];
                    inhib_matrix[j][i] += pot_rate * (winners[j] * winners[i]) * scale_inhib[i][j];
                    scale_inhib[i][j] += 1;
                }
            }
        }
    }
    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < size; ++j)
        {
            actual_sum_each_inhib[j] += inhib_matrix[i][j];
        }
    }
}

// --- Compute the Jacobian matrix --- //
std::vector<std::vector<double>> Network::computeJacobian() const {
    // Create an N x N matrix (N = size of the network)
    std::vector<std::vector<double>> jacobian(size,
                                              std::vector<double>(size, 0.0));

    // Loop over rows (output index i) and columns (input index j)
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            // Use f'(x_j). Since rate_list[j] = f(x_j), you can write:
            double fprime = rate_list[j] * (1.0 - rate_list[j]);
            jacobian[i][j] = weight_matrix[i][j] * fprime;
            // Subtract leak on the diagonal
            if (i == j) {
                jacobian[i][j] -= leak;
            }
        }
    }
    return jacobian;
}

// --- Correlate a Jacobian with one pattern --- //
double Network::correlateJacobianWithPattern(
    const std::vector<std::vector<double>>& J,
    const std::vector<bool>& pattern) const {
    double numerator = 0.0;
    double denominator = 0.0;
    // Make sure pattern has the correct dimension
    if (pattern.size() != static_cast<size_t>(size)) {
        std::cerr << "Error: Pattern size does not match network size."
                  << std::endl;
        return 0.0;
    }

    // Compute p^T * J * p and p^T * p
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            numerator += pattern[i] * J[i][j] * pattern[j];
        }
        denominator += pattern[i] * pattern[i];
    }
    return (denominator == 0.0) ? 0.0 : numerator / denominator;
}