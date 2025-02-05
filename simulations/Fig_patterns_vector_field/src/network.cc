#include "network.hpp"

#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "utils.hpp"

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

        double d = sum_weight - sum_inhib - (leak * activity_list[i]) + noise;
        activity_list[i] += delta * d;
        rate_list[i] = transfer(activity_list[i]);
    }
}

double Network::transfer(double activation) {
    return 1.0 / (1.0 + std::exp(-activation));
}

double Network::transfer_inverse(double activation) {
    return -std::log(-1.0 + 1.0 / activation);
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

void Network::set_state(std::vector<double> new_state) {
    for (int i = 0; i < size; i++) {
        rate_list[i] = new_state[i];
        activity_list[i] = transfer_inverse(rate_list[i]);
    }
}

void Network::reinforce_attractor(std::vector<double> target_state,
                                  double learning_rate) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (connectivity_matrix[i][j] == 1) {
                double update = (target_state[j] - rate_list[j]) *
                                target_state[i] * learning_rate;
                weight_matrix[i][j] += update;
                weight_matrix[j][i] += update;
            }
        }
    }
}

void Network::reinforce_attractor_bin(std::vector<bool> binary_state,
                                      double learning_rate) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (connectivity_matrix[i][j] == 1) {
                double update =
                    binary_state[j] * binary_state[i] * learning_rate;
                weight_matrix[i][j] += update;
                weight_matrix[j][i] += update;
            }
        }
    }
}

// No normalization, doesn't keep the sum of weight of synapses stable.
void Network::pot_inhib(double pot_rate) {
    actual_sum_each_inhib = std::vector<double>(size, 0);
    // Adjust weights
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            if (connectivity_matrix[i][j] == 1) {
                inhib_matrix[i][j] += pot_rate * (rate_list[j] * rate_list[i]);
                inhib_matrix[j][i] += pot_rate * (rate_list[j] * rate_list[i]);
            }
        }
    }
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            actual_sum_each_inhib[j] += inhib_matrix[i][j];
        }
    }
}
// TODO inhibitory synapses can be negative so dont need inhib strenght ?
void Network::pot_inhib_symmetric(double pot_rate) {
    actual_sum_each_inhib = std::vector<double>(size, 0);
    // Adjust weights
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            if (connectivity_matrix[i][j] == 1) {
                inhib_matrix[i][j] +=
                    pot_rate * (rate_list[j] * (rate_list[i] - 0.5));
                inhib_matrix[j][i] +=
                    pot_rate * (rate_list[j] * (rate_list[i] - 0.5));
            }
        }
    }
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            actual_sum_each_inhib[j] += inhib_matrix[i][j];
        }
    }
}
// TODO Not used
void Network::iterative_normalize(int nb_iter_normalize,
                                  double rate_normalize) {
    std::vector<double> new_sum_each_inhib(size, 0);
    for (int iter = 0; iter < nb_iter_normalize; iter++) {
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                if (connectivity_matrix[i][j] == 1) {
                    inhib_matrix[i][j] +=
                        (target_sum_each_inhib[j] - actual_sum_each_inhib[j]) *
                        rate_normalize;
                    inhib_matrix[j][i] +=
                        (target_sum_each_inhib[j] - actual_sum_each_inhib[j]) *
                        rate_normalize;
                }
            }
        }
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                new_sum_each_inhib[j] += inhib_matrix[i][j];
            }
        }
        for (int i = 0; i < size; ++i) {
            actual_sum_each_inhib[i] = new_sum_each_inhib[i];
            new_sum_each_inhib[i] = 0;
        }
    }
}

void Network::reset_inhib() {
    sum_all_inhib = 0;
    target_sum_each_inhib = std::vector<double>(size, 0.0);
    actual_sum_each_inhib = std::vector<double>(size, 0.0);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (connectivity_matrix[i][j] == 1) {
                inhib_matrix[i][j] = inhib_strenght;
                sum_all_inhib += inhib_strenght;
                target_sum_each_inhib[j] += inhib_strenght;
                actual_sum_each_inhib[j] += inhib_strenght;
            }
        }
    }
}

void Network::pot_inhib_bin(double pot_rate, std::vector<bool> winners) {
    actual_sum_each_inhib = std::vector<double>(size, 0);
    // Adjust weights
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            if (connectivity_matrix[i][j] == 1) {
                inhib_matrix[i][j] += pot_rate * (winners[j] * winners[i]);
                inhib_matrix[j][i] += pot_rate * (winners[j] * winners[i]);
            }
        }
    }
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            actual_sum_each_inhib[j] += inhib_matrix[i][j];
        }
    }
}

void Network::pot_inhib_bin_scale(double pot_rate, std::vector<bool> winners) {
    actual_sum_each_inhib = std::vector<double>(size, 0);
    // Adjust weights
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            if (connectivity_matrix[i][j] == 1) {
                if (winners[i] && winners[j]) {
                    inhib_matrix[i][j] += pot_rate * (winners[j] * winners[i]) *
                                          scale_inhib[i][j];
                    inhib_matrix[j][i] += pot_rate * (winners[j] * winners[i]) *
                                          scale_inhib[i][j];
                    scale_inhib[i][j] += 1;
                }
            }
        }
    }
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            actual_sum_each_inhib[j] += inhib_matrix[i][j];
        }
    }
}

// gradient descent code :

void Network::derivative_gradient_descent(std::vector<bool>& target_bin_state,
                                          std::vector<double>& target_rates,
                                          double target_drive,
                                          double learning_rate, double leak,
                                          std::vector<double>& drive_errors) {
    double input = 0;
    double ui;
    double update;
    double unit_target_drive;
    double diff;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (connectivity_matrix[i][j] == 1) {
                input += weight_matrix[i][j] * target_rates[j];
            }
        }
        // std::cout << input << std::endl;
        ui = input / leak;
        unit_target_drive = ((target_bin_state[i] * 2) - 1) *
                            target_drive;  // target drive of the unit
        diff = unit_target_drive - ui;
        drive_errors[i] = diff;
        for (int j = 0; j < size; j++) {
            if (connectivity_matrix[i][j] == 1) {
                update = learning_rate * 2 * diff * target_rates[j];
                weight_matrix[i][j] += update;
                weight_matrix[j][i] += update;
            }
        }
        input = 0;
    }
}

void Network::derivative_gradient_descent_with_momentum(
    std::vector<bool>& target_bin_state, std::vector<double>& target_rates,
    double target_drive, double learning_rate, double leak,
    std::vector<double>& drive_errors,
    std::vector<std::vector<double>>& velocity_matrix, double momentum_coef) {
    double input = 0;
    double ui;
    double update;
    double unit_target_drive;
    double diff;

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (connectivity_matrix[i][j] == 1) {
                input += weight_matrix[i][j] * target_rates[j];
            }
        }

        ui = input / leak;
        unit_target_drive = ((target_bin_state[i] * 2) - 1) * target_drive;
        diff = unit_target_drive - ui;
        drive_errors[i] = diff;

        for (int j = 0; j < size; j++) {
            if (connectivity_matrix[i][j] == 1) {
                // Calculate gradient update
                update = learning_rate * 2 * diff * target_rates[j];

                // Apply momentum update
                velocity_matrix[i][j] =
                    momentum_coef * velocity_matrix[i][j] + update;
                velocity_matrix[j][i] =
                    velocity_matrix[i][j];  // Maintain symmetry

                // Update weights with momentum
                weight_matrix[i][j] += velocity_matrix[i][j];
                weight_matrix[j][i] = weight_matrix[i][j];  // Maintain symmetry
            }
        }
        input = 0;
    }
}

// Basic iteration (no depression, no noise)
std::vector<double> Network::give_derivative_u() {
    std::vector<double> derivative_u(size,0.0);

    // Compute derivative for each neuron i
    for (int i = 0; i < size; i++) {
        // sum of excitatory or "normal" synapses
        double sum_weight =
            avx_dot_product(weight_matrix[i].data(), rate_list.data(), size);

        // sum of inhibitory synapses
        double sum_inhib =
            avx_dot_product(inhib_matrix[i].data(), rate_list.data(), size);

        double d = sum_weight - sum_inhib - (leak * activity_list[i]);
        derivative_u[i] = d;
    }

    return derivative_u;
}

std::vector<double> Network::give_derivative_v(std::vector<double> derivative_u) {

    std::vector<double> derivative_v(size,0.0);
    // Now multiply by derivative of transfer function for each neuron
    for (int i = 0; i < size; i++) {
        // Calculate v = sigmoid(u) first
        double v = rate_list[i];  // Assuming rate_list stores the v values

        // Calculate sigmoid'(u) = v * (1-v)
        double sigmoid_derivative = v * (1.0 - v);
        // Multiply the u derivative by sigmoid'(u)
        derivative_v[i] = sigmoid_derivative * derivative_u[i];
    }
    return derivative_v;
}

void Network::derivative_gradient_descent_with_momentum_null_sum(
    std::vector<bool>& target_bin_state, std::vector<double>& target_rates,
    double target_drive, double learning_rate, double leak,
    std::vector<double>& drive_errors,
    std::vector<std::vector<double>>& velocity_matrix, double momentum_coef,
    double alpha_homeo) {
    double input = 0;
    double ui;
    double update;
    double unit_target_drive;
    double diff;
    double sum_weights;

    for (int i = 0; i < size; i++) {
        input = 0;
        for (int j = 0; j < size; j++) {
            if (connectivity_matrix[i][j] == 1) {
                input += weight_matrix[i][j] * target_rates[j];
            }
        }
        ui = input / leak;
        unit_target_drive = ((target_bin_state[i] * 2) - 1) * target_drive;
        diff = unit_target_drive - ui;
        drive_errors[i] = diff;

        for (int j = 0; j < size; j++) {
            if (connectivity_matrix[i][j] == 1) {
                // Calculate gradient update
                update = learning_rate * 2 * diff * target_rates[j];

                // Apply momentum update
                velocity_matrix[i][j] =
                    momentum_coef * velocity_matrix[i][j] + update;
                velocity_matrix[j][i] =
                    velocity_matrix[i][j];  // Maintain symmetry

                // Update weights with momentum
                weight_matrix[i][j] += velocity_matrix[i][j];
                weight_matrix[j][i] = weight_matrix[i][j];  // Maintain symmetry
            }
        }
    }

    // ------------------------------------------------------
    // 2) Force sum_j w_{i,j} = 0 by post-update normalization
    // ------------------------------------------------------
    // Typically you want to do this with a small correction,
    // but if you want exact sum = 0 each time, do it directly:
    for (int i = 0; i < size; i++) {
        // Calculate the sum of row i
        double row_sum = 0.0;
        for (int j = 0; j < size; j++) {
            if (connectivity_matrix[i][j] == 1) {
                row_sum += weight_matrix[i][j];
            }
        }
        // Distribute a negative "average" across each w_{i,j}
        double correction = alpha_homeo * (row_sum / double(size));

        // Subtract the correction from each w_{i,j}
        for (int j = 0; j < size; j++) {
            if (connectivity_matrix[i][j] == 1) {
                weight_matrix[i][j] -= correction;
                // keep symmetry
                weight_matrix[j][i] = weight_matrix[i][j];
            }
        }
    }
}
