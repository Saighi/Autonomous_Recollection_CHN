#include "network.hpp"
#include "utils.hpp"
#include <vector>
#include <cmath>
#include <iostream>
#include <random>

// AVX helper: Dot-product for double-precision arrays
// Summation of (wRow[j] * rate[j]) for all j
// Processes 4 doubles at a time using AVX2 intrinsics
static inline double avx_dot_product(const double* wRow, const double* rate, int length) {
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

    // Handle remainder elements
    for (; idx < length; idx++) {
        sum += wRow[idx] * rate[idx];
    }

    return sum;
}

Network::Network(std::vector<std::vector<bool>> connect_mat, int size_network, double lk)
{
    leak = lk;
    connectivity_matrix = connect_mat;
    size = size_network;
    inhib_strenght = 10;

    // Initialize the random engine once (avoid re-init each call)
    std::random_device rd;
    generator.seed(rd());

    blank_init();
}

void Network::iterate(double delta)
{
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            derivative_activity_list[i] += weight_matrix[i][j] * rate_list[j];
        }
    }

    for (int i = 0; i < size; i++)
    {
        activity_list[i] += delta * (derivative_activity_list[i] - (leak * activity_list[i]));
        rate_list[i] = transfer(activity_list[i]);
    };

    std::fill(derivative_activity_list.begin(), derivative_activity_list.end(), 0);
}

void Network::iterate_query_drive(double delta, double strength_drive, std::vector<double>& query_drives)
{
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            derivative_activity_list[i] += weight_matrix[i][j] * rate_list[j];
        }
    }

    for (int i = 0; i < size; i++)
    {
        activity_list[i] += delta*(derivative_activity_list[i]-(leak * activity_list[i])+strength_drive*(query_drives[i]-activity_list[i]));
        rate_list[i] = transfer(activity_list[i]);
    };

    std::fill(derivative_activity_list.begin(), derivative_activity_list.end(), 0);
}

// Noisy iteration: add Gaussian noise to derivative (AVX optimized)
void Network::noisy_iterate(double delta, double mean, double stddev)
{
    std::normal_distribution<double> distribution(mean, stddev);

    for (int i = 0; i < size; i++)
    {
        double sum = avx_dot_product(weight_matrix[i].data(), rate_list.data(), size);
        double noise = distribution(generator);
        double d = sum - (leak * activity_list[i]) + noise;
        activity_list[i] += delta * d;
        rate_list[i] = transfer(activity_list[i]);
    }
}

// Depressed iteration without noise
void Network::depressed_iterate(double delta)
{
    for (int i = 0; i < size; i++)
    {
        double sum_weight = avx_dot_product(weight_matrix[i].data(), rate_list.data(), size);
        double sum_inhib = inhib_matrix[i][i] * rate_list[i];

        double d = sum_weight - sum_inhib - (leak * activity_list[i]);
        activity_list[i] += delta * d;
        rate_list[i] = transfer(activity_list[i]);
    }
}

// Noisy depressed iteration with diagonal-only inhibition (AVX optimized)
void Network::noisy_depressed_iterate(double delta, double mean, double stddev)
{
    std::normal_distribution<double> distribution(mean, stddev);

    for (int i = 0; i < size; i++)
    {
        double noise = distribution(generator);
        double sum_weight = avx_dot_product(weight_matrix[i].data(), rate_list.data(), size);
        double sum_inhib = inhib_matrix[i][i] * rate_list[i];

        double d = sum_weight - sum_inhib - (leak * activity_list[i]) + noise;
        activity_list[i] += delta * d;
        rate_list[i] = transfer(activity_list[i]);
    }
}

double Network::transfer(double activation)
{
    return 1.0 / (1.0 + std::exp(-activation));
}

double Network::transfer_inverse(double activation)
{
    return -std::log(-1.0 + 1.0 / activation);
}

// Blank initialization of weight matrix
void Network::blank_init()
{
    sum_all_inhib = 0.0;
    activity_list.assign(size, 0.0);
    rate_list.assign(size, transfer(0.0));
    derivative_activity_list.assign(size, 0.0);
    target_sum_each_inhib.assign(size, 0.0);
    actual_sum_each_inhib.assign(size, 0.0);
    bias.assign(size, 0.0);

    weight_matrix.assign(size, std::vector<double>(size, 0.0));
    inhib_matrix.assign(size, std::vector<double>(size, 0.0));
    scale_inhib.assign(size, std::vector<int>(size, 0));

    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            if (connectivity_matrix.empty() || connectivity_matrix[i][j])
            {
                weight_matrix[i][j] = 0.0;
                inhib_matrix[i][j] = 0.0;
                scale_inhib[i][j] = 1;
            }
        }
    }
}

void Network::set_state(std::vector<double> new_state)
{
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
        for (int j = 0; j < size; j++)
        {
            if (connectivity_matrix[i][j] == 1)
            {
                double update = (target_state[j] - rate_list[j]) * target_state[i] * learning_rate;
                weight_matrix[i][j] += update;
                weight_matrix[j][i] += update;
            }
        }
    }
}

void Network::derivative_gradient_descent(std::vector<bool>& target_bin_state, std::vector<double>& target_rates,
                                          double target_drive, double learning_rate, double leak,
                                          std::vector<double>& drive_errors)
{
    double input = 0;
    double ui;
    double update;
    double unit_target_drive;
    double diff;
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            if (connectivity_matrix[i][j] == 1)
            {
                input += weight_matrix[i][j] * target_rates[j];
            }
        }
        ui = input / leak;
        unit_target_drive = ((target_bin_state[i] * 2) - 1) * target_drive;
        diff = unit_target_drive - ui;
        drive_errors[i] = diff;
        for (int j = 0; j < size; j++)
        {
            if (connectivity_matrix[i][j] == 1)
            {
                update = learning_rate * 2 * diff * target_rates[j];
                weight_matrix[i][j] += update;
                weight_matrix[j][i] += update;
            }
        }
        input = 0;
    }
}

void Network::derivative_gradient_descent_with_momentum(std::vector<bool>& target_bin_state,
                                                        std::vector<double>& target_rates,
                                                        double target_drive,
                                                        double learning_rate,
                                                        double leak,
                                                        std::vector<double>& drive_errors,
                                                        std::vector<std::vector<double>>& velocity_matrix,
                                                        double momentum_coef)
{
    double input = 0;
    double ui;
    double update;
    double unit_target_drive;
    double diff;

    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            if (connectivity_matrix[i][j] == 1)
            {
                input += weight_matrix[i][j] * target_rates[j];
            }
        }

        ui = input / leak;
        unit_target_drive = ((target_bin_state[i] * 2) - 1) * target_drive;
        diff = unit_target_drive - ui;
        drive_errors[i] = diff;

        for (int j = 0; j < size; j++)
        {
            if (connectivity_matrix[i][j] == 1)
            {
                update = learning_rate * 2 * diff * target_rates[j];
                velocity_matrix[i][j] = momentum_coef * velocity_matrix[i][j] + update;
                velocity_matrix[j][i] = velocity_matrix[i][j];
                weight_matrix[i][j] += velocity_matrix[i][j];
                weight_matrix[j][i] = weight_matrix[i][j];
            }
        }
        input = 0;
    }
}

void Network::derivative_gradient_descent_with_bias(std::vector<double>& target_drives,
                                                    double learning_rate,
                                                    double leak,
                                                    std::vector<double>& drive_errors)
{
    double input = 0;
    double ui;
    double update;
    double unit_target_drive;
    double diff;
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            if (connectivity_matrix[i][j] == 1)
            {
                input += weight_matrix[i][j] * transfer(target_drives[j]);
            }
        }
        ui = (input + bias[i]) / leak;
        unit_target_drive = target_drives[i];
        diff = unit_target_drive - ui;
        drive_errors[i] = diff;
        for (int j = 0; j < size; j++)
        {
            if (connectivity_matrix[i][j] == 1)
            {
                update = learning_rate * 2 * diff * transfer(target_drives[j]);
                weight_matrix[i][j] += update;
            }
        }
        bias[i] += learning_rate * 2 * diff;
        input = 0;
    }
}

void Network::derivative_gradient_descent_with_bias_and_momentum(std::vector<double>& target_drives,
                                                                  double learning_rate,
                                                                  double leak,
                                                                  std::vector<double>& drive_errors,
                                                                  std::vector<std::vector<double>>& velocity_matrix,
                                                                  std::vector<double>& velocity_bias,
                                                                  double momentum_coef)
{
    double input = 0;
    double ui;
    double update;
    double unit_target_drive;
    double diff;

    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            if (connectivity_matrix[i][j] == 1)
            {
                input += weight_matrix[i][j] * transfer(target_drives[j]);
            }
        }

        ui = (input + bias[i]) / leak;
        unit_target_drive = target_drives[i];
        diff = unit_target_drive - ui;
        drive_errors[i] = diff;

        for (int j = 0; j < size; j++)
        {
            if (connectivity_matrix[i][j] == 1)
            {
                update = learning_rate * 2 * diff * transfer(target_drives[j]);
                velocity_matrix[i][j] = momentum_coef * velocity_matrix[i][j] + update;
                weight_matrix[i][j] += velocity_matrix[i][j];
            }
        }

        double bias_gradient = learning_rate * 2 * diff;
        velocity_bias[i] = momentum_coef * velocity_bias[i] + bias_gradient;
        bias[i] += velocity_bias[i];

        input = 0;
    }
}

// AVX-accelerated gradient descent with bias and momentum
void Network::derivative_gradient_descent_with_bias_and_momentum_avx(std::vector<double>& target_drives,
                                                                      double learning_rate,
                                                                      double leak,
                                                                      std::vector<double>& drive_errors,
                                                                      std::vector<std::vector<double>>& velocity_matrix,
                                                                      std::vector<double>& velocity_bias,
                                                                      double momentum_coef)
{
    // Pre-compute target rates
    std::vector<double> target_rates(size);
    for (int j = 0; j < size; j++)
    {
        target_rates[j] = transfer(target_drives[j]);
    }

    for (int i = 0; i < size; i++)
    {
        double input = avx_dot_product(weight_matrix[i].data(), target_rates.data(), size);

        double ui = (input + bias[i]) / leak;
        double unit_target_drive = target_drives[i];
        double diff = unit_target_drive - ui;
        drive_errors[i] = diff;

        double lr_2_diff = learning_rate * 2 * diff;
        __m256d v_lr_2_diff = _mm256_set1_pd(lr_2_diff);
        __m256d v_momentum = _mm256_set1_pd(momentum_coef);

        int j = 0;
        for (; j + 4 <= size; j += 4)
        {
            __m256d v_trate = _mm256_loadu_pd(&target_rates[j]);
            __m256d v_grad = _mm256_mul_pd(v_lr_2_diff, v_trate);
            __m256d v_vel = _mm256_loadu_pd(&velocity_matrix[i][j]);
            v_vel = _mm256_fmadd_pd(v_momentum, v_vel, v_grad);
            _mm256_storeu_pd(&velocity_matrix[i][j], v_vel);
            __m256d v_weight = _mm256_loadu_pd(&weight_matrix[i][j]);
            v_weight = _mm256_add_pd(v_weight, v_vel);
            _mm256_storeu_pd(&weight_matrix[i][j], v_weight);
        }

        for (; j < size; j++)
        {
            double update = lr_2_diff * target_rates[j];
            velocity_matrix[i][j] = momentum_coef * velocity_matrix[i][j] + update;
            weight_matrix[i][j] += velocity_matrix[i][j];
        }

        double bias_gradient = learning_rate * 2 * diff;
        velocity_bias[i] = momentum_coef * velocity_bias[i] + bias_gradient;
        bias[i] += velocity_bias[i];
    }
}

void Network::rate_derivative_gradient_descent(std::vector<double> target_rate, double learning_rate, double leak)
{
    double input = 0;
    double ui;
    double vi;
    double update;
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            if (connectivity_matrix[i][j] == 1)
            {
                input += weight_matrix[i][j] * target_rate[j];
            }
        }
        ui = input / leak;
        vi = transfer(ui);
        for (int j = 0; j < size; j++)
        {
            if (connectivity_matrix[i][j] == 1)
            {
                update = (target_rate[i] - vi) * vi * (1 - vi) * target_rate[i] * learning_rate;
                weight_matrix[i][j] += update;
                weight_matrix[j][i] += update;
            }
        }
        input = 0;
    }
}

// No normalization, doesn't keep the sum of weight of synapses stable.
void Network::pot_inhib(double pot_rate)
{
    actual_sum_each_inhib = std::vector<double>(size, 0);
    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < size; ++j)
        {
            if (connectivity_matrix[i][j] == 1)
            {
                inhib_matrix[i][j] += pot_rate * (rate_list[j] * rate_list[i]);
                inhib_matrix[j][i] += pot_rate * (rate_list[j] * rate_list[i]);
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

// Diagonal inhibition potentiation (used in sleep simulations)
void Network::pot_inhib_symmetric(double pot_rate)
{
    actual_sum_each_inhib = std::vector<double>(size, 0);
    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < size; ++j)
        {
            if (i == j)
            {
                inhib_matrix[i][j] += pot_rate * (rate_list[j] * (rate_list[i] - 0.5));
                inhib_matrix[j][i] += pot_rate * (rate_list[j] * (rate_list[i] - 0.5));
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

void Network::iterative_normalize(int nb_iter_normalize, double rate_normalize)
{
    std::vector<double> new_sum_each_inhib(size, 0);
    for (int iter = 0; iter < nb_iter_normalize; iter++)
    {
        for (int i = 0; i < size; ++i)
        {
            for (int j = 0; j < size; ++j)
            {
                if (connectivity_matrix[i][j] == 1)
                {
                    inhib_matrix[i][j] += (target_sum_each_inhib[j] - actual_sum_each_inhib[j]) * rate_normalize;
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
        for (int i = 0; i < size; ++i)
        {
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
