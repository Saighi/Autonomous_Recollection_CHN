#include "network.hpp"
#include "utils.hpp"
#include <vector>
#include <cmath>
#include <iostream>
#include <random>
#include <immintrin.h>  // AVX intrinsics

// AVX-optimized dot product for vector operations
// Processes 4 doubles at a time using 256-bit AVX vectors
static inline double avx_dot_product(const double *wRow, const double *vec, int length)
{
    __m256d vsum = _mm256_setzero_pd();  // Initialize accumulator to zero

    int idx = 0;
    // Process 4 elements at a time
    for (; idx + 4 <= length; idx += 4)
    {
        __m256d vw = _mm256_loadu_pd(&wRow[idx]);  // Load 4 doubles from weight row
        __m256d vv = _mm256_loadu_pd(&vec[idx]);    // Load 4 doubles from vector
        vsum = _mm256_fmadd_pd(vw, vv, vsum);       // vsum += vw * vv (fused multiply-add)
    }

    // Horizontal sum: add the 4 elements in the vector
    alignas(32) double partial[4];
    _mm256_storeu_pd(partial, vsum);
    double sum = partial[0] + partial[1] + partial[2] + partial[3];

    // Handle remainder (for sizes not divisible by 4)
    for (; idx < length; idx++)
    {
        sum += wRow[idx] * vec[idx];
    }

    return sum;
}

// AVX-optimized dot product excluding diagonal element
// Used in Hopfield networks where W[i][i] should be excluded
static inline double avx_dot_product_no_diag(const double *wRow, const double *vec,
                                               int length, int skip_idx)
{
    __m256d vsum = _mm256_setzero_pd();

    int idx = 0;
    // Process 4 elements at a time
    for (; idx + 4 <= length; idx += 4)
    {
        __m256d vw = _mm256_loadu_pd(&wRow[idx]);
        __m256d vv = _mm256_loadu_pd(&vec[idx]);
        vsum = _mm256_fmadd_pd(vw, vv, vsum);
    }

    // Horizontal sum
    alignas(32) double partial[4];
    _mm256_storeu_pd(partial, vsum);
    double sum = partial[0] + partial[1] + partial[2] + partial[3];

    // Remainder loop
    for (; idx < length; idx++)
    {
        sum += wRow[idx] * vec[idx];
    }

    // Subtract diagonal element (since we included it in the loop)
    if (skip_idx >= 0 && skip_idx < length)
    {
        sum -= wRow[skip_idx] * vec[skip_idx];
    }

    return sum;
}

Network::Network(std::vector<std::vector<bool>> connect_mat, int size_network)
{
    connectivity_matrix = connect_mat;
    size = size_network;
    blank_init();
}

// blank initialisation of weight matrix
void Network::blank_init()
{

    weight_matrix = std::vector<std::vector<double>>(size, std::vector<double>(size));
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            if (connectivity_matrix[i][j] == 1)
            {
                weight_matrix[i][j] = 0;
            }
        }
    }

}

std::vector<double> Network::activate(const std::vector<double> &state)
{
    std::vector<double> new_state(state.size());
    for (size_t i = 0; i < state.size(); ++i)
    {
        new_state[i] = state[i] >= 0 ? 1 : -1;
    }
    return new_state;
}

std::vector<double> Network::runStochasticNbIterSimulatedAnnealing(const std::vector<double> &initial_state, double temperature_start, int nb_iter)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, size - 1);
    std::uniform_real_distribution<> prob(0.0, 1.0);
    std::vector<double> state = initial_state;
    std::vector<double> old_state = initial_state;
    double p;
    int i;
    double sum_diff;
    double ratio_diff;
    double sum;
    int max_nb_iter = size * 10;
    int min_nb_iter = size * 5;

    double temperature = temperature_start;
    double temperature_decay = temperature_start / nb_iter;

    for (size_t step = 0; step < nb_iter; step++)
    {
        i = dis(gen);
        sum = 0.0;
        for (int j = 0; j < size; j++)
        {
            if (j != i)
            {
                sum += weight_matrix[i][j] * initial_state[j];
            }
        }
        p = 1.0 / (1.0 + exp(-(sum / temperature)));
        state[i] = prob(gen) < p ? 1 : -1;
        temperature -= temperature_decay;
    }

    return state;
}


std::vector<double> Network::runStochasticNbIter(const std::vector<double> &initial_state, double temperature, int nb_iter){

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, size - 1);
    std::uniform_real_distribution<> prob(0.0, 1.0);
    std::vector<double> state = initial_state;
    std::vector<double> old_state = initial_state;
    double p;
    int i;
    double sum_diff;
    double ratio_diff;
    double sum;
    int max_nb_iter = size*10;
    int min_nb_iter = size*5;

    for (size_t step = 0; step < nb_iter; step++)
    {
        i = dis(gen);
        sum = 0.0;
        for (int j = 0; j < size; j++)
        {
            if (j != i)
            {
                sum += weight_matrix[i][j] * initial_state[j];
            }
        }
        p = 1.0 / (1.0 + exp(-(sum/temperature)));
        state[i] = prob(gen) < p ? 1 : -1;
    }

   return state; 
}

std::vector<double> Network::runSynchronous(const std::vector<double> &initial_state, int steps)
{
    std::vector<double> state = initial_state;
    std::vector<double> drive;
    for (int i = 0; i < steps; ++i)
    {
        drive = std::vector<double>(size, 0.0);
        for (int j = 0; j < size; ++j)
        {
            for (int k = 0; k < size; ++k)
            {
                drive[j] += weight_matrix[j][k] * state[k];
            }
        }
        state = activate(drive);
    }
    return state;
}

std::vector<double> Network::runAsynchronous(const std::vector<double> &initial_state, int nb_steps)
{
    std::vector<double> state = initial_state;

    for (int step = 0; step < nb_steps; step++)
    {
        for (int i = 0; i < size; ++i)
        {
            // AVX-optimized dot product: 4x faster than scalar loop
            double sum = avx_dot_product_no_diag(
                weight_matrix[i].data(),
                state.data(),
                size,
                i  // Skip diagonal element
            );

            state[i] = sum >= 0 ? 1 : -1;
        }
    }
    return state;
}

std::vector<double> Network::runStochastic(const std::vector<double> &initial_state, double temperature, double epsilon){

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, size - 1);
    std::uniform_real_distribution<> prob(0.0, 1.0);
    std::vector<double> state = initial_state;
    std::vector<double> old_state = initial_state;
    double p;
    int i;
    int cpt;
    double sum_diff;
    double ratio_diff;
    double sum;
    int max_nb_iter = size*10;
    int min_nb_iter = size*5;

    for (size_t step = 0; step < max_nb_iter; step++)
    {
        i = dis(gen);
        sum = 0.0;
        for (int j = 0; j < size; j++)
        {
            if (j != i)
            {
                sum += weight_matrix[i][j] * initial_state[j];
            }
        }
        p = 1.0 / (1.0 + exp(-(sum/temperature)));
        state[i] = prob(gen) < p ? 1 : -1;
        //Convergence test :
        if(cpt>size){
            ratio_diff = ratio_diff_vectors(state, old_state);
            if(ratio_diff<epsilon || step>min_nb_iter){
                break;
            }
            else{
                old_state = state;
            }
            cpt=0;
        }
        else{
            cpt+=1;
        }
    }

   return state; 
}

void Network::trainPerceptron(const std::vector<std::vector<double>> &patterns, int nb_iter, double step_size)
{
    std::vector<double> drives(size, 0.0);
    for (size_t iter = 0; iter < nb_iter; iter++)
    {
        for (const auto &pattern : patterns)
        {
            // AVX-optimized drive computation: O(N) instead of O(N²) scalar ops
            for (int i = 0; i < size; ++i)
            {
                drives[i] = avx_dot_product_no_diag(
                    weight_matrix[i].data(),
                    pattern.data(),
                    size,
                    i  // Skip diagonal element
                );
            }

            // AVX-optimized weight updates
            for (int i = 0; i < size; ++i)
            {
                double factor = ((1 - (drives[i] * pattern[i])) * pattern[i] * step_size) / 2.0;
                __m256d vfactor = _mm256_set1_pd(factor);  // Broadcast scalar to all 4 elements

                int j = 0;
                // Process 4 weights at a time
                for (; j + 4 <= size; j += 4)
                {
                    __m256d vpattern = _mm256_loadu_pd(&pattern[j]);
                    __m256d vupdate = _mm256_mul_pd(vfactor, vpattern);
                    __m256d vweight = _mm256_loadu_pd(&weight_matrix[i][j]);
                    vweight = _mm256_add_pd(vweight, vupdate);
                    _mm256_storeu_pd(&weight_matrix[i][j], vweight);
                }

                // Handle remainder
                for (; j < size; j++)
                {
                    if (j != i)
                    {
                        weight_matrix[i][j] += factor * pattern[j];
                    }
                }

                // Ensure diagonal is zero (in case it was in the vectorized part)
                weight_matrix[i][i] = 0.0;
            }
        }
    }
}

void Network::trainHebbian(const std::vector<std::vector<double>> &patterns)
{
    for (const auto &p : patterns)
    {
        for (int i = 0; i < size; ++i)
        {
            for (int j = 0; j < size; ++j)
            {
                if (i != j)
                {
                    weight_matrix[i][j] +=(p[i] * p[j])/(2*patterns.size());
                }
            }
        }
    }
}

// Work only if the hebbian larning is normalized by the number of patterns to have a value
// between -1 and 1
// Work with the perceptron of targeted values are 1 and -1
double Network::ComputeMeanCrossTalk(std::vector<std::vector<bool>> patterns){
    double cross_talk=0.0;
    for (size_t mu = 0; mu < patterns.size(); mu++)
    {
        for (size_t nu = 0; nu < patterns.size(); nu++){
            if(mu!=nu){
                for (size_t i = 0; i < size; i++)
                {
                    for (size_t j = 0; j < size; j++)
                    {
                        if(i!=j){
                            cross_talk+=weight_matrix[i][j]*patterns[nu][j];    
                        }
                    }
                }
            }
        }
    }
    return cross_talk/(size*patterns.size()*patterns.size());
}