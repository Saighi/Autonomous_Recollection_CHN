#include <matio.h>
#include <algorithm>
#include <condition_variable>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <numeric>
#include <random>
#include <set>
#include <thread>
#include <unordered_map>
#include <vector>

#include "network.hpp"
#include "utils.hpp"

using namespace std;

namespace fs = std::filesystem;

void run_simulation(int sim_number, unordered_map<string, double> parameters,
                    const string foldername_results) {
    // Learning constants
    double epsilon_learning = parameters.at("epsilon_learning");
    double drive_target = parameters.at("drive_target");
    double learning_rate = parameters.at("learning_rate");
    int network_size = 20;
    double leak = parameters.at("leak");
    double delta = parameters.at("delta");
    double noise_level = parameters.at("noise_level");
    double beta = parameters.at("beta");
    int nb_sample_points_vector_field =
        parameters.at("nb_sample_points_vector_field");
    double up_lim_vector_field = parameters.at("up_lim_vector_field");
    std::cout << up_lim_vector_field << std::endl;
    string sim_data_foldername;
    string result_file_name;

    sim_data_foldername =
        foldername_results + "/sim_nb_" + to_string(sim_number);

    // Create directory if it doesn't exist
    if (!fs::exists(sim_data_foldername)) {
        if (!fs::create_directory(sim_data_foldername)) {
            std::cerr << "Error creating directory: " << sim_data_foldername
                      << std::endl;
            return;
        }
    }
    createParameterFile(sim_data_foldername, parameters);

    // Build Fully connected network
    vector<vector<bool>> connectivity_matrix(network_size,
                                             vector<bool>(network_size, false));
    for (int i = 0; i < network_size; i++) {
        for (int j = 0; j < network_size; j++) {
            if (i != j) {
                connectivity_matrix[i][j] = true;
            }
        }
    }

    Network net = Network(connectivity_matrix, network_size, leak);
    //---------------------------------------------------------- Generate data
    std::string patterns_file_name = sim_data_foldername + "/patterns.data";
    std::ofstream file(patterns_file_name, std::ios::trunc);
    vector<vector<bool>> patterns =
        generatePatterns(2, network_size, network_size / 2, noise_level);
    for (int i = 0; i < 2; i++) {
        writeBoolToCSV(file, patterns[i]);
        // show_vector_bool_grid(initial_patterns[i], col_with);
    }
    vector<vector<double>> patterns_rates = patterns_as_states(
        net.transfer(drive_target), net.transfer(-drive_target), patterns);
    vector<vector<double>> patterns_potentials= patterns_as_states(
        drive_target,-drive_target, patterns);

    // compute_and_save_rate_vector_field_two_pattern_bilinear(delta,
    //     net, sim_data_foldername, "pre_train_interpolate_plane",
    //     patterns_rates[0], patterns_rates[1], nb_sample_points_vector_field);
    compute_and_save_potential_vector_field_two_pattern(
        delta, net, sim_data_foldername, "pre_train", patterns_potentials[0],
        patterns_potentials[1], nb_sample_points_vector_field,
        up_lim_vector_field);

    //---------------------------------------------------------- Training
    std::cout << "WRITING THE ATTRACTOR" << std::endl;
    // Initialize velocity matrix for momentum
    std::vector<std::vector<double>> velocity_matrix(
        network_size, std::vector<double>(network_size, 0.0));
    double momentum_coef = 0.9;  // You can adjust this value
    vector<double> drives_error;
    drives_error.resize(network_size, 0.0);
    // Training loop
    double max_error = 1000;
    int cpt = 0;
    while (max_error > epsilon_learning && cpt <= 10 / learning_rate) {
        for (int j = 0; j < patterns.size(); j++) {
            net.derivative_gradient_descent(patterns[j], patterns_rates[j],
                                            drive_target, learning_rate, leak,
                                            drives_error);
            // net.derivative_gradient_descent_with_momentum(
            //     patterns[j], patterns_rates[j], drive_target,
            //     learning_rate, leak, drives_error, velocity_matrix,
            //     momentum_coef);
        }
        max_error = std::abs(
            *std::max_element(drives_error.begin(), drives_error.end()));
        cpt += 1;
    }
    std::cout << "nombre d'iterations : " << std::endl;
    std::cout << cpt << std::endl;
    // net.reinforce_attractor_bin(patterns[1],0.1);
    // compute_and_save_rate_vector_field_two_pattern_bilinear(delta,
    //     net, sim_data_foldername, "post_train_interpolate_plane",
    //     patterns_rates[0], patterns_rates[1], nb_sample_points_vector_field);
    compute_and_save_potential_vector_field_two_pattern(delta,
        net, sim_data_foldername, "post_train", patterns_potentials[0],
        patterns_potentials[1], nb_sample_points_vector_field,
        up_lim_vector_field);

    // //---------------------------------------------------------- Network
    // evolution
    std::cout << "Letting the network evolve from neutral state"<<endl;
    // // net.set_state(vector<double>(network_size, 0.5));
    // std::vector<std::vector<double>> interpolate =
    //     linspace_vector(vector<double>(network_size, 0.0), patterns_rates[0], 10);
    // vector<double> state = interpolate[1];
    // for (size_t i = 0; i < state.size(); i++)
    // {
    //     std::cout<<state[i]<<" ";
    // }
    net.set_state(vector<double>(network_size, 0.5));
    std::cout<< std::endl;
    string result_file_traj_1_name =
        sim_data_foldername + "/results_evolution_1.data";
    std::ofstream result_file_1_traj(result_file_traj_1_name, std::ios::trunc);

    SimulationConfig config;

    config.output = &result_file_1_traj;
    config.delta = delta;
    config.epsilon = delta / 10000;
    config.depressed = true;
    config.save = true;
    config.max_iter = 100 / delta;
    config.noise = false;
    // config.stddev = stddev;
    int nb_iter_sim=0;

    nb_iter_sim += run_net_sim_choice(net, config);
    std::cout << "Number of iteration convergence :" << std::endl;
    std::cout << nb_iter_sim << std::endl;
    std::string weights_file_name = sim_data_foldername + "/weights.data";

    // writeMatrixToFile(net.weight_matrix, weights_file_name);

    net.pot_inhib_symmetric(beta);
    // net.pot_inhib(1);
    // compute_and_save_rate_vector_field_two_pattern_bilinear(delta,
    //     net, sim_data_foldername, "post_inhib_interpolate_plane",
    //     patterns_rates[0], patterns_rates[1], nb_sample_points_vector_field);
    compute_and_save_potential_vector_field_two_pattern(
        delta, net, sim_data_foldername, "post_inhib", patterns_potentials[0],
        patterns_potentials[1], nb_sample_points_vector_field,
        up_lim_vector_field);

    net.set_state(vector<double>(network_size, 0.5));
    nb_iter_sim =0;
    string result_file_traj_2_name =
        sim_data_foldername + "/results_evolution_2.data";
    std::ofstream result_file_2_traj(result_file_traj_2_name, std::ios::trunc);
    config.output = &result_file_2_traj;
    nb_iter_sim += run_net_sim_choice(net, config);
    std::cout << "Number of iteration convergence 2 :" << std::endl;
    std::cout << nb_iter_sim << std::endl;

    net.pot_inhib_symmetric(beta*2);
    // net.pot_inhib(1);
    // compute_and_save_rate_vector_field_two_pattern_bilinear(delta,
    //     net, sim_data_foldername, "post_inhib_2_interpolate_plane",
    //     patterns_rates[0], patterns_rates[1], nb_sample_points_vector_field);
    compute_and_save_potential_vector_field_two_pattern(
        delta, net, sim_data_foldername, "post_inhib_2", patterns_potentials[0],
        patterns_potentials[1], nb_sample_points_vector_field,
        up_lim_vector_field);

    net.reset_inhib();

    std::cout << "WRITING THE ATTRACTOR" << std::endl;
    // Initialize velocity matrix for momentum
    velocity_matrix = std::vector<std::vector<double>>(
        network_size, std::vector<double>(network_size, 0.0));
    drives_error.resize(network_size, 0.0);
    // Training loop
    cpt = 0;
    while (max_error > epsilon_learning && cpt <= 10 / learning_rate) {
        for (int j = 0; j < patterns.size(); j++) {
            net.derivative_gradient_descent_with_momentum_null_sum(
                patterns[j], patterns_rates[j], drive_target, learning_rate,
                leak, drives_error, velocity_matrix, momentum_coef,0.1);

        }
        max_error = std::abs(
            *std::max_element(drives_error.begin(), drives_error.end()));
        cpt += 1;
    }
    std::cout << "nombre d'iterations second learning: " << std::endl;
    std::cout << cpt << std::endl;

    // compute_and_save_rate_vector_field_two_pattern_bilinear(delta,
    //     net, sim_data_foldername, "post_null_w_wum_interpolate_plane",
    //     patterns_rates[0], patterns_rates[1], nb_sample_points_vector_field);
    compute_and_save_potential_vector_field_two_pattern(
        delta, net, sim_data_foldername, "post_null_w", patterns_potentials[0],
        patterns_potentials[1], nb_sample_points_vector_field,
        up_lim_vector_field);
}

int main(int argc, char **argv) {
    // string sim_name = "write_net_sizes_relative_num_patterns";
    string sim_name = "Fig_patterns_vector_field";
    string foldername_results =
        "../../../data/all_data_splited/trained_networks_fast/" + sim_name;

    // Create directory if it doesn't exist
    if (!fs::exists(foldername_results)) {
        if (!fs::create_directory(foldername_results)) {
            std::cerr << "Error creating directory: " << foldername_results
                      << std::endl;
            return 1;
        }
    }
    double learning_rate ={0.0001};
    unordered_map<string, vector<double>> varying_params = {
        {"beta", {0.35}},
        {"nb_sample_points_vector_field", {6}},
        {"drive_target", {6}},
        {"learning_rate", {learning_rate}},
        {"leak", {1.3}},
        {"delta", {0.02}},
        {"epsilon_learning", {learning_rate / 1000000}},
        {"noise_level", {1}},
        {"up_lim_vector_field", {1.5}}};

    const int max_threads = 20;  // Set the maximum number of concurrent threads
    int active_threads = 0;
    std::mutex mtx;
    std::condition_variable cv;
    std::vector<std::thread> threads;
    vector<unordered_map<string, double>> combinations =
        generateCombinations(varying_params);
    for (int sim_number = 0; sim_number < combinations.size(); ++sim_number) {
        {
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait(lock, [&] { return active_threads < max_threads; });
            ++active_threads;
        }

        threads.emplace_back([=, &mtx, &cv, &active_threads] {
            run_simulation(sim_number, combinations[sim_number],
                           foldername_results);
            {
                std::lock_guard<std::mutex> lock(mtx);
                --active_threads;
            }
            cv.notify_all();
        });
    }

    for (auto &t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }

    collectSimulationData(foldername_results);

    return 0;
}
