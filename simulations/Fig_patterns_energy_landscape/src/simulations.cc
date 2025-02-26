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
    std::cout << "Vector field upper limit: " << up_lim_vector_field
              << std::endl;
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
    }
    file.close();

    vector<vector<double>> patterns_rates = patterns_as_states(
        net.transfer(drive_target), net.transfer(-drive_target), patterns);
    vector<vector<double>> patterns_potentials =
        patterns_as_states(drive_target, -drive_target, patterns);

    // Compute and save vector field and energy field for pre-training state
    std::cout << "Computing pre-training vector field and energy landscape..."
              << std::endl;
    compute_and_save_potential_vector_field_two_pattern(
        delta, net, sim_data_foldername, "pre_train", patterns_potentials[0],
        patterns_potentials[1], nb_sample_points_vector_field,
        up_lim_vector_field);

    compute_and_save_energy_field_two_pattern(
        delta, net, sim_data_foldername, "pre_train", patterns_potentials[0],
        patterns_potentials[1], nb_sample_points_vector_field,
        up_lim_vector_field);

    //---------------------------------------------------------- Training
    std::cout << "TRAINING THE NETWORK" << std::endl;
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
        }
        max_error = std::abs(
            *std::max_element(drives_error.begin(), drives_error.end()));
        cpt += 1;
    }
    std::cout << "Number of training iterations: " << cpt << std::endl;

    // Compute and save vector field and energy field for post-training state
    std::cout << "Computing post-training vector field and energy landscape..."
              << std::endl;
    compute_and_save_potential_vector_field_two_pattern(
        delta, net, sim_data_foldername, "post_train", patterns_potentials[0],
        patterns_potentials[1], nb_sample_points_vector_field,
        up_lim_vector_field);

    compute_and_save_energy_field_two_pattern(
        delta, net, sim_data_foldername, "post_train", patterns_potentials[0],
        patterns_potentials[1], nb_sample_points_vector_field,
        up_lim_vector_field);

    //---------------------------------------------------------- Network
    //evolution
    std::cout << "Letting the network evolve from neutral state" << endl;
    net.set_state(vector<double>(network_size, 0.5));
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
    int nb_iter_sim = 0;

    nb_iter_sim += run_net_sim_choice(net, config);
    std::cout << "Number of iterations to convergence: " << nb_iter_sim
              << std::endl;

    // Apply inhibitory potentiation
    std::cout << "Applying inhibitory potentiation (beta = " << beta << ")"
              << std::endl;
    net.pot_inhib_symmetric(beta);

    // Compute and save vector field and energy field after inhibitory
    // potentiation
    std::cout
        << "Computing post-inhibition vector field and energy landscape..."
        << std::endl;
    compute_and_save_potential_vector_field_two_pattern(
        delta, net, sim_data_foldername, "post_inhib", patterns_potentials[0],
        patterns_potentials[1], nb_sample_points_vector_field,
        up_lim_vector_field);

    compute_and_save_energy_field_two_pattern(
        delta, net, sim_data_foldername, "post_inhib", patterns_potentials[0],
        patterns_potentials[1], nb_sample_points_vector_field,
        up_lim_vector_field);

    // Run evolution again after inhibitory potentiation
    net.set_state(vector<double>(network_size, 0.5));
    nb_iter_sim = 0;
    string result_file_traj_2_name =
        sim_data_foldername + "/results_evolution_2.data";
    std::ofstream result_file_2_traj(result_file_traj_2_name, std::ios::trunc);
    config.output = &result_file_2_traj;
    nb_iter_sim += run_net_sim_choice(net, config);
    std::cout << "Number of iterations to convergence after inhibition: "
              << nb_iter_sim << std::endl;

    // Apply more inhibitory potentiation
    std::cout << "Applying additional inhibitory potentiation (beta = "
              << beta * 2 << ")" << std::endl;
    net.pot_inhib_symmetric(beta * 2);

    // Compute and save vector field and energy field after second inhibitory
    // potentiation
    std::cout
        << "Computing post-inhibition-2 vector field and energy landscape..."
        << std::endl;
    compute_and_save_potential_vector_field_two_pattern(
        delta, net, sim_data_foldername, "post_inhib_2", patterns_potentials[0],
        patterns_potentials[1], nb_sample_points_vector_field,
        up_lim_vector_field);

    compute_and_save_energy_field_two_pattern(
        delta, net, sim_data_foldername, "post_inhib_2", patterns_potentials[0],
        patterns_potentials[1], nb_sample_points_vector_field,
        up_lim_vector_field);

    // Reset inhibition and try null-sum weights
    net.reset_inhib();

    std::cout << "TRAINING WITH NULL-SUM WEIGHTS" << std::endl;
    // Initialize velocity matrix for momentum
    velocity_matrix = std::vector<std::vector<double>>(
        network_size, std::vector<double>(network_size, 0.0));
    drives_error.resize(network_size, 0.0);
    // Training loop
    max_error = 1000;
    cpt = 0;
    while (max_error > epsilon_learning && cpt <= 10 / learning_rate) {
        for (int j = 0; j < patterns.size(); j++) {
            net.derivative_gradient_descent_with_momentum_null_sum(
                patterns[j], patterns_rates[j], drive_target, learning_rate,
                leak, drives_error, velocity_matrix, momentum_coef, 0.1);
        }
        max_error = std::abs(
            *std::max_element(drives_error.begin(), drives_error.end()));
        cpt += 1;
    }
    std::cout << "Number of null-sum training iterations: " << cpt << std::endl;

    // Compute and save vector field and energy field after null-sum weight
    // training
    std::cout << "Computing post-null-sum vector field and energy landscape..."
              << std::endl;
    compute_and_save_potential_vector_field_two_pattern(
        delta, net, sim_data_foldername, "post_null_w", patterns_potentials[0],
        patterns_potentials[1], nb_sample_points_vector_field,
        up_lim_vector_field);

    compute_and_save_energy_field_two_pattern(
        delta, net, sim_data_foldername, "post_null_w", patterns_potentials[0],
        patterns_potentials[1], nb_sample_points_vector_field,
        up_lim_vector_field);

    std::cout << "Simulation " << sim_number << " completed successfully."
              << std::endl;
}

int main(int argc, char **argv) {
    // string sim_name = "write_net_sizes_relative_num_patterns";
    string sim_name = "Fig_patterns_vector_field_energy";
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

    double learning_rate = 0.0001;
    unordered_map<string, vector<double>> varying_params = {
        {"beta", {0.35}},
        {"nb_sample_points_vector_field",
         {24}},  // Increased for better resolution
        {"drive_target", {6}},
        {"learning_rate", {learning_rate}},
        {"leak", {1.3}},
        {"delta", {0.02}},
        {"epsilon_learning", {learning_rate / 1000000}},
        {"noise_level", {1,0.35}},
        {"up_lim_vector_field", {1.5}}};

    const int max_threads = 20;  // Set the maximum number of concurrent threads
    int active_threads = 0;
    std::mutex mtx;
    std::condition_variable cv;
    std::vector<std::thread> threads;
    vector<unordered_map<string, double>> combinations =
        generateCombinations(varying_params);

    std::cout << "Starting simulation with " << combinations.size()
              << " parameter combinations" << std::endl;

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

    std::cout << "All simulations completed. Collecting data..." << std::endl;
    collectSimulationData(foldername_results);
    std::cout << "Data collection complete." << std::endl;

    return 0;
}