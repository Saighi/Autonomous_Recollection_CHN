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
    int network_size = 400;
    float dst_mul = 1;
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
    

    float drive_target_0 = drive_target;
    float drive_target_1 = drive_target*dst_mul;

    patterns_rates[0] = pattern_as_states(
        net.transfer(drive_target_0), net.transfer(-drive_target_0), patterns[0]);
    patterns_rates[1] = pattern_as_states(
        net.transfer(drive_target_1), net.transfer(-drive_target_1), patterns[1]);
    patterns_potentials[0] = pattern_as_states(drive_target_0, -drive_target_0, patterns[0]);
    patterns_potentials[1] = pattern_as_states(drive_target_1, -drive_target_1, patterns[1]);

    // Compute and save vector field and energy field for pre-training state
    std::cout << "Computing pre-training vector field and energy landscape..."
              << std::endl;
    compute_and_save_potential_vector_field_two_pattern(
        delta, net, sim_data_foldername, "pre_train", patterns_potentials[0],
        patterns_potentials[1], nb_sample_points_vector_field,
        up_lim_vector_field);
    // Also save decomposed synaptic pushes (excit-only and inhib-only)
    compute_and_save_potential_vector_field_two_pattern_excit_only(
        delta, net, sim_data_foldername, "pre_train", patterns_potentials[0],
        patterns_potentials[1], nb_sample_points_vector_field,
        up_lim_vector_field);
    compute_and_save_energy_field_two_pattern_inhib_only(
        net, sim_data_foldername, "pre_train", patterns_potentials[0],
        patterns_potentials[1], nb_sample_points_vector_field,
        up_lim_vector_field, false);

    // Bias-only (no W, no inhibition, no leak)
    // compute_and_save_potential_vector_field_two_pattern_bias_only(
    //     delta, net, sim_data_foldername, "pre_train", patterns_potentials[0],
    //     patterns_potentials[1], nb_sample_points_vector_field,
    //     up_lim_vector_field);

    // Bias-only energy landscape (from learned biases)
    compute_and_save_energy_field_two_pattern_bias_only(
        net, sim_data_foldername, "pre_train", patterns_potentials[0],
        patterns_potentials[1], nb_sample_points_vector_field,
        up_lim_vector_field, false);

    // Weights-only (no leak, no inhibition, no bias)
    compute_and_save_potential_vector_field_two_pattern_weights_only(
        delta, net, sim_data_foldername, "pre_train", patterns_potentials[0],
        patterns_potentials[1], nb_sample_points_vector_field,
        up_lim_vector_field);

    compute_and_save_energy_field_two_pattern(
        delta, net, sim_data_foldername, "pre_train", patterns_potentials[0],
        patterns_potentials[1], nb_sample_points_vector_field,
        up_lim_vector_field, false);
    // Weights-only energy landscape pre-train
    // compute_and_save_energy_field_two_pattern_weights_only(
    //     net, sim_data_foldername, "pre_train", patterns_potentials[0],
    //     patterns_potentials[1], nb_sample_points_vector_field,
    //     up_lim_vector_field, false);

    compute_and_save_potential_vector_field_two_pattern_inhib_only(
        delta, net, sim_data_foldername, "pre_train", patterns_potentials[0],
        patterns_potentials[1], nb_sample_points_vector_field,
        up_lim_vector_field);

    //---------------------------------------------------------- Training
    std::cout << "TRAINING THE NETWORK (with bias)" << std::endl;
    // Initialize velocity matrix for momentum
    std::vector<std::vector<double>> velocity_matrix(
        network_size, std::vector<double>(network_size, 0.0));
    double momentum_coef = 0.9;  // You can adjust this value
    vector<double> drives_error;
    vector<double> neutral_state_rates(network_size,0.5); 
    drives_error.resize(network_size, 0.0);
    // Training loop
    double max_error = 1000;
    int cpt = 0;
    while (max_error > epsilon_learning && cpt <= 10 / learning_rate) {
        // net.derivative_gradient_descent(patterns_potentials[0],
        //                                     learning_rate, leak,
        //                                     drives_error);
        // net.derivative_gradient_descent(patterns_potentials[1],learning_rate, leak,
        //                                       drives_error);
        net.derivative_gradient_descent_with_bias(patterns_potentials[0],
                                                  learning_rate, leak,
                                                  drives_error);
        net.derivative_gradient_descent_with_bias(patterns_potentials[1],
                                                  learning_rate, leak,
                                                  drives_error);
        // net.derivative_gradient_descent_arbitrary(neutral_state_rates,
        //                                   learning_rate, leak,
        //                                   drives_error);
        max_error = std::abs(
            *std::max_element(drives_error.begin(), drives_error.end()));
        cpt += 1;
    }

    std::cout << "Number of training iterations: " << cpt << std::endl;

    // Save learned bias vector after training
    {
        std::string bias_file = sim_data_foldername + "/bias_post_train.data";
        std::ofstream bout(bias_file, std::ios::trunc);
        if (bout.is_open()) {
            writeToCSV(&bout, net.bias);
            bout.close();
        } else {
            std::cerr << "Unable to open file to save bias: " << bias_file
                      << std::endl;
        }
    }

    // Compute and save vector field and energy field for post-training state
    std::cout << "Computing post-training vector field and energy landscape..."
              << std::endl;
    compute_and_save_potential_vector_field_two_pattern(
        delta, net, sim_data_foldername, "post_train", patterns_potentials[0],
        patterns_potentials[1], nb_sample_points_vector_field,
        up_lim_vector_field);

    compute_and_save_potential_vector_field_two_pattern_excit_only(
        delta, net, sim_data_foldername, "post_train", patterns_potentials[0],
        patterns_potentials[1], nb_sample_points_vector_field,
        up_lim_vector_field);
    compute_and_save_energy_field_two_pattern_inhib_only(
        net, sim_data_foldername, "post_train", patterns_potentials[0],
        patterns_potentials[1], nb_sample_points_vector_field,
        up_lim_vector_field, false);
    compute_and_save_energy_field_two_pattern(
        delta, net, sim_data_foldername, "post_train", patterns_potentials[0],
        patterns_potentials[1], nb_sample_points_vector_field,
        up_lim_vector_field, false);
    // Weights-only energy landscape post-train
    // compute_and_save_energy_field_two_pattern_weights_only(
    //     net, sim_data_foldername, "post_train", patterns_potentials[0],
    //     patterns_potentials[1], nb_sample_points_vector_field,
    //     up_lim_vector_field, false);

    compute_and_save_potential_vector_field_two_pattern_inhib_only(
        delta, net, sim_data_foldername, "post_train", patterns_potentials[0],
        patterns_potentials[1], nb_sample_points_vector_field,
        up_lim_vector_field);

    // Bias-only energy landscape (from learned biases)
    compute_and_save_energy_field_two_pattern_bias_only(
        net, sim_data_foldername, "post_train", patterns_potentials[0],
        patterns_potentials[1], nb_sample_points_vector_field,
        up_lim_vector_field, false);

    //---------------------------------------------------------- Trajectory + inhibitory potentiation loop
    int num_inhib_iterations = 4;  // repeat n times
    for (int iter = 1; iter <= num_inhib_iterations; ++iter) {
        std::cout << "Iteration " << iter
                  << ": resetting state to neutral and running trajectory"
                  << std::endl;

        // Reset network state to neutral (keep inhibition)
        net.set_state(vector<double>(network_size, 0.5));

        // Prepare trajectory output file
        string traj_file_name = sim_data_foldername + "/results_evolution_iter_" +
                                 to_string(iter) + ".data";
        std::ofstream traj_out(traj_file_name, std::ios::trunc);

        SimulationConfig config;
        config.output = &traj_out;
        config.delta = delta;
        config.epsilon = delta / 10000;
        config.depressed = true;
        config.save = true;
        config.max_iter = 100 / delta;
        config.noise = false;

        int nb_iter_sim = run_net_sim_choice(net, config);
        std::cout << "Converged in " << nb_iter_sim
                  << " iterations for loop iteration " << iter << std::endl;

        // Potentiate inhibition after trajectory
        std::cout << "Potentiating inhibition (beta = " << beta
                  << ") after iteration " << iter << std::endl;
        net.pot_inhib_symmetric(beta);

        // Save inhibitory-only vector field and energy landscape
        std::string tag = "iter_" + to_string(iter);
        compute_and_save_potential_vector_field_two_pattern_inhib_only(
            delta, net, sim_data_foldername, tag, patterns_potentials[0],
            patterns_potentials[1], nb_sample_points_vector_field,
            up_lim_vector_field);

        compute_and_save_energy_field_two_pattern_inhib_only(
            net, sim_data_foldername, tag, patterns_potentials[0],
            patterns_potentials[1], nb_sample_points_vector_field,
            up_lim_vector_field, false);

        compute_and_save_potential_vector_field_two_pattern(
            delta, net, sim_data_foldername, tag,
            patterns_potentials[0], patterns_potentials[1],
            nb_sample_points_vector_field, up_lim_vector_field);

        compute_and_save_energy_field_two_pattern(
            delta, net, sim_data_foldername, tag,
            patterns_potentials[0], patterns_potentials[1],
            nb_sample_points_vector_field, up_lim_vector_field, false);

        // Bias-only energy landscape (from learned biases)
        compute_and_save_energy_field_two_pattern_bias_only(
            net, sim_data_foldername, tag, patterns_potentials[0],
            patterns_potentials[1], nb_sample_points_vector_field,
            up_lim_vector_field, false);
    }

    // (Removed) null-sum training path

    std::cout << "Simulation " << sim_number << " completed successfully."
              << std::endl;
}

int main(int argc, char **argv) {
    // string sim_name = "Fig_vector_fields_patterns_different_distances";
    string sim_name = "Fig_vector_fields_patterns_same_distances";
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
        {"beta", {0.009}},
        {"nb_sample_points_vector_field",
         {24}},  // Increased for better resolution
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
