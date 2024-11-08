#include "network.hpp"
#include "utils.hpp"
#include <numeric>
#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <filesystem>
#include <set>
#include <algorithm>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <condition_variable>

using namespace std;

namespace fs = std::filesystem;

void run_simulation(int sim_number, unordered_map<string, double> parameters, const string foldername_results)
{
    std::cout <<"sim bumber : "<< sim_number << std::endl;
    // Learning constants
    double epsilon_learning=parameters.at("epsilon_learning");
    double drive_target = parameters.at("drive_target");
    double learning_rate = parameters.at("learning_rate");
    // Network constants
    int network_size = static_cast<int>(parameters.at("network_size"));
    // int nb_winners =static_cast<int>(parameters.at("nb_winners"));
    int nb_winners = max(2,static_cast<int>(parameters.at("relative_nb_winner")*network_size)); // number of 1's neurons
    parameters["nb_winners"] = static_cast<double>(nb_winners);
    double leak = parameters.at("leak");
    //Simulation constant
    double noise_level = parameters.at("noise_level");
    double delta = parameters.at("delta");
    double init_drive = parameters.at("init_drive");
    double ratio_flip_writing = parameters.at("ratio_flip_writing");
    int num_patterns = parameters.at("num_patterns");

    int col_with = sqrt(network_size);

    string sim_data_foldername;
    string patterns_file_name;
    string result_file_name;
    string weights_file_name;
    string connectivity_file_name;
    vector<vector<bool>> initial_patterns;
    vector<vector<double>> initial_patterns_rates;
    vector<bool> winning_units;

    sim_data_foldername = foldername_results + "/sim_nb_" + to_string(sim_number);

    // Create directory if it doesn't exist
    if (!fs::exists(sim_data_foldername))
    {
        if (!fs::create_directory(sim_data_foldername))
        {
            std::cerr << "Error creating directory: " << sim_data_foldername << std::endl;
            return;
        }
    }

    patterns_file_name = sim_data_foldername + "/patterns.data";
    std::ofstream file(patterns_file_name, std::ios::trunc);
    initial_patterns = generatePatterns(num_patterns, network_size, nb_winners, noise_level);
    for (int i = 0; i < num_patterns; i++)
    {
        writeBoolToCSV(file, initial_patterns[i]);
        // show_vector_bool_grid(initial_patterns[i], col_with);
    }
    file.close();
    createParameterFile(sim_data_foldername, parameters);
    // Build Fully connected network
    vector<vector<bool>> connectivity_matrix(network_size, vector<bool>(network_size, false));
    for (int i = 0; i < network_size; i++)
    {
        for (int j = 0; j < network_size; j++)
        {
            if (i != j)
            {
                connectivity_matrix[i][j] = true;
            }
        }
    }

    Network net = Network(connectivity_matrix, network_size, leak);
    // Loading training data
    initial_patterns = loadPatterns(patterns_file_name);
    initial_patterns_rates = patterns_as_states(net.transfer(drive_target), net.transfer(-drive_target), initial_patterns);
    vector<double> drives_error;
    // Initialize velocity matrix for momentum
    std::vector<std::vector<double>> velocity_matrix(network_size, 
                                                     std::vector<double>(network_size, 0.0));
    double momentum_coef = 0.9; // You can adjust this value
    drives_error.resize(network_size,0.0);
    // Training loop
    double max_error=1000;
    int cpt=0;
    std::cout << "WRITING ATTRACTORS" << std::endl;
    while (max_error > epsilon_learning && cpt <= 10/learning_rate)
    {
        for (int j = 0; j < initial_patterns.size(); j++)
        {
            // net.derivative_gradient_descent(initial_patterns[j],initial_patterns_rates[j],drive_target,learning_rate, leak, drives_error);
          net.derivative_gradient_descent_with_momentum(
                                                        initial_patterns[j],
                                                        initial_patterns_rates[j],
                                                        drive_target,
                                                        learning_rate,
                                                        leak,
                                                        drives_error,
                                                        velocity_matrix,
                                                        momentum_coef
                                                        );
        }
        max_error = std::abs(*std::max_element(drives_error.begin(),drives_error.end()));
        cpt+=1;
    }
    std::cout << "nombre d'iterations" << std::endl;
    std::cout << cpt << std::endl;
    // Querying
    std::cout << "Querying initial memories" << std::endl;
    vector<double> query_pattern;
    int succes= 0 ;
    // double strength_drive = 0.1;
    for (int i = 0; i < num_patterns; i++)
    {
        //TODO - change the pattern_as_states and link the target drive not magic number
        query_pattern=pattern_as_states(net.transfer(drive_target),net.transfer(-drive_target),initial_patterns[i]);
        query_pattern=setToValueRandomElements(query_pattern, int(network_size*ratio_flip_writing), init_drive);
        // noisy_pattern = std::vector<double>(network_size,0.5);
        net.set_state(query_pattern);
        run_net_sim(net,1/delta, delta);
        winning_units = assignBoolToTopNValues(net.activity_list, nb_winners);
        if (comparestates(winning_units,initial_patterns[i])){
            succes+=1;
        }
    }
    // The number of unique vectors found
    std::cout << "Number of vectors found: " << succes << " nb_patterns : " << num_patterns << "nb_winners : " << nb_winners << " nb_flip : " <<int(network_size*ratio_flip_writing)<<" Network size: "<<network_size<<std::endl;
    result_file_name = sim_data_foldername + "/results.data";
    std::ofstream result_file(result_file_name, std::ios::trunc);
    result_file << "nb_found_patterns="<<succes;
    result_file.close();

    weights_file_name = sim_data_foldername + "/weights.data";
    writeMatrixToFile(net.weight_matrix, weights_file_name);

    connectivity_file_name = sim_data_foldername + "/connectivity.data";
    writeBoolMatrixToFile(net.connectivity_matrix, connectivity_file_name);
}

int main(int argc, char **argv)
{
    // string sim_name = "write_net_sizes_relative_num_patterns";
    string sim_name = "Fig_load_SR_average_new";
    string foldername_results = "../../../data/all_data_splited/trained_networks_fast/" + sim_name;

    // Create directory if it doesn't exist
    if (fs::exists(foldername_results))
    {
        fs::remove_all(foldername_results);
    }
    if (!fs::create_directory(foldername_results))
    {
        std::cerr << "Error creating directory: " << foldername_results << std::endl;
        return 1;
    }
    // Define varying parameters
    vector<double> num_patterns = generateEvenlySpacedIntegers(1, 20, 20);
    // vector<double> num_patterns = generateEvenlySpacedIntegers(1, 15, 10);
    // vector<double> num_patterns = {6};
    vector<double> drive_targets = {6};
    vector<double> network_sizes = generateEvenlySpacedIntegers(20, 200, 20);
    // vector<double> network_sizes = {200};
    vector<double> init_drive = {0.25};
    // vector<double> noise_level = linspace(0.2, 1, 15);
    vector<double> noise_level = {0.5};
    double learning_rate= 0.0001;
    // vector<double> noise_level = {0.5};
    vector<double> repetition = generateEvenlySpacedIntegers(0,10,10);
    unordered_map<string, vector<double>> varying_params = {
        {"repetitions", {repetition}},
        {"ratio_flip_writing", {0.1}},
        {"drive_target", drive_targets},
        {"max_pattern",{*(std::max_element(num_patterns.begin(), num_patterns.end()))}},
        {"num_patterns", num_patterns},
        {"learning_rate", {learning_rate}}, // REMOVED-target rates
        {"network_size", network_sizes},
        {"relative_nb_winner", {1.0/2.0}},
        {"noise_level", {noise_level}},
        {"epsilon_learning", {learning_rate/1000000}},
        {"delta",{0.1}},
        {"init_drive", {0.5}},
        {"leak", {1}}};

    vector<unordered_map<string, double>> combinations = generateCombinations(varying_params);

    const int max_threads = 20; // Set the maximum number of concurrent threads
    int active_threads = 0;
    std::mutex mtx;
    std::condition_variable cv;
    std::vector<std::thread> threads;
    for (int sim_number = 0; sim_number < combinations.size(); ++sim_number)
    {

        {
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait(lock, [&]
                    { return active_threads < max_threads; });
            ++active_threads;
        }

        threads.emplace_back([=, &mtx, &cv, &active_threads]
                                {
            run_simulation(sim_number, combinations[sim_number],foldername_results);
            {
                std::lock_guard<std::mutex> lock(mtx);
                --active_threads;
            }
            cv.notify_all(); });
    }

    for (auto &t : threads)
    {
        if (t.joinable())
        {
            t.join();
        }
    }

    collectSimulationData(foldername_results);

    return 0;
}
