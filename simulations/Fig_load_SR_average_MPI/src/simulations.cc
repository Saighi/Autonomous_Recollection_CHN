#include "network.hpp"
#include "utils.hpp"
#include <mpi.h>
#include <numeric>
#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <set>
#include <algorithm>
#include <unordered_map>
#include <sys/stat.h>
#include <sys/types.h>

using namespace std;

// Helper function to create directory
inline bool createDirectory(const std::string &path)
{
    return mkdir(path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == 0 || errno == EEXIST;
}

void run_simulation(int sim_number, unordered_map<string, double> parameters, const string foldername_results)
{
    // Learning constants
    double epsilon_learning = parameters.at("epsilon_learning");
    double drive_target = parameters.at("drive_target");
    double learning_rate = parameters.at("learning_rate");
    // Network constants
    int network_size = static_cast<int>(parameters.at("network_size"));
    int nb_winners = max(2, static_cast<int>(parameters.at("relative_nb_winner") * network_size));
    parameters["nb_winners"] = static_cast<double>(nb_winners);
    double leak = parameters.at("leak");
    // Simulation constant
    double noise_level = parameters.at("noise_level");
    double delta = parameters.at("delta");
    double init_drive = parameters.at("init_drive");
    double ratio_flip_writing = parameters.at("ratio_flip_writing");
    int num_patterns = parameters.at("num_patterns");

    int col_with = sqrt(network_size);

    string sim_data_foldername = foldername_results + "/sim_nb_" + to_string(sim_number);
    createDirectory(sim_data_foldername.c_str());

    string patterns_file_name = sim_data_foldername + "/patterns.data";
    std::ofstream file(patterns_file_name, std::ios::trunc);
    vector<vector<bool>> initial_patterns = generatePatterns(num_patterns, network_size, nb_winners, noise_level);
    for (int i = 0; i < num_patterns; i++)
    {
        writeBoolToCSV(file, initial_patterns[i]);
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
    vector<vector<double>> initial_patterns_rates = patterns_as_states(net.transfer(drive_target), net.transfer(-drive_target), initial_patterns);
    vector<double> drives_error;
    drives_error.resize(network_size, 0.0);

    // Training loop
    double max_error = 1000;
    int cpt = 0;
    while (max_error > epsilon_learning && cpt <= 10000)
    {
        for (int j = 0; j < initial_patterns.size(); j++)
        {
            net.derivative_gradient_descent(initial_patterns[j], initial_patterns_rates[j], drive_target, learning_rate, leak, drives_error);
        }
        max_error = std::abs(*std::max_element(drives_error.begin(), drives_error.end()));
        cpt += 1;
    }

    // Querying
    vector<double> query_pattern;
    int succes = 0;
    vector<bool> winning_units;

    for (int i = 0; i < num_patterns; i++)
    {
        query_pattern = pattern_as_states(net.transfer(drive_target), net.transfer(-drive_target), initial_patterns[i]);
        query_pattern = setToValueRandomElements(query_pattern, int(network_size * ratio_flip_writing), init_drive);
        net.set_state(query_pattern);
        run_net_sim(net, 1 / delta, delta);
        winning_units = assignBoolToTopNValues(net.activity_list, nb_winners);
        if (comparestates(winning_units, initial_patterns[i]))
        {
            succes += 1;
        }
    }

    string result_file_name = sim_data_foldername + "/results.data";
    std::ofstream result_file(result_file_name, std::ios::trunc);
    result_file << "nb_found_patterns=" << succes;
    result_file.close();

    string weights_file_name = sim_data_foldername + "/weights.data";
    writeMatrixToFile(net.weight_matrix, weights_file_name);

    string connectivity_file_name = sim_data_foldername + "/connectivity.data";
    writeBoolMatrixToFile(net.connectivity_matrix, connectivity_file_name);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    string sim_name = "Fig_load_SR_average_test_MPI";
    string foldername_results = "../../../data/all_data_splited/trained_networks_fast/" + sim_name;

    // Only rank 0 creates the main directory
    if (rank == 0)
    {
        struct stat st;
        if (stat(foldername_results.c_str(), &st) == 0)
        {
            system(("rm -rf " + foldername_results).c_str());
        }
        createDirectory(foldername_results.c_str());
    }

    // Make sure all processes wait until directory is created
    MPI_Barrier(MPI_COMM_WORLD);

    vector<double> num_patterns = generateEvenlySpacedIntegers(5, 25, 10);
    // vector<double> num_patterns = {1};
    vector<double> drive_targets = {6};
    // vector<double> network_sizes = generateEvenlySpacedIntegers(50, 300, 10);
    vector<double> network_sizes = {200};
    vector<double> ratio_flip_writing = {0.5};
    vector<double> init_drive = {0.25};
    // vector<double> repetitions = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    unordered_map<string, vector<double>> varying_params = {
        // {"repetitions", repetitions},
        {"ratio_flip_writing", ratio_flip_writing},
        {"drive_target", drive_targets},
        {"max_pattern", {*(std::max_element(num_patterns.begin(), num_patterns.end()))}},
        {"num_patterns", num_patterns},
        {"learning_rate", {0.001}},
        {"network_size", network_sizes},
        {"relative_nb_winner", {1.0 / 3.0}},
        {"noise_level", {1}},
        {"epsilon_learning", {0.01}},
        {"delta", {0.01}},
        {"init_drive", {0.25}},
        {"leak", {1.3}}};

    vector<unordered_map<string, double>> combinations = generateCombinations(varying_params);

    // Calculate which simulations each process should handle
    if(size>combinations.size()){
        if(rank<combinations.size()){
            std::cout << "simulation: "<< rank << " is running"<< std::endl;
            run_simulation(rank, combinations[rank], foldername_results);
            std::cout << "simulation: "<< rank << " is is over"<< std::endl;
        }
        else {
            std::cout << "unused rank" << std::endl;
        }
    }
    else{
        int num_combinations = combinations.size();

        std::vector<std::vector<int>> ranks_affiliated_sim = ranks_processes(size, num_combinations);
        
        //Each rank processes its batch of simulations
        for (int sim_number: ranks_affiliated_sim[rank])
        {
            cout << "Process " << rank << " running simulation " << sim_number
                    << " of " << num_combinations << endl;
            run_simulation(sim_number, combinations[sim_number], foldername_results);
        }
        if(rank==0){
            for(vector<int> var : ranks_affiliated_sim)
            {
                for(int in_var : var)
                {
                    std::cout << in_var <<" ";   
                }
                std::cout << std::endl;
            }
        }
    }


    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();
    return 0;
}