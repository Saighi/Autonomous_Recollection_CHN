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

// Define MPI tags
#define TASK_TAG 1
#define RESULT_TAG 2
#define TERMINATE_TAG 3

// Helper function to create directory
inline bool createDirectory(const std::string &path) {
    return mkdir(path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == 0 || errno == EEXIST;
}

// Existing run_simulation function
void run_simulation(int sim_number, unordered_map<string, double> parameters, const string foldername_results)
{
    std::cout << sim_number << std::endl;
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

// Master process function
void masterProcess(const vector<unordered_map<string, double>> &combinations, const string &foldername_results, int size) {
    int numTasks = combinations.size();
    int taskIndex = 0;
    int numWorkers = size - 1;
    int tasksCompleted = 0;
    MPI_Status status;

    // Initial task distribution
    for (int i = 1; i <= numWorkers && taskIndex < numTasks; ++i) {
        // Send task to worker i
        MPI_Send(&taskIndex, 1, MPI_INT, i, TASK_TAG, MPI_COMM_WORLD);
        taskIndex++;
    }

    // Dynamic task assignment
    while (tasksCompleted < numTasks) {
        int workerRank;
        int completedTaskIndex;

        // Receive completed task index from any worker
        MPI_Recv(&completedTaskIndex, 1, MPI_INT, MPI_ANY_SOURCE, RESULT_TAG, MPI_COMM_WORLD, &status);
        workerRank = status.MPI_SOURCE;
        tasksCompleted++;

        // Assign new task if available
        if (taskIndex < numTasks) {
            MPI_Send(&taskIndex, 1, MPI_INT, workerRank, TASK_TAG, MPI_COMM_WORLD);
            taskIndex++;
        } else {
            // No more tasks; send termination signal
            MPI_Send(0, 0, MPI_INT, workerRank, TERMINATE_TAG, MPI_COMM_WORLD);
        }
    }

    // Send termination signal to any workers that might still be waiting
    for (int i = 1; i <= numWorkers; ++i) {
        MPI_Send(0, 0, MPI_INT, i, TERMINATE_TAG, MPI_COMM_WORLD);
    }
}

// Worker process function
void workerProcess(const string &foldername_results, const vector<unordered_map<string, double>> &combinations) {
    MPI_Status status;

    while (true) {
        int taskIndex;

        // Receive task from the master
        MPI_Recv(&taskIndex, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        if (status.MPI_TAG == TERMINATE_TAG) {
            // No more tasks; exit loop
            break;
        } else if (status.MPI_TAG == TASK_TAG) {
            // Run the simulation with the given task index
            run_simulation(taskIndex, combinations[taskIndex], foldername_results);

            // Notify the master that the task is completed
            MPI_Send(&taskIndex, 1, MPI_INT, 0, RESULT_TAG, MPI_COMM_WORLD);
        }
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Parse command-line arguments
    string sim_name = argv[1];
    string foldername_results = argv[2];
    foldername_results = foldername_results + sim_name;

    // Only rank 0 creates the main directory
    if (rank == 0) {
        struct stat st;
        if (stat(foldername_results.c_str(), &st) == 0) {
            system(("rm -rf " + foldername_results).c_str());
        }
        createDirectory(foldername_results.c_str());
    }

    // Make sure all processes wait until directory is created
    MPI_Barrier(MPI_COMM_WORLD);

    // Define varying parameters
    vector<double> num_patterns = generateEvenlySpacedIntegers(5, 25, 10);
    vector<double> drive_targets = {6};
    vector<double> network_sizes = generateEvenlySpacedIntegers(50, 300, 10);
    vector<double> ratio_flip_writing = {0.5};
    vector<double> init_drive = {0.25};

    unordered_map<string, vector<double>> varying_params = {
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
        {"leak", {1.3}}
    };

    // Generate parameter combinations (tasks)
    vector<unordered_map<string, double>> combinations = generateCombinations(varying_params);

    if (rank == 0) {
        // Master process
        masterProcess(combinations, foldername_results, size);
    } else {
        // Worker processes
        workerProcess(foldername_results, combinations);
    }

    MPI_Finalize();
    return 0;
}
