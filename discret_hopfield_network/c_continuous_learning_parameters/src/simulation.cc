#include "network.hpp"
#include "utils.hpp"
#include <vector>
#include <iostream>
#include <random>
#include <ctime>
#include <thread>
#include <mutex>
#include <condition_variable>

namespace fs = std::filesystem;
using namespace std;

void run_simulation(int sim_number, unordered_map<string, double> parameters, const string foldername_results){
    
    int initial_network_size = static_cast<int>(parameters.at("network_size"));
    int num_patterns = static_cast<int>(initial_network_size*parameters.at("relative_num_patterns"));
    int nb_winners = initial_network_size / 2;
    double noise_level = 0.3;
    int nb_iter = 500;
    string sim_data_foldername = foldername_results + "/sim_nb_" + to_string(sim_number);

    // Create directory if it doesn't exist
    if (!fs::exists(sim_data_foldername))
    {
        if (!fs::create_directory(sim_data_foldername))
        {
            std::cerr << "Error creating directory: " << sim_data_foldername << std::endl;
            return;
        }
    }

    string result_file_name_retrieval = sim_data_foldername + "/results.data";
    std::ofstream result_file_retrieval(result_file_name_retrieval, std::ios::trunc);

    vector<vector<bool>> bool_patterns = generatePatterns(num_patterns, initial_network_size, nb_winners, noise_level);
    vector<vector<bool>> bool_combined_patterns = generateCombinedPatterns(bool_patterns);
    vector<vector<double>> patterns = patterns_as_states(1, -1, bool_patterns);
    vector<vector<double>> combined_patterns = patterns_as_states(1, -1, bool_combined_patterns);

    int network_size = initial_network_size * 2;
    num_patterns = combined_patterns.size();
    // cout<< "combined patterns"<<endl;
    // for (const auto &pattern : combined_patterns)
    // {
    //     cout << "Pattern: ";
    //     for (const auto &val : pattern)
    //         cout << val << " ";
    //     cout << "\n";
    // }
    // cout << endl;
    // cout << "base patterns" << endl;
    // for (const auto &pattern : patterns)
    // {
    //     cout << "Pattern: ";
    //     for (const auto &val : pattern)
    //         cout << val << " ";
    //     cout << "\n";
    // }
    // cout << endl;

    vector<vector<bool>> connectivity_matrix(network_size, vector<bool>(network_size, true));
    for (int i = 0; i < network_size; i++)
    {
        connectivity_matrix[i][i] = false;
    }

    Network net(connectivity_matrix, network_size);

    for(int i =0; i <nb_iter; i++){
        net.trainPerceptronInput(combined_patterns,initial_network_size);
    }

    // vector<double> retrieved_state;

    vector<double> retrieved_state = patterns[0];
    int nb_found_patterns= 1;
    for(int i=0; i<num_patterns-1;i++){
        vector<double> initial_state(network_size,1.0);
        for(int i = 0; i< initial_network_size; i++){
            initial_state[i] = retrieved_state[i];
        }
        vector<double> ret = net.runAndClamp(initial_state,initial_network_size);
        retrieved_state = pickLastNElements(ret, initial_network_size);
        bool success = compareStates(retrieved_state, patterns[i+1]);
        nb_found_patterns+=success;
        // to_write_states.emplace_back(retrieved_state);
    }
    result_file_retrieval << "nb_found_patterns=" << nb_found_patterns << "\n";
    result_file_retrieval.close();

    createParameterFile(sim_data_foldername, parameters);

}

int main(int argc, char **argv)
{
    string sim_name = "network_size_num_pattern";
    string foldername_results = "../../../data/all_data_discret/" + sim_name;

    // Create directory if it doesn't exist
    if (!fs::exists(foldername_results))
    {
        if (!fs::create_directory(foldername_results))
        {
            std::cerr << "Error creating directory: " << foldername_results << std::endl;
            return 1;
        }
    }
    vector<double> all_relative_num_patterns = linspace(0.4, 1, 20);
    vector<double> network_sizes = {50, 55, 60, 65, 70, 75, 80, 85, 90,95,100,105,110,115,120};
    unordered_map<string, vector<double>> varying_params = {
        {"relative_num_patterns", all_relative_num_patterns},
        {"network_size", network_sizes}};

    int active_threads = 0;
    std::vector<std::thread> threads;
    vector<unordered_map<string, double>> combinations = generateCombinations(varying_params);

    for (int sim_number = 0; sim_number < combinations.size(); ++sim_number)
    {

        // run_simulation(sim_number, combinations[sim_number], foldername_results);
        threads.emplace_back(run_simulation, sim_number, combinations[sim_number], foldername_results);

    }

    for (auto &t : threads)
    {
        if (t.joinable())
        {
            t.join();
        }
    }

    return 0;
}
