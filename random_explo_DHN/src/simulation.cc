#include "network.hpp"
#include "utils.hpp"
#include <vector>
#include <iostream>
#include <random>
#include <ctime>
#include <numeric>
#include <thread>
#include <string>

namespace fs = std::filesystem;
using namespace std;

// Function to randomize some elements of the initial state
vector<double> randomizeInitialState(const vector<double> &pattern, int num_random_elements)
{
    vector<double> randomized_state = pattern;
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, pattern.size() - 1);

    for (int i = 0; i < num_random_elements; ++i)
    {
        int index = dis(gen);
        randomized_state[index] = (randomized_state[index] == 1) ? -1 : 1;
    }

    return randomized_state;
}

// Function to compare two states
bool compareStates(const vector<double> &state1, const vector<double> &state2)
{
    if (state1.size() != state2.size())
        return false;

    bool direct_match = true;
    bool inverse_match = true;

    for (size_t i = 0; i < state1.size(); ++i)
    {
        if (state1[i] != state2[i])
            direct_match = false;
        if (state1[i] != -state2[i])
            inverse_match = false;
    }

    return direct_match || inverse_match;
}


void run_simulation(int sim_number, const unordered_map<string, double>& parameters, const string& foldername_results){
    std::cout << to_string(sim_number) << std::endl;

    int network_size = parameters.at("net_size");
    int nb_patterns = parameters.at("nb_pat");
    int nb_query = parameters.at("nb_query");
    int learning_rule = parameters.at("learning_rule");
    double noise_level = parameters.at("noise_level");
    string sim_data_foldername;
    string result_file_name;
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
    vector<vector<bool>> bool_patterns = generateCorrelatedPatterns(nb_patterns, network_size, noise_level);

    vector<vector<double>> patterns = patterns_as_states(1, -1, bool_patterns);
    vector<vector<bool>> connectivity_matrix(network_size, vector<bool>(network_size, true));
    for (int i = 0; i < network_size; i++)
    {
        connectivity_matrix[i][i] = false;
    }
    Network net(connectivity_matrix, network_size);
    // net.trainPerceptron(patterns,network_size,1.0/network_size);
    if(learning_rule==0){
        net.trainHebbian(patterns);
    }else{
        net.trainPerceptron(patterns,network_size,1.0/network_size);
    }

    // parameters["mean_crosstalk"] = net.ComputeMeanCrossTalk(bool_patterns);

    // Create hash set of stored patterns for O(1) lookup (instead of O(P×N))
    unordered_set<vector<double>, VectorDoubleHashSymmetric, VectorDoubleEqualSymmetric> pattern_set;
    for(const auto& pattern : patterns)
    {
        pattern_set.insert(pattern);
    }

    vector<bool> query;
    vector<double> initial_state;
    unordered_set<vector<double>, VectorDoubleHashSymmetric, VectorDoubleEqualSymmetric> retrieved_patterns_set;

    bool spurious=false;
    bool success;
    result_file_name = sim_data_foldername + "/results.data";
    std::ofstream result_file(result_file_name, std::ios::trunc);
    result_file << "success,"<<endl;
    for (size_t index = 0; index < nb_query; index++)
    {
        query = generateRandomPattern(network_size);
        initial_state = pattern_as_states(1, -1, query);
        vector<double>  queried_state = net.runAsynchronous(initial_state,2);

        // O(1) hash lookup instead of O(P×N) linear search
        if(pattern_set.find(queried_state) != pattern_set.end())
        {
            // Pattern matches one of the stored patterns
            // Insert into retrieved set (automatically handles duplicates)
            retrieved_patterns_set.insert(queried_state);
        }
        else
        {
            // This is a spurious pattern
            spurious = true;
            break;
        }
    }
    if ((retrieved_patterns_set.size() == nb_patterns) && !spurious)
    {
        success=1;
    }
    else{
        success=0;
    }
    result_file << to_string(success) << ",";
    result_file.close();
    createParameterFile(sim_data_foldername, parameters);
}


int main(int argc, char **argv)
{
    // string sim_name = "write_net_sizes_relative_num_patterns";
    string sim_name = "correlation_random_query_perceptron_success";
    string foldername_results = "./" + sim_name;

    // Create directory if it doesn't exist
    if (!fs::exists(foldername_results))
    {
        if (!fs::create_directory(foldername_results))
        {
            std::cerr << "Error creating directory: " << foldername_results << std::endl;
            return 1;
        }
    }

    vector<double> nb_query = {30};
    vector<double> noise_level = {0.0, 0.25, 0.5, 0.75, 1};
    vector<double> num_patterns = generateEvenlySpacedIntegers(1, 25, 25);
    vector<double> network_sizes = generateEvenlySpacedIntegers(25, 250, 20);
    // vector<double> ratio_rnd_bits= {0.5};
    vector<double> repetition = generateEvenlySpacedIntegers(0, 10, 10);
    unordered_map<string, vector<double>> varying_params = {
        {"net_size", network_sizes},
        {"nb_pat", num_patterns},
        {"nb_query", nb_query},
        {"noise_level", noise_level},
        {"learning_rule", {1}},
        {"repetition", repetition}};

    lunchParalSimThreadLimit(12,foldername_results, varying_params, run_simulation);
    collectSimulationDataSeries(foldername_results);

    return 0;
}
