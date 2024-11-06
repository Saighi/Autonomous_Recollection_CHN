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
#include <matio.h>

const int IMAGE_HEIGHT = 20;
const int IMAGE_WIDTH = 16;
using namespace std;

namespace fs = std::filesystem;


void run_simulation(int sim_number, unordered_map<string, double> parameters, const string foldername_results)
{
    // Learning constants
    int cpt=0;
    double epsilon_learning=0.06;
    double drive_target = parameters.at("drive_target");
    double learning_rate = parameters.at("learning_rate");
    int network_size = parameters.at("network_size");
    int nb_winners;
    double leak = parameters.at("leak");
    double delta = parameters.at("delta");
    int col_with = IMAGE_HEIGHT;
    string sim_data_foldername;
    string result_file_name;
    string weights_file_name;
    string connectivity_file_name;
    vector<vector<bool>> initial_patterns;
    vector<vector<bool>> query_patterns;
    vector<vector<double>> initial_patterns_rates;
    vector<vector<double>> query_patterns_rates;
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
    // Open the .mat file
    mat_t *matfp = Mat_Open("binaryalphadigs.mat", MAT_ACC_RDONLY);
    // Read the 'dat' variable
    matvar_t *matvar = Mat_VarRead(matfp, "dat");
    // Check if 'dat' is a cell array
    // Access the first cell
    vector<int> numbers ={3,3,3,3,6,6};
    for (size_t n = 0; n < numbers.size(); n++)
    {
        matvar_t *cell = static_cast<matvar_t **>(matvar->data)[numbers[n]+(36*n)];
        // Get the data pointer
        void *data = cell->data;
        // Display the first image
        cpt=0;
        std::cout << "First Image:" << std::endl;
        initial_patterns.emplace_back(vector<bool>(20*16,0));
        query_patterns.emplace_back(vector<bool>(20*16,0));
        for (size_t i = 0; i < IMAGE_HEIGHT; ++i)
        {
            for (size_t j = 0; j < IMAGE_WIDTH; ++j)
            {
                // Adjust for column-major order
                size_t idx = i + j * IMAGE_HEIGHT;
                double pixel_value;

                if (cell->class_type == MAT_C_DOUBLE)
                {
                    pixel_value = static_cast<double *>(data)[idx];
                }
                else if (cell->class_type == MAT_C_UINT8)
                {
                    pixel_value = static_cast<uint8_t *>(data)[idx];
                }

                // Interpret pixel value as binary
                bool is_pixel_on = (pixel_value == 1);
                initial_patterns[initial_patterns.size()-1][cpt]=is_pixel_on;
                std::cout << (is_pixel_on ? "#" : " ");
                cpt+=1;
            }
            std::cout << std::endl;
        }

        for (size_t i = 0; i < initial_patterns[initial_patterns.size()-1].size(); i++)
        {
            if (i>initial_patterns[initial_patterns.size()-1].size()/3){
                query_patterns[initial_patterns.size()-1][i]= false;
            }
            else{

                query_patterns[initial_patterns.size()-1][i]= initial_patterns[initial_patterns.size()-1][i];
            }
        }

    }

    // Clean up
    Mat_VarFree(matvar);
    Mat_Close(matfp);
    
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
    initial_patterns_rates = patterns_as_states(net.transfer(drive_target), net.transfer(-drive_target), initial_patterns);
    query_patterns_rates = patterns_as_states(net.transfer(drive_target), net.transfer(-drive_target), query_patterns);
    vector<double> drive_errors;
    drive_errors.resize(network_size,0.0);
    double sum_errors;
    bool stop_learning = false;
    cpt=0;
    // Training loop
    // show_vector_bool_grid(initial_patterns[0],IMAGE_HEIGHT);
    // show_vector_bool_grid(query_patterns[0],IMAGE_HEIGHT);
    std::cout << "WRITING ATTRACTORS" << std::endl;
    while(!stop_learning)
    {
        sum_errors=0.0;
        for (int j = 0; j < initial_patterns.size(); j++)
        {
            net.derivative_gradient_descent(initial_patterns[j],initial_patterns_rates[j],drive_target,learning_rate, leak, drive_errors);
            sum_errors+=std::accumulate(drive_errors.begin(),drive_errors.end(),0.0);
        }
        if(abs(sum_errors)/(network_size*initial_patterns.size())<epsilon_learning){
            stop_learning=true;
        }
        if(cpt==5000){
            stop_learning=true;
        }
        cpt+=1;
        std::cout <<  abs(sum_errors)/(network_size*initial_patterns.size()) << std::endl;
        std::cout << cpt << std::endl;
    }
    std::cout << "nombre d'iterations" << std::endl;
    std::cout << cpt << std::endl;
    // Querying
    std::cout << "Querying initial memories" << std::endl;
    vector<double> query_pattern;
    int succes = 0 ;
    // double strength_drive = 0.1;
    for (int i = 0; i < initial_patterns_rates.size(); i++)
    {
        nb_winners=0;
        string result_file_traj_name = sim_data_foldername + "/results_" + to_string(i) + ".data";
        std::ofstream result_file_traj(result_file_traj_name, std::ios::trunc);
        //TODO - change the pattern_as_states and link the target drive not magic number
        query_pattern=query_patterns_rates[i];
        net.set_state(query_pattern);
        // run_net_sim_query_drive(net, noisy_pattern, strength_drive, 1200, delta);
        // run_net_sim_noisy(net,2800, delta,0.0,0.01);
        run_net_sim_save(net,300, delta, result_file_traj);
        // run_net_sim_noisy_save_display(net,10, 2800, delta,0,0.01, result_file_traj);// 
        // run_net_sim_noisy_save(net, 2800, delta,0,0.01, result_file_traj);
        for (size_t j = 0; j < initial_patterns[i].size(); j++)
        {
            nb_winners+=initial_patterns[i][j];   
        }
        winning_units = assignBoolThreshold(net.activity_list, 0.5);
        if (comparestates(winning_units,initial_patterns[i])){
            succes+=1;
        }
        std::cout << "writed pattern :" << std::endl;
        std::cout << i << std::endl;
        show_vector_double_grid(net.activity_list,IMAGE_HEIGHT);
    }
    std::cout << "success ?" << std::endl;
    std::cout << to_string(succes) << std::endl;

    weights_file_name = sim_data_foldername + "/weights.data";
    writeMatrixToFile(net.weight_matrix, weights_file_name);

    connectivity_file_name = sim_data_foldername + "/connectivity.data";
    writeBoolMatrixToFile(net.connectivity_matrix, connectivity_file_name);
    std::cout << nb_winners << std::endl;
}

int main(int argc, char **argv)
{
    // string sim_name = "write_net_sizes_relative_num_patterns";
    string sim_name = "Fig_Biased_Recollection";
    string foldername_results = "../../../data/all_data_splited/trained_networks_fast/" + sim_name;

    // Create directory if it doesn't exist
    if (!fs::exists(foldername_results))
    {
        if (!fs::create_directory(foldername_results))
        {
            std::cerr << "Error creating directory: " << foldername_results << std::endl;
            return 1;
        }
    }
    // vector<double> all_relative_num_patterns = {0.5};
    // vector<double> network_sizes = {10,20,30,40,50,60,70,80,90,100};
    vector<double> drive_targets = {6};
    vector<double> network_sizes = {IMAGE_HEIGHT*IMAGE_WIDTH};
    // vector<double> repetitions = {0,1,2,3,4,5,6,7,8,9};
    unordered_map<string, vector<double>> varying_params = {
        // {"repetitions", repetitions},
        {"drive_target", drive_targets},
        {"learning_rate", {0.0001}}, // REMOVED-target rates
        {"network_size", network_sizes},
        {"leak", {1.3}},
        {"delta", {0.02}}};
    

    lunchParalSim(foldername_results,varying_params,run_simulation);
    collectSimulationData(foldername_results);
    
    return 0;
}
