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
#include <cstdlib>

using namespace std;

namespace fs = std::filesystem;

void run_sleep(int sim_number, std::vector<std::vector<double>> net_weights, std::vector<std::vector<bool>> net_connectivity, const unordered_map<string, double> parameters, const string foldername_results, vector<vector<bool>> patterns)
{
    srand(sim_number);
    std::cout <<"sim_number :"<< sim_number<< std::endl;
    bool save_trajectories=false;
    if (parameters.at("save")){
        save_trajectories=true;
    }
    // Inherited
    double drive_target = parameters.at("drive_target");
    double init_drive = parameters.at("init_drive");
    double learning_rate = parameters.at("learning_rate");
    int network_size = static_cast<int>(parameters.at("network_size"));
    int nb_winners = static_cast<int>(parameters.at("nb_winners")); // number of 1's neurons
    double noise_level = parameters.at("noise_level");
    double leak = parameters.at("leak");
    double delta = parameters.at("delta");
    int num_patterns = static_cast<int>(parameters.at("num_patterns"));
    // not Inherited
    float beta = parameters.at("beta");
    bool noise = false;
    if(parameters.at("noise")==1){
        noise=true;
    }
    double stddev=parameters.at("stddev");
    int nb_iter_max = static_cast<int>(parameters.at("nb_iter_mult_max") * parameters.at("num_patterns"));
    int col_with = sqrt(network_size);

    Network net = Network(net_connectivity, network_size, leak);
    net.weight_matrix= net_weights;

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

    // Store the inherited patterns
    string patterns_file_name = sim_data_foldername + "/patterns.data";
    std::ofstream file(patterns_file_name, std::ios::trunc);
    for (int i = 0; i < num_patterns; i++)
    {
        writeBoolToCSV(file, patterns[i]);
        // show_vector_bool_grid(patterns[i], 10);
    }
    file.close();
    createParameterFile(sim_data_foldername, parameters);

    // PREPARATION SLEEP SIMULATION

    string result_file_name_trajectories;
    string result_file_name_retrieval;
    string result_file_name_corr_stable_states;
    result_file_name_retrieval = sim_data_foldername + "/results.data";
    std::ofstream result_file_retrieval(result_file_name_retrieval, std::ios::trunc);
    result_file_name_corr_stable_states = sim_data_foldername + "/results_corr_stable_states.data";
    std::ofstream result_file_corr_stable_states(
        result_file_name_corr_stable_states, std::ios::trunc);

    int iter_all_retrieved;
    int nb_spurious_patterns = 0;
    std::set<std::vector<bool>> foundVectors;
    vector<bool> winning_units;


    // Global configuration of the simulation    
    SimulationConfig config;
    config.delta = delta;
    config.epsilon = delta/10000;
    config.save = save_trajectories;
    config.max_iter = 100/delta;
    config.noise = noise;
    config.stddev=stddev;
    
    // std::cout << "SLEEP PHASE" << std::endl;
    vector<bool> is_found_patterns;
    float sum_rates;

    result_file_retrieval << "query_iter,";
    result_file_retrieval << "nb_fnd_pat,"<< endl;

    int cpt = 0;
    int nb_iter_sim = 0;
    bool spurious = false;
    bool all_found = false;
    std::cout << "RETRIEVAL" << std::endl;
    while(cpt<nb_iter_max && !spurious && !all_found)
    {
        if (save_trajectories) {
            result_file_name_trajectories =
                sim_data_foldername + "/results_" +
                to_string(is_found_patterns.size()) + ".data";
        } else {
            result_file_name_trajectories = "no_save.data";
        }
        std::ofstream result_file_trajectories(result_file_name_trajectories,
                                        std::ios::trunc);
        config.output = &result_file_trajectories;

        config.depressed = true;
        check_stable_states(net, patterns, init_drive, drive_target, config,
                            &result_file_corr_stable_states);

        // std::cout << "NEW ITER" << std::endl;
        net.set_state(vector<double>(network_size, init_drive));
        // std::cout << "Initial random state:" << std::endl;
        // show_state_grid(net, 3); // Show initial state
        nb_iter_sim += run_net_sim_choice(net, config); 
        config.depressed = false;
        nb_iter_sim += run_net_sim_choice(net, config);
        winning_units = assignBoolToTopNValues(net.activity_list, nb_winners);
        // show_vector_bool_grid(winning_units,10);
        // net.pot_inhib_bin(beta, winning_units); // works with 0.005
        net.pot_inhib_symmetric(beta); // works with 0.005
        // Check if the output vector is in the target set and hasn't been counted yet
        is_found_patterns.push_back(false);
        if (std::find(patterns.begin(), patterns.end(), winning_units) != patterns.end())
        {
            if (foundVectors.find(winning_units) == foundVectors.end())
            {
                // Count the vector
                foundVectors.insert(winning_units);
                is_found_patterns.back() = true;
            }
        }
        else
        {
            nb_spurious_patterns += 1;
            spurious=true;
            std::cout << "SPURIOUS" << std::endl; 
        }
        if (foundVectors.size() == patterns.size())
        {
            iter_all_retrieved = is_found_patterns.size()-1;
            all_found = true;
            std::cout << "ALL_FOUND" << std::endl; 
        }

        result_file_retrieval << to_string(cpt) <<",";
        result_file_retrieval << to_string(foundVectors.size()) <<std::endl;
        
        result_file_trajectories.close();
        
        cpt+=1;
    }

    // result_file_retrieval << "general results :" << std::endl;
    // result_file_retrieval << "iter_all_retrieved=" << iter_all_retrieved << std::endl;
    // // The number of unique vectors found
    // std::cout << "Number of unique vectors found: " << foundVectors.size() << " nb_patterns : " << num_patterns << " beta : " << beta << " nb_spurious : " << nb_spurious_patterns << std::endl
    //           << " nb_winers : " << nb_winners << std::endl;
    std::cout <<"sim_number " << sim_number << std::endl;
    std::cout <<" nb_found_patterns : "<< foundVectors.size() << " nb_patterns : " << num_patterns << " beta : " <<" nb_flip : " << " Network size: " << network_size << std::endl;

    result_file_retrieval.close();
}

int main(int argc, char **argv)
{
    // string sim_name = "Fig_load_SR_average_new_inh_plas_many_betta_larger_networks_2";
    // string inputs_name = "Fig_load_SR_average_new_inh_plas_many_betta_larger_networks_2";
    string sim_name = "Explo_stable_state_deformation";
    string inputs_name = "Explo_stable_state_deformation";
    // string inputs_name = "write_parameter_many_nb_iter_learning";
    string foldername_results = "../../../data/all_data_splited/sleep_simulations/" + sim_name;
    fs::path foldername_inputs = "../../../data/all_data_splited/trained_networks_fast/" + inputs_name;
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
    // vector<double> beta = linspace(0.00125/10, 0.00125, 3);
    vector<double> beta = {0.01};
    
    unordered_map<string, vector<double>> varying_params = {
        {"save", {0}},
        {"beta", beta},
        {"delta",{0.01}},
        {"noise",{1}},
        {"stddev",{0.01}},
        {"nb_iter_mult_max", {20}}};

    unordered_map<string, double> inherited_params;
    vector<vector<bool>> patterns;
    vector<vector<double>> net_weights;
    vector<vector<bool>> net_connectivity;
    string patterns_file_name;

    vector<unordered_map<string, double>> combinations = generateCombinations(varying_params);
    unordered_map<string,double> fused_parameters;
    vector<string> all_paths;
    // Check if the path exists and is a directory
    std::cout << foldername_inputs << std::endl;
    if (fs::exists(foldername_inputs) && fs::is_directory(foldername_inputs))
    {
        // Iterate over the directory entries
        for (const auto &entry : fs::directory_iterator(foldername_inputs))
        {
            std::cout << entry << std::endl;
            // Check if the entry is a directory
            if (fs::is_directory(entry.path()))
            {
                all_paths.push_back(entry.path().generic_string());
           }
        }
    }
    const int max_threads = 20; // Set the maximum number of concurrent threads
    int active_threads = 0;
    std::mutex mtx;
    std::condition_variable cv;
    std::vector<std::thread> threads;
    int all_sim_number = 0;
    for (const auto &path : all_paths)
    {
        std::cout << path << std::endl;
        inherited_params = readParametersFile(path + "/parameters.data");
        net_weights = readMatrixFromFile(path + "/weights.data");
        net_connectivity = readBoolMatrixFromFile(path + "/connectivity.data");
        patterns_file_name = path + "/patterns.data";


        patterns = loadPatterns(patterns_file_name);

        for (int sim_number = 0; sim_number < combinations.size(); ++sim_number)
        {
            fused_parameters = fuseMaps(inherited_params, combinations[sim_number]);

            {
                std::unique_lock<std::mutex> lock(mtx);
                cv.wait(lock, [&]
                        { return active_threads < max_threads; });
                ++active_threads;
            }

            threads.emplace_back([=, &mtx, &cv, &active_threads]
                                 {
                run_sleep(all_sim_number, net_weights, net_connectivity, fused_parameters, foldername_results, patterns);
                {
                    std::lock_guard<std::mutex> lock(mtx);
                    --active_threads;
                }
                cv.notify_all(); });

            all_sim_number += 1;
        }
    }

    for (auto &t : threads)
    {
        if (t.joinable())
        {
            t.join();
        }
    }

    collectSimulationDataSeries(foldername_results);

    return 0;
}
