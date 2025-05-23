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

void run_sleep(int sim_number, std::vector<std::vector<double>> net_weights, std::vector<std::vector<bool>> net_connectivity, const unordered_map<string, double> parameters, const string foldername_results, vector<vector<bool>> patterns, bool save_trajectories)
{
    save_trajectories = false;
    // Inherited
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
    int nb_iter = static_cast<int>(parameters.at("nb_iter_mult") * parameters.at("max_pattern"));
    std::cout <<"sim_number :"<< sim_number<< std::endl;
    int col_with = sqrt(network_size);
    Network net = Network(net_connectivity, network_size, leak);
    net.weight_matrix= net_weights;

    string sim_data_foldername = foldername_results + "/sim_nb_" + to_string(sim_number);

    createDirectory(sim_data_foldername.c_str());

    // Store the inherited patterns
    string patterns_file_name = sim_data_foldername + "/patterns.data";
    std::ofstream file(patterns_file_name, std::ios::trunc);
    for (int i = 0; i < num_patterns; i++)
    {
        writeBoolToCSV(file, patterns[i]);
        // show_vector_bool_grid(patterns[i], col_with);
    }
    file.close();

    createParameterFile(sim_data_foldername, parameters);
    // SLEEPING SIMULATIONS

    string result_file_name_sleep;
    string result_file_name_retrieval;
    int iter_all_retrieved;
    int nb_spurious_patterns = 0;
    bool check = false;
    std::set<std::vector<bool>> foundVectors;
    vector<bool> winning_units;

    result_file_name_retrieval = sim_data_foldername + "/results.data";
    std::ofstream result_file_retrieval(result_file_name_retrieval, std::ios::trunc);
    // std::cout << "SLEEP PHASE" << std::endl;
    vector<bool> is_found_patterns;
    float sum_rates;
    
    result_file_retrieval << "query_iter,";
    result_file_retrieval << "nb_fnd_pat,";
    result_file_retrieval << "nb_spurious," << endl;

    int cpt = 0;
    int nb_iter_sim = 0;
    while(cpt<nb_iter)
    {
        // std::cout << "NEW ITER" << std::endl;
        net.set_state(vector<double>(network_size, init_drive));
        // std::cout << "Initial random state:" << std::endl;
        // show_state_grid(net, 3); // Show initial state

        SimulationConfig config;
        if (save_trajectories){
            result_file_name_sleep = sim_data_foldername + "/results_" + to_string(is_found_patterns.size()) + ".data";
        }
        else{
            result_file_name_sleep = "no_save.data";
        }
        std::ofstream result_file_sleep(result_file_name_sleep, std::ios::trunc);
        config.delta = delta;
        config.epsilon = delta/100;
        config.depressed = true;
        config.save = false;
        config.max_iter = 100/delta;
        nb_iter_sim += run_net_sim_choice(net, config); 
        config.depressed = false;
        nb_iter_sim += run_net_sim_choice(net, config); 
        result_file_sleep.close();
        winning_units = assignBoolToTopNValues(net.activity_list, nb_winners);
        net.pot_inhib_bin(beta, winning_units); // works with 0.005
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
        }
        if (foundVectors.size() == patterns.size() && !check)
        {
            iter_all_retrieved = is_found_patterns.size()-1;
            check = true;
        }

        result_file_retrieval << to_string(cpt) <<",";
        result_file_retrieval << to_string(foundVectors.size()) <<",";
        result_file_retrieval << to_string(nb_spurious_patterns) <<"," << endl;
        cpt+=1;
    }

    // result_file_retrieval << "general results :" << std::endl;
    // result_file_retrieval << "iter_all_retrieved=" << iter_all_retrieved << std::endl;
    // // The number of unique vectors found
    // std::cout << "Number of unique vectors found: " << foundVectors.size() << " nb_patterns : " << num_patterns << " beta : " << beta << " nb_spurious : " << nb_spurious_patterns << std::endl
    //           << " nb_winers : " << nb_winners << std::endl;
    std::cout <<"sim_number " << sim_number << std::endl;
    std::cout <<"nb_spurious :"<< nb_spurious_patterns <<" nb_found_patterns : "<< foundVectors.size() << " nb_patterns : " << num_patterns << " beta : " <<" nb_flip : " << " Network size: " << network_size << std::endl;

    result_file_retrieval.close();
    
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    bool save_trajectories = false;
    // string sim_name = "Fig_load_SR_average_MPI_test";
    string sim_name = argv[1];
    string inputs_name = sim_name;
    string foldername_inputs = argv[2];
    foldername_inputs = foldername_inputs+sim_name;
    string foldername_results = argv[3];
    foldername_results = foldername_results+sim_name;
    // string inputs_name = "write_parameter_many_nb_iter_learning";
    // string foldername_results = "../../../data/all_data_splited/sleep_simulations/" + sim_name;
    // string foldername_inputs = "../../../data/all_data_splited/trained_networks_fast/" + inputs_name;
    // Create directory if it doesn't exist
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

    MPI_Barrier(MPI_COMM_WORLD);
    vector<double> beta = linspace(0.0001,0.005,15);
    unordered_map<string, vector<double>> varying_params = {
        {"beta", beta},
        {"nb_iter_mult", {4}}};   

    vector<vector<bool>> patterns;
    vector<vector<double>> net_weights;
    vector<vector<bool>> net_connectivity;
    string patterns_file_name;

    vector<unordered_map<string, double>> combinations = generateCombinations(varying_params);
    vector<string> all_paths;
    // Check if the path exists and is a directory
    all_paths=getSubfolderPaths(foldername_inputs);

    vector<pair<string,int>> path_configId_pairs;
    for (const auto &path : all_paths)
    {
        for(int i=0; i<combinations.size();i++){
            path_configId_pairs.push_back({path,i});
        }
    }

    int nb_simulations = path_configId_pairs.size();
    vector<vector<int>> ranks_affiliated_sim = ranks_processes(size, nb_simulations);
    unordered_map<string,double> fused_parameters;
    unordered_map<string,double> local_parameters;
    unordered_map<string, double> inherited_params;
    string path;
    for(int sim_number: ranks_affiliated_sim[rank])
    {
        pair path_and_paramas=path_configId_pairs[sim_number];
        path = path_and_paramas.first;
        local_parameters = combinations[path_and_paramas.second];
        inherited_params = readParametersFile(path + "/parameters.data");
        net_weights = readMatrixFromFile(path + "/weights.data");
        net_connectivity = readBoolMatrixFromFile(path + "/connectivity.data");
        patterns_file_name = path + "/patterns.data";
        patterns = loadPatterns(patterns_file_name);
        fused_parameters = fuseMaps(inherited_params, local_parameters);
        cout << "Process " << rank << " running simulation " << sim_number
                << " of " << nb_simulations << endl;
        run_sleep(sim_number, net_weights, net_connectivity, fused_parameters, foldername_results, patterns, save_trajectories);
        // cout<<"process : "<< rank<< " is dealing with path"<< path_and_paramas.first<<endl; 
        // cout<<"process : "<< rank<< " is dealing with beta"<< combinations[path_and_paramas.second].at("beta")<<endl; 
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

    MPI_Barrier(MPI_COMM_WORLD);
    if(rank==0){
        collectSimulationDataSeries(foldername_results);
    }
    MPI_Finalize();
    return 0;
}
