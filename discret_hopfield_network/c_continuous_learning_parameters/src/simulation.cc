#include "network.hpp"
#include "utils.hpp"
#include <vector>
#include <iostream>
#include <random>
#include <ctime>

using namespace std;

int main(int argc, char **argv)
{

    int initial_network_size = 50;
    int num_patterns = 10;
    int nb_winners = initial_network_size / 2;
    double noise_level = 0.3;
    int nb_iter = 500;

    vector<vector<bool>> bool_patterns = generatePatterns(num_patterns, initial_network_size, nb_winners, noise_level);
    // vector<vector<bool>> bool_combined_patterns = generateCombinedPatterns(bool_patterns);
    vector<vector<double>> patterns = patterns_as_states(1, -1, bool_patterns);
    // vector<vector<double>> combined_patterns = patterns_as_states(1, -1, bool_combined_patterns);

    int network_size = initial_network_size * 2;
    // num_patterns = combined_patterns.size();
    cout<< "combined patterns"<<endl;
    // for (const auto &pattern : combined_patterns)
    // {
    //     cout << "Pattern: ";
    //     for (const auto &val : pattern)
    //         cout << val << " ";
    //     cout << "\n";
    // }
    // cout << endl;
    cout << "base patterns" << endl;
    for (const auto &pattern : patterns)
    {
        cout << "Pattern: ";
        for (const auto &val : pattern)
            cout << val << " ";
        cout << "\n";
    }
    cout << endl;

    vector<vector<bool>> connectivity_matrix(network_size, vector<bool>(network_size, true));
    for (int i = 0; i < network_size; i++)
    {
        connectivity_matrix[i][i] = false;
    }

    Network net(connectivity_matrix, network_size);

    vector<vector<bool>> connectivity_matrix_2(initial_network_size, vector<bool>(initial_network_size, true));
    for (int i = 0; i < initial_network_size; i++)
    {
        connectivity_matrix_2[i][i] = false;
    }
    Network second_net(connectivity_matrix_2,initial_network_size);
    
    vector<vector<bool>> first_patterns(2);
    first_patterns[0] = bool_patterns[0];
    first_patterns[1] = bool_patterns[1];
    vector<vector<bool>> bool_combined_patterns = generateCombinedPatterns(first_patterns);
    vector<vector<double>> combined_patterns = patterns_as_states(1,-1,bool_combined_patterns); 
    for(int i =0; i <nb_iter; i++){
        net.trainPerceptronInput(combined_patterns,initial_network_size);
    }

    // vector<double> retrieved_state;

    int nb_stored_patterns = 1;
    vector<double> pattern;
    vector<vector<double>> to_write_states;
    for (int k = 0; k<patterns.size();  k++){
        to_write_states.clear();
        pattern = patterns[k];
        to_write_states.emplace_back(patterns[0]);
        std::cout << "nombre de pattern écrits : " <<nb_stored_patterns << std::endl;
        to_write_states.emplace_back(pattern);
        vector<double> retrieved_state = patterns[0];
        for(int i=0; i<nb_stored_patterns;i++){
            std::cout << "exploring stored patterns : " <<i<< std::endl;
            vector<double> initial_state(network_size,1.0);
            for(int i = 0; i< initial_network_size; i++){
                initial_state[i] = retrieved_state[i];
            }
            vector<double> ret = net.runAndClamp(initial_state,initial_network_size);
            retrieved_state = pickLastNElements(ret, initial_network_size);
            to_write_states.emplace_back(retrieved_state);
        }
        combined_patterns = generateCombinedStates(to_write_states);
        for(int i =0; i <nb_iter; i++){
            net.trainPerceptronInput(combined_patterns,initial_network_size);
        }
        nb_stored_patterns++;
    }

    for(int i =0; i <nb_iter; i++){
        second_net.trainPerceptron(to_write_states);
    }

    
    int num_random_elements = 15;
    vector<double> initial_state;
    vector<double> final_state;
    int iter = 0;
    int nb_successd = 0;
    for (const auto &pattern : patterns){
        initial_state = randomizeInitialState(pattern, num_random_elements);
        final_state = second_net.run(initial_state);
        // final_state = second_net.stochastic_run_annealing(initial_state, 200, 0.05, 0.0);
        bool success = compareStates(pattern, final_state);
        cout << "Noisy pattern: ";
        for (const auto &val : initial_state)
            cout << val << " ";
        cout << "\nPattern: ";
        for (const auto &val : pattern)
            cout << val << " ";
        cout << "\nRetriez: ";
        for (const auto &val : final_state)
            cout << val << " ";
        cout << "\n"
             << (success ? "Success" : "Failure") << endl
             << endl;
            
        if (success){
            nb_successd+=1;
        }
        iter+=1;
    }
    std::cout << "nb win : " << nb_successd << std::endl;
    std::cout << "nb lost : " << ((patterns.size()) - nb_successd) << std::endl;


    // int nb_successd = 1;
    // vector<double> retrieved_state = patterns[0];
    // vector<double> random_vector;
    // int iter = 0;

    // for (const auto &pattern : combined_patterns)
    // {
    //     vector<double> initial_state(network_size,1.0);
    //     for(int i = 0; i< initial_network_size; i++){
    //         initial_state[i] = retrieved_state[i];
    //     }
    //     vector<double> ret = net.runAndClamp(initial_state,initial_network_size);
    //     retrieved_state = pickLastNElements(ret, initial_network_size);
    //     for (const auto &tr : retrieved_state)
    //         cout << tr << " ";
    //     cout << endl;

    //     bool success = compareStates(patterns[iter+1], retrieved_state);

    //     // cout << "Noisy pattern: ";
    //     // for (const auto &val : initial_state)
    //     //     cout << val << " ";
    //     cout << "\nPattern: ";
    //     for (const auto &val : patterns[iter + 1])
    //         cout << val << " ";
    //     cout << "\nRetriez: ";
    //     for (const auto &val : retrieved_state)
    //         cout << val << " ";
    //     cout << "\n"
    //          << (success ? "Success" : "Failure") << endl
    //          << endl;
            
    //     if (success){
    //         nb_successd+=1;
    //     }
    //     iter+=1;
    // }
    // std::cout << "nb win : " << nb_successd << std::endl;
    // std::cout << "nb lost : " << ((patterns.size()) - nb_successd) << std::endl;

    return 0;
}
