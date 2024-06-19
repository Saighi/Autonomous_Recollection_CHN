#include "network.hpp"
#include "utils.hpp"
#include <vector>
#include <iostream>
#include <random>
#include <ctime>

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

int main(int argc, char **argv)
{
    int network_size = 100;
    int num_patterns = 49;
    int nb_winners = network_size / 2;
    double noise_level = 1;
    int nb_iter = 500;
    int index_input = 50;

    vector<vector<bool>> bool_patterns = generatePatterns(num_patterns, network_size, nb_winners, noise_level);
    vector<vector<double>> patterns = patterns_as_states(1, -1, bool_patterns);

    for(const auto& pattern: patterns){
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
    for(int i =0; i <nb_iter; i++){ 
        net.trainPerceptronInput(patterns,index_input);
    }
    // net.train(patterns);

    // int num_random_elements = 50; // Number of elements to randomize
    int nb_successd = 0;
    vector<double> initial_state(network_size,0);
    for (const auto &pattern : patterns)
    {
        for(int i =0; i<index_input;i++){
            initial_state[i] = pattern[i];
        }

        // cout << "iNITIZAL STATE: ";
        // for (const auto &val : initial_state)
        //     cout << val << " ";
            // vector<double> initial_state = randomizeInitialState(pattern, num_random_elements);
        // vector<double> retrieved_state = net.run(initial_state);
        vector<double> retrieved_state = net.runAndClamp(initial_state,index_input,10);
        bool success = compareStates(pattern, retrieved_state);
        // cout << "Noisy pattern: ";
        // for (const auto &val : initial_state)
        //     cout << val << " ";
        cout << "\nPattern: ";
        for (const auto &val : pattern)
            cout << val << " ";
        cout << "\nRetrieved: ";
        for (const auto &val : retrieved_state)
            cout << val << " ";
        cout << "\n"
             << (success ? "Success" : "Failure") << endl
             << endl;
            
        if (success){
            nb_successd+=1;
        }
    }
    std::cout << "nb win : " << nb_successd << std::endl;
    std::cout << "nb lost : "<<(patterns.size()-nb_successd) << std::endl;

    return 0;
}
