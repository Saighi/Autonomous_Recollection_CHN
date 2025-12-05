#include "utils.hpp"
#include "network.hpp"
#include <numeric>
#include <iostream>
#include <iomanip>
#include <vector>
#include <queue>
#include <utility>
#include <algorithm>
#include <fstream>
#include <filesystem>
#include <string>
#include <unordered_set>
#include <cstdlib>
#include <ctime>
#include <random>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <regex>

namespace fs = std::filesystem;

// Hash function implementation for vector<bool>
std::size_t VectorBoolHash::operator()(const std::vector<bool>& vec) const {
    std::size_t hash = 0;
    std::size_t prime = 31;
    for (size_t i = 0; i < vec.size(); ++i) {
        hash = hash * prime + (vec[i] ? 1 : 0);
    }
    return hash;
}

// Hash function implementation for vector<double>
// Treats values as discrete by comparing them (works for -1, 1 patterns)
std::size_t VectorDoubleHash::operator()(const std::vector<double>& vec) const {
    std::size_t hash = 0;
    std::size_t prime = 31;
    for (size_t i = 0; i < vec.size(); ++i) {
        // Convert to discrete values (1 if positive, 0 otherwise)
        hash = hash * prime + (vec[i] > 0 ? 1 : 0);
    }
    return hash;
}

// Symmetric hash: same hash for pattern and its inverse
// Achieves this by XORing the normal hash with the inverse hash
std::size_t VectorDoubleHashSymmetric::operator()(const std::vector<double>& vec) const {
    std::size_t hash1 = 0;
    std::size_t hash2 = 0;
    std::size_t prime = 31;
    for (size_t i = 0; i < vec.size(); ++i) {
        int val = (vec[i] > 0 ? 1 : 0);
        hash1 = hash1 * prime + val;
        hash2 = hash2 * prime + (1 - val);  // Inverse pattern
    }
    // XOR ensures same result regardless of inversion
    return hash1 ^ hash2;
}

// Equality comparator that returns true for direct match OR inverse match
bool VectorDoubleEqualSymmetric::operator()(const std::vector<double>& a, const std::vector<double>& b) const {
    if (a.size() != b.size()) return false;

    bool direct_match = true;
    bool inverse_match = true;

    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) direct_match = false;
        if (a[i] != -b[i]) inverse_match = false;
        if (!direct_match && !inverse_match) return false;
    }

    return direct_match || inverse_match;
}

// Function to convert a single binary pattern to a state vector
std::vector<double> pattern_as_states(double up_rate, double down_rate, const std::vector<bool>& bin_pattern) {
    std::vector<double> state_input(bin_pattern.size());
    for (int j = 0; j < bin_pattern.size(); j++) {
        if (bin_pattern[j]) {
            state_input[j] = up_rate;
        } else {
            state_input[j] = down_rate;
        }
    }
    return state_input;
}

// Function to convert multiple binary patterns to state vectors
std::vector<std::vector<double>> patterns_as_states(double up_rate, double down_rate, const std::vector<std::vector<bool>>& bin_patterns) {
    std::vector<std::vector<double>> initial_patterns_state_list(bin_patterns.size());
    for (int i = 0; i < bin_patterns.size(); i++) {
        initial_patterns_state_list[i] = pattern_as_states(up_rate, down_rate, bin_patterns[i]);
    }
    return initial_patterns_state_list;
}



// Function to generate a base pattern with a specified number of 1s
std::vector<bool> generateRandomPattern(int N)
{
    std::vector<bool> basePattern(N, false);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 1);

    for (int i = 0; i < N; ++i)
    {
        basePattern[i] = (dis(gen) == 1);
    }

    return basePattern;
}


// Function to generate a vector of randomized indices
std::vector<int> generateRandomIndices(int size) {
    std::vector<int> indices(size);
    std::iota(indices.begin(), indices.end(), 0); // Fill with 0, 1, ..., size-1

    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(indices.begin(), indices.end(), gen); // Shuffle the indices

    return indices;
}

// Function to flip bits in a balanced pattern based on noise level
std::vector<bool> randomizePattern(const std::vector<bool> &basePattern, int nb_random_bits)
{
    std::vector<bool> noisyPattern = basePattern;
    int N = basePattern.size();
    std::vector<int> random_indices = generateRandomIndices(basePattern.size());
    int cpt=0;
    std::random_device rd;  // Obtain a random number from hardware
    std::mt19937 gen(rd()); // Seed the generator
    std::uniform_int_distribution<> dis(0, 1); // Define the range

    while (cpt < nb_random_bits) // can loop a long time if not enough are 1s
    {
        noisyPattern[random_indices[cpt]] = (dis(gen)==1);
        cpt+=1;
    }
    return noisyPattern;
}

// Function to generate K unique noisy balanced patterns
// Optimized with hash set for O(1) uniqueness checking
std::vector<std::vector<bool>> generateCorrelatedPatterns(int nbPattern, int networkSize, double ratio_random_bits)
{
    std::vector<std::vector<bool>> patterns;
    std::unordered_set<std::vector<bool>, VectorBoolHash> pattern_set;
    int nb_random_bits = static_cast<int>(ratio_random_bits*networkSize);

    // If noise level is too low to generate enough unique patterns,
    // generate completely random patterns instead
    if (nb_random_bits == 0 && nbPattern > 1)
    {
        while (patterns.size() < nbPattern)
        {
            std::vector<bool> newPattern = generateRandomPattern(networkSize);
            if (pattern_set.find(newPattern) == pattern_set.end())
            {
                patterns.push_back(newPattern);
                pattern_set.insert(newPattern);
            }
        }
    }
    else
    {
        // Normal case: generate correlated patterns from a base pattern
        std::vector<bool> basePattern = generateRandomPattern(networkSize);

        while (patterns.size() < nbPattern)
        {
            std::vector<bool> newPattern = randomizePattern(basePattern, nb_random_bits);
            // O(1) average case lookup instead of O(K) linear search
            if (pattern_set.find(newPattern) == pattern_set.end())
            {
                patterns.push_back(newPattern);
                pattern_set.insert(newPattern);
            }
        }
    }

    return patterns;
}

void show_vector(std::vector<double> vector)
{
    for (const auto &element : vector)
    {
        std::cout << element << " ";
    }
    std::cout << "" << std::endl;
}

double ratio_diff_vectors(const std::vector<double> &state1, const std::vector<double> &state2)
{
    double sum_diff = 0.0;
    for (size_t k = 0; k < state1.size(); k++)
    {
        sum_diff += abs(state1[k] - state2[k]);
    }
    return sum_diff / (state1.size()*2);
}

std::vector<std::unordered_map<std::string, double>> generateCombinations(const std::unordered_map<std::string, std::vector<double>> &varying_params)
{
    std::vector<std::unordered_map<std::string, double>> combinations;

    // Calculate the total number of combinations
    size_t total_combinations = 1;
    for (const auto &param : varying_params)
    {
        total_combinations *= param.second.size();
    }

    // Generate all combinations
    for (size_t i = 0; i < total_combinations; ++i)
    {
        std::unordered_map<std::string, double> combination;
        size_t index = i;
        for (const auto &param : varying_params)
        {
            combination[param.first] = param.second[index % param.second.size()];
            index /= param.second.size();
        }
        combinations.push_back(combination);
    }

    return combinations;
}

std::vector<double> linspace(double start, double end, int num)
{
    std::vector<double> result;
    if (num <= 0)
    {
        return result;
    }
    if (num == 1)
    {
        result.push_back(start);
        return result;
    }

    double step = (end - start) / (num - 1);
    for (int i = 0; i < num; ++i)
    {
        result.push_back(start + i * step);
    }
    return result;
}

void collectSimulationDataSeries(const std::string &folderResultsPath)
{
    std::vector<std::unordered_map<std::string, std::string>> allSimData;
    std::vector<std::string> allKeys;
    std::vector<std::string> resultKeys;
    std::string path_name;
    int first_sim_visited = true;
    allKeys.push_back("sim_ID");

    // Iterate through all subdirectories
    for (const auto &entry : fs::directory_iterator(folderResultsPath))
    {
        if (fs::is_directory(entry))
        {
            std::unordered_map<std::string, std::string> simData;
            path_name=entry.path().filename().string();
            std::regex regex_pattern(R"(\d+$)");
            std::smatch match;
            std::string sim_id;
            if(std::regex_search(path_name,match,regex_pattern)){
                sim_id = match.str();
                std::cout <<"Extracted Sim ID "<< sim_id << std::endl;
            }else{
                std::cout << "No SIM ID found" << std::endl;
            }
            simData["sim_ID"] = sim_id;
            // Read parameters file
            std::ifstream paramFile(entry.path() / "parameters.data");
            if (paramFile.is_open())
            {
                std::string line;
                while (std::getline(paramFile, line))
                {
                    std::istringstream iss(line);
                    std::string key, value;
                    if (std::getline(iss, key, '=') && std::getline(iss, value))
                    {
                        simData[key] = value;
                        if (first_sim_visited)
                        {
                            allKeys.push_back(key);
                        }
                        
                    }
                }
                paramFile.close();
            }

            // Read results file
            std::ifstream resultFile(entry.path() / "results.data");
            if (resultFile.is_open())
            {
                std::string line;
                std::getline(resultFile, line);
                if(first_sim_visited){
                    std::istringstream iss(line);
                    std::string key;
                    while(std::getline(iss, key, ','))
                    {
                        allKeys.push_back(key);
                        resultKeys.push_back(key);
                    }
                    first_sim_visited = false;
                } 
                while (std::getline(resultFile, line))
                {
                    int nb_elements=0;
                    std::istringstream iss(line);
                    std::string value;
                    while(std::getline(iss, value, ','))
                    {
                        simData[resultKeys[nb_elements]] = value;
                        nb_elements++;
                    }
                    allSimData.push_back(simData);
                }
                resultFile.close();
            }

        }
    }

    // Write all data to a single CSV file
    std::ofstream csvFile(folderResultsPath + "/all_simulation_data.csv");
    if (csvFile.is_open())
    {
        // Write header
        for (const auto &key : allKeys)
        {
            csvFile << key << ",";
        }
        csvFile << "\n";

        // Write data
        for (const auto &simData : allSimData)
        {
            for (const auto &key : allKeys)
            {
                auto it = simData.find(key);
                if (it != simData.end())
                {
                    csvFile << it->second;
                }
                csvFile << ",";
            }
            csvFile << "\n";
        }
        csvFile.close();
        std::cout << "All simulation data has been written to all_simulation_data.csv" << std::endl;
    }
    else
    {
        std::cerr << "Unable to open file for writing CSV data." << std::endl;
    }
}

void collectSimulationData(const std::string &folderResultsPath)
{
    std::vector<std::unordered_map<std::string, std::string>> allSimData;
    std::unordered_set<std::string> allKeys;

    // Iterate through all subdirectories
    for (const auto &entry : fs::directory_iterator(folderResultsPath))
    {
        if (fs::is_directory(entry))
        {
            std::unordered_map<std::string, std::string> simData;

            // Read parameters file
            std::ifstream paramFile(entry.path() / "parameters.data");
            if (paramFile.is_open())
            {
                std::string line;
                while (std::getline(paramFile, line))
                {
                    std::istringstream iss(line);
                    std::string key, value;
                    if (std::getline(iss, key, '=') && std::getline(iss, value))
                    {
                        simData[key] = value;
                        allKeys.insert(key);
                    }
                }
                paramFile.close();
            }

            // Read results file
            std::ifstream resultFile(entry.path() / "results.data");
            if (resultFile.is_open())
            {
                std::string line;
                while (std::getline(resultFile, line))
                {
                    std::istringstream iss(line);
                    std::string key, value;
                    if (std::getline(iss, key, '=') && std::getline(iss, value))
                    {
                        simData[key] = value;
                        allKeys.insert(key);
                    }
                }
                resultFile.close();
            }

            allSimData.push_back(simData);
        }
    }

    // Write all data to a single CSV file
    std::ofstream csvFile(folderResultsPath + "/all_simulation_data.csv");
    if (csvFile.is_open())
    {
        // Write header
        std::vector<std::string> sortedKeys(allKeys.begin(), allKeys.end());
        std::sort(sortedKeys.begin(), sortedKeys.end());
        for (const auto &key : sortedKeys)
        {
            csvFile << key << ",";
        }
        csvFile << "\n";

        // Write data
        for (const auto &simData : allSimData)
        {
            for (const auto &key : sortedKeys)
            {
                auto it = simData.find(key);
                if (it != simData.end())
                {
                    csvFile << it->second;
                }
                csvFile << ",";
            }
            csvFile << "\n";
        }
        csvFile.close();
        std::cout << "All simulation data has been written to all_simulation_data.csv" << std::endl;
    }
    else
    {
        std::cerr << "Unable to open file for writing CSV data." << std::endl;
    }
}

void createParameterFile(const std::string &directory, const std::unordered_map<std::string, double> &parameters)
{
    // Create the directory if it doesn't exist
    std::filesystem::create_directories(directory);

    // Create the full path to the file
    std::string filePath = directory + "/parameters.data";

    // Open a file stream to write the parameters
    std::ofstream outFile(filePath);

    // Check if the file was opened successfully
    if (!outFile)
    {
        std::cerr << "Error: Could not create file " << filePath << std::endl;
        return;
    }

    // Write each parameter to the file
    for (const auto &param : parameters)
    {
        outFile << param.first << "=" << param.second << "\n";
    }

    // Close the file stream
    outFile.close();
}

double computeCorrelation(const std::vector<bool>& original, const std::vector<bool>& noisy) {
    int n = original.size();
    int sum_original = accumulate(original.begin(), original.end(), 0);
    int sum_noisy = accumulate(noisy.begin(), noisy.end(), 0);
    int sum_product = inner_product(original.begin(), original.end(), noisy.begin(), 0);
    
    double mean_original = static_cast<double>(sum_original) / n;
    double mean_noisy = static_cast<double>(sum_noisy) / n;
    
    double numerator = sum_product - n * mean_original * mean_noisy;
    double denominator = sqrt((sum_original - n * mean_original * mean_original) * (sum_noisy - n * mean_noisy * mean_noisy));
    
    return denominator == 0 ? 0 : numerator / denominator;
}

void lunchParalSim(std::string foldername_results, std::unordered_map<std::string, std::vector<double>> varying_params, void (*run_simulation)(int, const std::unordered_map<std::string, double>&, const std::string&))
{
    std::vector<std::unordered_map<std::string, double>> combinations = generateCombinations(varying_params);
    std::vector<std::thread> threads;

    for (int sim_number = 0; sim_number < combinations.size(); ++sim_number)
    {
        threads.emplace_back(run_simulation, sim_number, combinations[sim_number], foldername_results);
    }

    for (auto &t : threads)
    {
        t.join();
    }
}

void lunchParalSimThreadLimit(int nb_thread_max, std::string foldername_results, std::unordered_map<std::string, std::vector<double>> varying_params, void (*run_simulation)(int, const std::unordered_map<std::string, double>&, const std::string&))
{
    int active_threads = 0;
    int completed_simulations = 0;
    std::mutex mtx;
    std::condition_variable cv;
    std::vector<std::unordered_map<std::string, double>> combinations = generateCombinations(varying_params);
    std::vector<std::thread> threads;

    int total_simulations = combinations.size();
    std::cout << "Starting " << total_simulations << " simulations with " << nb_thread_max << " parallel threads..." << std::endl;

    for (int sim_number = 0; sim_number < combinations.size(); ++sim_number)
    {

        {
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait(lock, [&]
                    { return active_threads < nb_thread_max; });
            ++active_threads;
        }

        threads.emplace_back([=, &mtx, &cv, &active_threads, &completed_simulations, total_simulations]
                                {
            run_simulation(sim_number, combinations[sim_number],foldername_results);
            {
                std::lock_guard<std::mutex> lock(mtx);
                --active_threads;
                ++completed_simulations;

                // Print progress every 100 simulations or at specific milestones
                if (completed_simulations % 100 == 0 ||
                    completed_simulations == total_simulations ||
                    completed_simulations % (total_simulations / 10) == 0)
                {
                    double progress = (100.0 * completed_simulations) / total_simulations;
                    std::cout << "Progress: " << completed_simulations << "/" << total_simulations
                              << " (" << std::fixed << std::setprecision(1) << progress << "%)" << std::endl;
                }
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

    std::cout << "All " << total_simulations << " simulations completed!" << std::endl;
}


std::vector<double> generateEvenlySpacedIntegers(int a, int b, int n)
{
    std::vector<double> result;

    if (n <= 0)
    {
        return result;
    }

    if (n == 1)
    {
        result.push_back(a);
        return result;
    }

    double step = static_cast<double>(b - a) / (n - 1);

    for (int i = 0; i < n; ++i)
    {
        int value = static_cast<int>(std::round(a + i * step));
        result.push_back(value);
    }

    // Ensure the last element is exactly b
    if (!result.empty())
    {
        result.back() = b;
    }

    return result;
}