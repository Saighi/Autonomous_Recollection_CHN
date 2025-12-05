#include "network.hpp"
#include "utils.hpp"
#include <numeric>
#include <iostream>
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
#include <sstream>
#include <unordered_map>
#include <cerrno>
#include <cstring>
#include <regex>
#include <thread>
#include <mutex>
#include <condition_variable>

namespace fs = std::filesystem;

// ============================================================================
// CSV/File I/O Functions (overloads)
// ============================================================================

void writeToCSV(std::ofstream& file, const std::vector<double>& data)
{
    for (size_t i = 0; i < data.size(); ++i)
    {
        file << data[i];
        if (i != data.size() - 1)
        {
            file << " ";
        }
    }
    file << "\n";
}

void writeToCSV(std::ostream* file, const std::vector<double>& data)
{
    for (size_t i = 0; i < data.size(); ++i)
    {
        *file << data[i];
        if (i != data.size() - 1)
        {
            *file << " ";
        }
    }
    *file << "\n";
}

void writeBoolToCSV(std::ofstream& file, const std::vector<bool>& data)
{
    for (size_t i = 0; i < data.size(); ++i)
    {
        file << data[i];
        if (i != data.size() - 1)
        {
            file << " ";
        }
    }
    file << "\n";
}

void writeBoolToCSV(std::ostream& file, const std::vector<bool>& data)
{
    for (size_t i = 0; i < data.size(); ++i)
    {
        file << data[i];
        if (i != data.size() - 1)
        {
            file << " ";
        }
    }
    file << "\n";
}

// ============================================================================
// Visualization Functions
// ============================================================================

void show_state(Network& net)
{
    std::cout << "activity :" << std::endl;
    for (const auto& element : net.activity_list)
    {
        std::cout << element << " ";
    }
    std::cout << "" << std::endl;
    std::cout << "rates :" << std::endl;
    for (const auto& element : net.rate_list)
    {
        std::cout << element << " ";
    }
    std::cout << "" << std::endl;
}

void show_state_grid(Network& net, int rows)
{
    int iter = 0;
    std::cout << "activity :" << std::endl;
    for (const auto& element : net.activity_list)
    {
        if ((iter % (net.size / rows) == 0) && iter != 0)
        {
            std::cout << "" << std::endl;
        }
        std::cout << element << " ";
        iter++;
    }
    iter = 0;
    std::cout << "" << std::endl;
    std::cout << "rates :" << std::endl;
    for (const auto& element : net.rate_list)
    {
        if ((iter % (net.size / rows) == 0.0) && iter != 0)
        {
            std::cout << "" << std::endl;
        }
        std::cout << element << " ";
        iter++;
    }
    std::cout << "" << std::endl;
}

void show_matrix(std::vector<std::vector<double>> matrix)
{
    for (const auto& row : matrix)
    {
        for (const auto& element : row)
        {
            std::cout << element << " ";
        }
        std::cout << "" << std::endl;
    }
}

void show_vector(std::vector<double> vector)
{
    for (const auto& element : vector)
    {
        std::cout << element << " ";
    }
    std::cout << "" << std::endl;
}

void show_vector_bool_grid(std::vector<bool> vec, int rows)
{
    int iter = 0;
    std::cout << "rates :" << std::endl;
    for (const auto& element : vec)
    {
        if ((iter % (vec.size() / rows) == 0.0) && iter != 0)
        {
            std::cout << "" << std::endl;
        }
        std::cout << element << " ";
        iter++;
    }
    std::cout << "" << std::endl;
}

// ============================================================================
// Simulation Runner Functions
// ============================================================================

void run_net_sim(Network& net, int nb_iter, double delta)
{
    for (int i = 0; i < nb_iter; i++)
    {
        net.iterate(delta);
    }
}

void run_net_sim_query_drive(Network& net, std::vector<double>& query_drives, double strength_drive, int nb_iter, double delta)
{
    for (int i = 0; i < nb_iter; i++)
    {
        net.iterate_query_drive(delta, strength_drive, query_drives);
    }
}

void run_net_sim_noisy(Network& net, int nb_iter, double delta, double mean, double stddev)
{
    for (int i = 0; i < nb_iter; i++)
    {
        net.noisy_iterate(delta, mean, stddev);
    }
}

void run_net_sim_noisy_depressed(Network& net, int nb_iter, double delta, double mean, double stddev)
{
    for (int i = 0; i < nb_iter; i++)
    {
        net.noisy_depressed_iterate(delta, mean, stddev);
    }
}

void run_net_sim_noisy_depressed_save(Network& net, int nb_iter, double delta, double mean, double stddev, std::ofstream& file)
{
    for (int i = 0; i < nb_iter; i++)
    {
        net.noisy_depressed_iterate(delta, mean, stddev);
        writeToCSV(file, net.rate_list);
    }
}

int run_net_sim_choice(Network& net, SimulationConfig& conf)
{
    int nb_iter = 0;
    std::vector<double> rates_past(net.size, 1000.0);
    std::vector<double> rates_new(net.size, 0.0);
    std::vector<double> differences(net.size, 1000.0);
    double max = 1000.0;
    while (max > conf.epsilon && nb_iter <= conf.max_iter)
    {
        if (conf.save)
        {
            writeToCSV(conf.output, net.rate_list);
        }
        if (conf.depressed)
        {
            net.noisy_depressed_iterate(conf.delta, conf.mean, conf.stddev);
        }
        else
        {
            net.noisy_iterate(conf.delta, conf.mean, conf.stddev);
        }
        rates_past = rates_new;
        rates_new = net.rate_list;
        std::transform(rates_past.begin(), rates_past.end(), rates_new.begin(), differences.begin(), std::minus<>());
        max = std::abs(*std::max_element(differences.begin(), differences.end()));
        nb_iter += 1;
    }
    return nb_iter;
}

// ============================================================================
// Winner-Take-All Functions
// ============================================================================

struct Compare
{
    bool operator()(const std::pair<double, int>& a, const std::pair<double, int>& b)
    {
        return a.first > b.first;
    }
};

std::vector<double> assignStateToTopNValues(std::vector<double>& vec, int n, double winner_state, double loser_state)
{
    std::priority_queue<std::pair<double, int>, std::vector<std::pair<double, int>>, Compare> pq;

    for (size_t i = 0; i < vec.size(); ++i)
    {
        pq.push(std::make_pair(vec[i], i));
        if (pq.size() > static_cast<size_t>(n))
        {
            pq.pop();
        }
    }

    std::vector<double> state_vector(vec.size(), loser_state);

    while (!pq.empty())
    {
        state_vector[pq.top().second] = winner_state;
        pq.pop();
    }

    return state_vector;
}

std::vector<bool> assignBoolToTopNValues(std::vector<double>& vec, int n)
{
    std::priority_queue<std::pair<double, int>, std::vector<std::pair<double, int>>, Compare> pq;

    for (size_t i = 0; i < vec.size(); ++i)
    {
        pq.push(std::make_pair(vec[i], i));
        if (pq.size() > static_cast<size_t>(n))
        {
            pq.pop();
        }
    }

    std::vector<bool> bool_vector(vec.size(), false);

    while (!pq.empty())
    {
        bool_vector[pq.top().second] = true;
        pq.pop();
    }

    return bool_vector;
}

// ============================================================================
// Math Utilities
// ============================================================================

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

    if (!result.empty())
    {
        result.back() = b;
    }

    return result;
}

// ============================================================================
// Pattern Generation and Loading
// ============================================================================

std::vector<bool> generateBasePattern(int N, int nb_winning_units)
{
    std::vector<bool> basePattern(N, false);
    for (int i = 0; i < nb_winning_units; ++i)
    {
        basePattern[i] = true;
    }
    return basePattern;
}

std::vector<bool> generateNoisyBalancedPattern(const std::vector<bool>& basePattern, int numFlips)
{
    std::vector<bool> noisyPattern = basePattern;
    int N = basePattern.size();
    int index;
    int cpt = 0;
    while (cpt < numFlips)
    {
        index = rand() % N;
        if (noisyPattern[index] == true)
        {
            noisyPattern[index] = !noisyPattern[index];
            index = rand() % N;
            while (noisyPattern[index] != false)
            {
                index = rand() % N;
            }
            noisyPattern[index] = !noisyPattern[index];
            cpt += 1;
        }
    }
    return noisyPattern;
}

bool patternExists(const std::vector<std::vector<bool>>& patterns, const std::vector<bool>& pattern)
{
    for (const auto& p : patterns)
    {
        if (p == pattern)
        {
            return true;
        }
    }
    return false;
}

std::vector<std::vector<bool>> generatePatterns(int K, int N, double sparsity, double rho)
{
    // sparsity = fraction of active units (0 to 1)
    // rho = pattern correlation: 1 = identical patterns, 0 = maximally different
    std::vector<std::vector<bool>> patterns;
    int nb_winning_units = std::max(1, static_cast<int>(sparsity * N));
    std::vector<bool> basePattern = generateBasePattern(N, nb_winning_units);
    int numFlips = static_cast<int>((1.0 - rho) * nb_winning_units);

    while (patterns.size() < static_cast<size_t>(K))
    {
        std::vector<bool> newPattern = generateNoisyBalancedPattern(basePattern, numFlips);
        if (!patternExists(patterns, newPattern))
        {
            patterns.push_back(newPattern);
        }
    }

    return patterns;
}

std::vector<std::vector<bool>> loadPatterns(const std::string& filename)
{
    std::ifstream file(filename);
    std::string line;
    std::vector<std::vector<bool>> boolVectors;

    if (file.is_open())
    {
        while (getline(file, line))
        {
            std::istringstream iss(line);
            std::vector<bool> boolVector;
            std::string value;

            while (iss >> value)
            {
                boolVector.push_back(value == "1");
            }
            boolVectors.push_back(boolVector);
        }
        file.close();
    }
    else
    {
        std::cerr << "Unable to open file" << std::endl;
    }

    return boolVectors;
}

bool areVectorsEqual(const std::vector<bool>& v1, const std::vector<bool>& v2)
{
    return v1 == v2;
}

bool comparestates(const std::vector<bool>& state1, const std::vector<bool>& state2)
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

// ============================================================================
// Pattern to State Conversion
// ============================================================================

std::vector<double> pattern_as_states(double up_rate, double down_rate, std::vector<bool> bin_pattern)
{
    std::vector<double> state_input(bin_pattern.size());
    for (size_t j = 0; j < state_input.size(); j++)
    {
        if (bin_pattern[j])
        {
            state_input[j] = up_rate;
        }
        else
        {
            state_input[j] = down_rate;
        }
    }
    return state_input;
}

std::vector<std::vector<double>> patterns_as_states(double up_rate, double down_rate, std::vector<std::vector<bool>> bin_patterns)
{
    std::vector<std::vector<double>> initial_patterns_state_list(bin_patterns.size());
    for (size_t i = 0; i < bin_patterns.size(); i++)
    {
        initial_patterns_state_list[i] = pattern_as_states(up_rate, down_rate, bin_patterns[i]);
    }
    return initial_patterns_state_list;
}

std::vector<double> pattern_as_states_with_distance_noise(double drive_target, std::vector<bool> bin_pattern, double distance_noise_level, Network& net)
{
    std::vector<double> state_input(bin_pattern.size());
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-distance_noise_level * drive_target, distance_noise_level * drive_target);

    for (size_t j = 0; j < state_input.size(); j++)
    {
        double noisy_drive;
        if (bin_pattern[j])
        {
            noisy_drive = drive_target + dis(gen);
        }
        else
        {
            noisy_drive = -drive_target + dis(gen);
        }
        state_input[j] = noisy_drive;
    }
    return state_input;
}

std::vector<std::vector<double>> patterns_as_states_with_distance_noise(double drive_target, std::vector<std::vector<bool>> bin_patterns, double distance_noise_level, Network& net)
{
    std::vector<std::vector<double>> initial_patterns_state_list(bin_patterns.size());
    for (size_t i = 0; i < bin_patterns.size(); i++)
    {
        initial_patterns_state_list[i] = pattern_as_states_with_distance_noise(drive_target, bin_patterns[i], distance_noise_level, net);
    }
    return initial_patterns_state_list;
}

std::vector<double> setToValueRandomElements(const std::vector<double>& baseValues, int numFlips, double value)
{
    std::vector<double> newVector = baseValues;
    int N = newVector.size();
    int index;
    int cpt = 0;
    while (cpt < numFlips)
    {
        index = rand() % N;
        newVector[index] = value;
        cpt += 1;
    }
    return newVector;
}

// ============================================================================
// Parameter Handling
// ============================================================================

void createParameterFile(const std::string& directory, const std::unordered_map<std::string, double>& parameters)
{
    std::filesystem::create_directories(directory);
    std::string filePath = directory + "/parameters.data";
    std::ofstream outFile(filePath);

    if (!outFile)
    {
        std::cerr << "Error: Could not create file " << filePath << std::endl;
        return;
    }

    for (const auto& param : parameters)
    {
        outFile << param.first << "=" << param.second << "\n";
    }

    outFile.close();
}

std::unordered_map<std::string, double> readParametersFile(const std::string& filePath)
{
    std::unordered_map<std::string, double> parameters;
    std::ifstream file(filePath);

    if (file.is_open())
    {
        std::string line;
        while (std::getline(file, line))
        {
            std::istringstream lineStream(line);
            std::string key;
            if (std::getline(lineStream, key, '='))
            {
                std::string valueStr;
                if (std::getline(lineStream, valueStr))
                {
                    try
                    {
                        double value = std::stod(valueStr);
                        parameters[key] = value;
                    }
                    catch (const std::invalid_argument& e)
                    {
                        std::cerr << "Invalid value for key: " << key << std::endl;
                    }
                    catch (const std::out_of_range& e)
                    {
                        std::cerr << "Value out of range for key: " << key << std::endl;
                    }
                }
            }
        }
        file.close();
    }
    else
    {
        std::cerr << "Unable to open file: " << filePath << std::endl;
    }

    return parameters;
}

std::unordered_map<std::string, double> fuseMaps(std::unordered_map<std::string, double> map1, std::unordered_map<std::string, double> map2)
{
    for (const auto& element : map2)
    {
        map1[element.first] = element.second;
    }
    return map1;
}

std::vector<std::unordered_map<std::string, double>> generateCombinations(const std::unordered_map<std::string, std::vector<double>>& varying_params)
{
    std::vector<std::unordered_map<std::string, double>> combinations;

    size_t total_combinations = 1;
    for (const auto& param : varying_params)
    {
        total_combinations *= param.second.size();
    }

    for (size_t i = 0; i < total_combinations; ++i)
    {
        std::unordered_map<std::string, double> combination;
        size_t index = i;
        for (const auto& param : varying_params)
        {
            combination[param.first] = param.second[index % param.second.size()];
            index /= param.second.size();
        }
        combinations.push_back(combination);
    }

    return combinations;
}

// ============================================================================
// Matrix I/O
// ============================================================================

void writeMatrixToFile(const std::vector<std::vector<double>>& matrix, const std::string& filePath)
{
    std::ofstream outFile(filePath);

    if (!outFile.is_open())
    {
        std::cerr << "Error opening file for writing: " << filePath << std::endl;
        return;
    }

    for (const auto& row : matrix)
    {
        for (const auto& element : row)
        {
            outFile << element << " ";
        }
        outFile << "\n";
    }

    outFile.close();
}

void writeBoolMatrixToFile(const std::vector<std::vector<bool>>& matrix, const std::string& filePath)
{
    std::ofstream outFile(filePath);

    if (!outFile.is_open())
    {
        std::cerr << "Error opening file for writing: " << filePath << std::endl;
        return;
    }

    for (const auto& row : matrix)
    {
        for (const auto& element : row)
        {
            outFile << element << " ";
        }
        outFile << "\n";
    }

    outFile.close();
}

std::vector<std::vector<double>> readMatrixFromFile(const std::string& filePath)
{
    std::vector<std::vector<double>> matrix;
    std::ifstream inFile(filePath);
    if (!inFile)
    {
        std::cerr << "Error: Unable to open file '" << filePath << "'." << std::endl;
        std::cerr << "Error code: " << errno << " (" << strerror(errno) << ")" << std::endl;

        std::ofstream test(filePath, std::ios::in);
        if (test.is_open())
        {
            std::cerr << "File exists but cannot be opened for reading. Check permissions." << std::endl;
            test.close();
        }
        else
        {
            std::cerr << "File does not exist or path is incorrect." << std::endl;
        }

        return matrix;
    }

    std::string line;
    while (std::getline(inFile, line))
    {
        std::istringstream ss(line);
        std::vector<double> row;
        double value;

        while (ss >> value)
        {
            row.push_back(value);
        }

        matrix.push_back(row);
    }

    inFile.close();
    return matrix;
}

std::vector<std::vector<bool>> readBoolMatrixFromFile(const std::string& filePath)
{
    std::vector<std::vector<bool>> matrix;
    std::ifstream inFile(filePath);

    if (!inFile.is_open())
    {
        std::cerr << "Error opening file for reading: " << filePath << std::endl;
        return matrix;
    }

    std::string line;
    while (std::getline(inFile, line))
    {
        std::istringstream ss(line);
        std::vector<bool> row;
        double value;

        while (ss >> value)
        {
            row.push_back(value);
        }

        matrix.push_back(row);
    }

    inFile.close();
    return matrix;
}

// ============================================================================
// Parallel Simulation Launcher
// ============================================================================

void lunchParalSim(std::string foldername_results, std::unordered_map<std::string, std::vector<double>> varying_params, void (*run_simulation)(int, std::unordered_map<std::string, double>, const std::string))
{
    std::vector<std::unordered_map<std::string, double>> combinations = generateCombinations(varying_params);
    std::vector<std::thread> threads;
    for (size_t sim_number = 0; sim_number < combinations.size(); ++sim_number)
    {
        threads.emplace_back(run_simulation, sim_number, combinations[sim_number], foldername_results);
    }

    for (auto& t : threads)
    {
        t.join();
    }
}

// ============================================================================
// Data Collection/Aggregation
// ============================================================================

void collectSimulationData(const std::string& folderResultsPath)
{
    std::vector<std::unordered_map<std::string, std::string>> allSimData;
    std::unordered_set<std::string> allKeys;

    for (const auto& entry : fs::directory_iterator(folderResultsPath))
    {
        if (fs::is_directory(entry))
        {
            std::unordered_map<std::string, std::string> simData;

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

    std::ofstream csvFile(folderResultsPath + "/all_simulation_data.csv");
    if (csvFile.is_open())
    {
        std::vector<std::string> sortedKeys(allKeys.begin(), allKeys.end());
        std::sort(sortedKeys.begin(), sortedKeys.end());
        for (const auto& key : sortedKeys)
        {
            csvFile << key << ",";
        }
        csvFile << "\n";

        for (const auto& simData : allSimData)
        {
            for (const auto& key : sortedKeys)
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

void collectSimulationDataSeries(const std::string& folderResultsPath)
{
    std::vector<std::unordered_map<std::string, std::string>> allSimData;
    std::vector<std::string> allKeys;
    std::vector<std::string> resultKeys;
    std::string path_name;
    int first_sim_visited = true;
    allKeys.push_back("sim_ID");

    for (const auto& entry : fs::directory_iterator(folderResultsPath))
    {
        if (fs::is_directory(entry))
        {
            std::unordered_map<std::string, std::string> simData;
            path_name = entry.path().filename().string();
            std::regex regex_pattern(R"(\d+$)");
            std::smatch match;
            std::string sim_id;
            if (std::regex_search(path_name, match, regex_pattern))
            {
                sim_id = match.str();
                std::cout << "Extracted Sim ID " << sim_id << std::endl;
            }
            else
            {
                std::cout << "No SIM ID found" << std::endl;
            }
            simData["sim_ID"] = sim_id;

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

            std::ifstream resultFile(entry.path() / "results.data");
            if (resultFile.is_open())
            {
                std::string line;
                std::getline(resultFile, line);
                if (first_sim_visited)
                {
                    std::istringstream iss(line);
                    std::string key;
                    while (std::getline(iss, key, ','))
                    {
                        allKeys.push_back(key);
                        resultKeys.push_back(key);
                    }
                    first_sim_visited = false;
                }
                while (std::getline(resultFile, line))
                {
                    int nb_elements = 0;
                    std::istringstream iss(line);
                    std::string value;
                    while (std::getline(iss, value, ','))
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

    std::ofstream csvFile(folderResultsPath + "/all_simulation_data.csv");
    if (csvFile.is_open())
    {
        for (const auto& key : allKeys)
        {
            csvFile << key << ",";
        }
        csvFile << "\n";

        for (const auto& simData : allSimData)
        {
            for (const auto& key : allKeys)
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
