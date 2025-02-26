#include "utils.hpp"

#include <string.h>

#include <algorithm>
#include <cerrno>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <numeric>
#include <queue>
#include <random>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "network.hpp"
namespace fs = std::filesystem;

// Matrix shape manipulation
std::vector<std::vector<double>> flattenFirstDim3D(
    const std::vector<std::vector<std::vector<double>>> &input) {
    if (input.empty())
        return {};

    // Calculate total size needed for first dimension
    size_t totalSize = 0;
    for (const auto &matrix : input) {
        totalSize += matrix.size();
    }

    // Pre-allocate the result vector for efficiency
    std::vector<std::vector<double>> flattened;
    flattened.reserve(totalSize);

    // Copy all elements
    for (const auto &matrix : input) {
        flattened.insert(flattened.end(), matrix.begin(), matrix.end());
    }

    return flattened;
}

std::vector<double> flattenFirstDim2D(
    const std::vector<std::vector<double>> &input) {
    if (input.empty())
        return {};

    // Calculate total size needed for first dimension
    size_t totalSize = 0;
    for (const auto &matrix : input) {
        totalSize += matrix.size();
    }

    // Pre-allocate the result vector for efficiency
    std::vector<double> flattened;
    flattened.reserve(totalSize);

    // Copy all elements
    for (const auto &matrix : input) {
        flattened.insert(flattened.end(), matrix.begin(), matrix.end());
    }

    return flattened;
}

std::vector<std::vector<double>> transposeMatrix(
    const std::vector<std::vector<double>> &matrix) {
    if (matrix.empty())
        return {};

    int rows = matrix.size();
    int cols = matrix[0].size();

    // Create the transposed matrix with swapped dimensions
    std::vector<std::vector<double>> transposed(cols,
                                                std::vector<double>(rows));

    // Fill the transposed matrix
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            transposed[j][i] = matrix[i][j];
        }
    }

    return transposed;
}

void writeToCSV(std::ostream *file, const std::vector<double> &data) {
    for (size_t i = 0; i < data.size(); ++i) {
        *file << data[i];
        if (i != data.size() - 1) {
            *file << " ";
        }
    }
    *file << "\n";  // Add a new line after each vector
}

void writeBoolToCSV(std::ostream &file, const std::vector<bool> &data) {
    for (size_t i = 0; i < data.size(); ++i) {
        file << data[i];
        if (i != data.size() - 1) {
            file << " ";
        }
    }
    file << "\n";  // Add a new line after each vector
}

// VIZUALISATION

void show_state(Network &net) {
    std::cout << "activity :" << std::endl;
    for (const auto &element : net.activity_list) {
        std::cout << element << " ";
    }
    std::cout << "" << std::endl;
    std::cout << "rates :" << std::endl;
    for (const auto &element : net.rate_list) {
        std::cout << element << " ";
    }
    std::cout << "" << std::endl;
}

void show_state_grid(Network &net, int rows) {
    int iter = 0;
    std::cout << "activity :" << std::endl;
    for (const auto &element : net.activity_list) {
        if ((iter % (net.size / rows) == 0) && iter != 0) {
            std::cout << "" << std::endl;
        }

        std::cout << element << " ";
        iter++;
    }
    iter = 0;
    std::cout << "" << std::endl;
    std::cout << "rates :" << std::endl;
    for (const auto &element : net.rate_list) {
        if ((iter % (net.size / rows) == 0.0) && iter != 0) {
            std::cout << "" << std::endl;
        }
        std::cout << element << " ";
        iter++;
    }
    std::cout << "" << std::endl;
}

void show_matrix(std::vector<std::vector<double>> matrix) {
    for (const auto &row : matrix) {
        for (const auto &element : row) {
            std::cout << element << " ";
        }
        std::cout << "" << std::endl;
    }
}

void show_vector(std::vector<double> vector) {
    for (const auto &element : vector) {
        std::cout << element << " ";
    }
    std::cout << "" << std::endl;
}

// SIMULATION

int run_net_sim_choice(Network &net, SimulationConfig &conf) {
    int nb_iter = 0;
    std::vector<double> rates_past(net.size, 1000.0);
    std::vector<double> rates_new(net.size, 0.0);
    std::vector<double> differences(net.size, 1000.0);
    double max = 1000.0;
    while (max > conf.epsilon && nb_iter <= conf.max_iter) {
        if (conf.save) {
            writeToCSV(conf.output, net.rate_list);
        }
        if (conf.depressed) {
            if (conf.noise) {
                net.noisy_depressed_iterate(conf.delta, conf.mean, conf.stddev);
            } else {
                net.depressed_iterate(conf.delta);
            }
        } else {
            if (conf.noise) {
                net.noisy_iterate(conf.delta, conf.mean, conf.stddev);
            } else {
                net.iterate(conf.delta);
            }
        }
        rates_past = rates_new;
        rates_new = net.rate_list;
        std::transform(rates_past.begin(), rates_past.end(), rates_new.begin(),
                       differences.begin(), std::minus<>());
        max =
            std::abs(*std::max_element(differences.begin(), differences.end()));
        nb_iter += 1;
    }
    return nb_iter;
}

// TOOLS

// Comparator for priority queue
struct Compare {
    bool operator()(const std::pair<double, int> &a,
                    const std::pair<double, int> &b) {
        return a.first > b.first;  // Min Heap based on the value of the double
    }
};

std::vector<double> assignStateToTopNValues(std::vector<double> &vec, int n,
                                            double winner_state,
                                            double loser_state) {
    std::priority_queue<std::pair<double, int>,
                        std::vector<std::pair<double, int>>, Compare>
        pq;

    // Iterate through the vector and maintain a heap of top 'n' elements
    for (int i = 0; i < vec.size(); ++i) {
        pq.push(std::make_pair(vec[i], i));
        // If the heap size exceeds 'n', pop the smallest element out
        if (pq.size() > n) {
            pq.pop();
        }
    }
    // displayPriorityQueue(pq);
    //  Create a new vector to store the final states
    std::vector<double> state_vector(vec.size(), loser_state);
    std::vector<bool> bool_vector(vec.size(), false);

    // Extract the indexes from the heap and set 'winner_state' for top 'n'
    // elements
    while (!pq.empty()) {
        state_vector[pq.top().second] = winner_state;
        bool_vector[pq.top().second] = true;
        pq.pop();
    }

    return state_vector;
}

std::vector<bool> assignBoolToTopNValues(std::vector<double> &vec, int n) {
    std::priority_queue<std::pair<double, int>,
                        std::vector<std::pair<double, int>>, Compare>
        pq;

    // Iterate through the vector and maintain a heap of top 'n' elements
    for (int i = 0; i < vec.size(); ++i) {
        pq.push(std::make_pair(vec[i], i));
        // If the heap size exceeds 'n', pop the smallest element out
        if (pq.size() > n) {
            pq.pop();
        }
    }
    // displayPriorityQueue(pq);
    //  Create a new vector to store the final states
    std::vector<bool> bool_vector(vec.size(), false);

    // Extract the indexes from the heap and set 'winner_state' for top 'n'
    // elements
    while (!pq.empty()) {
        bool_vector[pq.top().second] = true;
        pq.pop();
    }

    return bool_vector;
}

void show_vector_bool_grid(std::vector<bool> vec, int rows) {
    int iter = 0;
    // std::cout << "activity :" << std::endl;
    // for (const auto &element : net.activity_list)
    // {
    //     if ((iter % (net.size / rows) == 0) && iter != 0)
    //     {
    //         std::cout << "" << std::endl;
    //     }

    //     std::cout << std::fixed << std::setprecision(2) << element << " ";
    //     iter ++;
    // }
    // iter = 0;
    // std::cout << "" << std::endl;
    std::cout << "rates :" << std::endl;
    for (const auto &element : vec) {
        if ((iter % (vec.size() / rows) == 0.0) && iter != 0) {
            std::cout << "" << std::endl;
        }
        // std::cout << element << " ";
        std::cout << element << " ";
        iter++;
    }
    std::cout << "" << std::endl;
}

// Function to generate a base pattern with a specified number of 1s
std::vector<bool> generateBasePattern(int N, int nb_winning_units) {
    std::vector<bool> basePattern(N, false);
    for (int i = 0; i < nb_winning_units; ++i) {
        basePattern[i] = true;
    }

    return basePattern;
}

// Function to flip bits in a balanced pattern based on noise level
std::vector<bool> generateNoisyBalancedPattern(
    const std::vector<bool> &basePattern, int numFlips) {
    std::vector<bool> noisyPattern = basePattern;
    int N = basePattern.size();
    int index;
    int cpt = 0;
    // Flip numFlips bits from 1 to 0
    while (cpt < numFlips)  // can loop a long time if not enough are 1s
    {
        index = rand() % N;  // trying to find a 1 index
        if (noisyPattern[index] == true) {
            noisyPattern[index] = !noisyPattern[index];  // turning the 1 into 0
            index = rand() % N;
            while (noisyPattern[index] !=
                   false) {  // can loop a long time if not enough are 0s
                index = rand() % N;  // trying to find a 0 index
            }
            noisyPattern[index] = !noisyPattern[index];  // turning the 0 into 1
            cpt += 1;
        }
    }
    return noisyPattern;
}

// Function to check if a pattern already exists in a vector of patterns
bool patternExists(const std::vector<std::vector<bool>> &patterns,
                   const std::vector<bool> &pattern) {
    for (const auto &p : patterns) {
        if (p == pattern) {
            return true;
        }
    }
    return false;
}

// Function to generate K unique noisy balanced patterns
std::vector<std::vector<bool>> generatePatterns(int K, int N,
                                                int nb_winning_units,
                                                double noiseLevel) {
    std::vector<std::vector<bool>> patterns;
    std::vector<bool> basePattern = generateBasePattern(N, nb_winning_units);
    int numFlips = static_cast<int>(
        noiseLevel *
        nb_winning_units);  // Calculate number of flips based on noise level

    while (patterns.size() < K) {
        std::vector<bool> newPattern =
            generateNoisyBalancedPattern(basePattern, numFlips);
        if (!patternExists(patterns, newPattern)) {
            patterns.push_back(newPattern);
        }
    }

    return patterns;
}

std::vector<std::vector<bool>> loadPatterns(const std::string &filename) {
    std::ifstream file(filename);
    std::string line;
    std::vector<std::vector<bool>> boolVectors;

    if (file.is_open()) {
        while (getline(file, line)) {
            std::istringstream iss(line);
            std::vector<bool> boolVector;
            std::string value;

            while (iss >> value) {
                boolVector.push_back(value == "1");
            }
            boolVectors.push_back(boolVector);
        }
        file.close();
    } else {
        std::cerr << "Unable to open file" << std::endl;
    }

    return boolVectors;
}

// Helper function to compare vectors of bool
bool areVectorsEqual(const std::vector<bool> &v1, const std::vector<bool> &v2) {
    return v1 == v2;
}

// Function to create a text file with parameter values
void createParameterFile(
    const std::string &directory,
    const std::unordered_map<std::string, double> &parameters) {
    // Create the directory if it doesn't exist
    std::filesystem::create_directories(directory);

    // Create the full path to the file
    std::string filePath = directory + "/parameters.data";

    // Open a file stream to write the parameters
    std::ofstream outFile(filePath);

    // Check if the file was opened successfully
    if (!outFile) {
        std::cerr << "Error: Could not create file " << filePath << std::endl;
        return;
    }

    // Write each parameter to the file
    for (const auto &param : parameters) {
        outFile << param.first << "=" << param.second << "\n";
    }

    // Close the file stream
    outFile.close();
}

std::vector<std::vector<double>> patterns_as_states(
    double up_rate, double down_rate,
    std::vector<std::vector<bool>> bin_patterns) {
    std::vector<double> state_input(bin_patterns[0].size());
    std::vector<std::vector<double>> initial_patterns_state_list(
        bin_patterns.size());
    for (int i = 0; i < bin_patterns.size(); i++) {
        for (int j = 0; j < bin_patterns[i].size(); j++) {
            if (bin_patterns[i][j]) {
                state_input[j] = up_rate;
            } else {
                state_input[j] = down_rate;
            }
        }
        initial_patterns_state_list[i] = state_input;
    }
    return initial_patterns_state_list;
}

// Helper function to generate all combinations of parameters
// funny function generated by chatgpt, you can use division and module to do
// combinatorials tricks
std::vector<std::unordered_map<std::string, double>> generateCombinations(
    const std::unordered_map<std::string, std::vector<double>>
        &varying_params) {
    std::vector<std::unordered_map<std::string, double>> combinations;

    // Calculate the total number of combinations
    size_t total_combinations = 1;
    for (const auto &param : varying_params) {
        total_combinations *= param.second.size();
    }

    // Generate all combinations
    for (size_t i = 0; i < total_combinations; ++i) {
        std::unordered_map<std::string, double> combination;
        size_t index = i;
        for (const auto &param : varying_params) {
            combination[param.first] =
                param.second[index % param.second.size()];
            index /= param.second.size();
        }
        combinations.push_back(combination);
    }

    return combinations;
}

std::unordered_map<std::string, double> fuseMaps(
    std::unordered_map<std::string, double> map1,
    std::unordered_map<std::string, double> map2) {
    for (const auto &element : map2) {
        map1[element.first] = element.second;
    }
    return map1;
}

// Function to read a file and store the content in an unordered_map
std::unordered_map<std::string, double> readParametersFile(
    const std::string &filePath) {
    std::unordered_map<std::string, double> parameters;
    std::ifstream file(filePath);

    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            std::istringstream lineStream(line);
            std::string key;
            if (std::getline(lineStream, key, '=')) {
                std::string valueStr;
                if (std::getline(lineStream, valueStr)) {
                    try {
                        double value = std::stod(valueStr);
                        parameters[key] = value;
                    } catch (const std::invalid_argument &e) {
                        std::cerr << "Invalid value for key: " << key
                                  << std::endl;
                    } catch (const std::out_of_range &e) {
                        std::cerr << "Value out of range for key: " << key
                                  << std::endl;
                    }
                }
            }
        }
        file.close();
    } else {
        std::cerr << "Unable to open file: " << filePath << std::endl;
    }

    return parameters;
}

std::vector<std::vector<double>> readMatrixFromFile(
    const std::string &filePath) {
    std::vector<std::vector<double>> matrix;
    std::ifstream inFile(filePath);
    if (!inFile) {
        std::cerr << "Error: Unable to open file '" << filePath << "'."
                  << std::endl;
        std::cerr << "Error code: " << errno << " (" << strerror(errno) << ")"
                  << std::endl;

        // Check if the file exists by trying to open it in write mode
        std::ofstream test(filePath, std::ios::in);
        if (test.is_open()) {
            std::cerr << "File exists but cannot be opened for reading. Check "
                         "permissions."
                      << std::endl;
            test.close();
        } else {
            std::cerr << "File does not exist or path is incorrect."
                      << std::endl;
        }

        return matrix;
    }

    std::string line;
    while (std::getline(inFile, line)) {
        std::istringstream ss(line);
        std::vector<double> row;
        double value;

        while (ss >> value) {
            row.push_back(value);
        }

        matrix.push_back(row);
    }

    inFile.close();
    return matrix;
}

std::vector<std::vector<bool>> readBoolMatrixFromFile(
    const std::string &filePath) {
    std::vector<std::vector<bool>> matrix;
    std::ifstream inFile(filePath);

    if (!inFile.is_open()) {
        std::cerr << "Error opening file for reading: " << filePath
                  << std::endl;
        return matrix;
    }

    std::string line;
    while (std::getline(inFile, line)) {
        std::istringstream ss(line);
        std::vector<bool> row;
        double value;

        while (ss >> value) {
            row.push_back(value);
        }

        matrix.push_back(row);
    }

    inFile.close();
    return matrix;
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


void collectSimulationDataSeries(const std::string &folderResultsPath) {
    std::vector<std::unordered_map<std::string, std::string>> allSimData;
    std::vector<std::string> allKeys;
    std::vector<std::string> resultKeys;
    std::string path_name;
    int first_sim_visited = true;
    allKeys.push_back("sim_ID");

    // Iterate through all subdirectories
    for (const auto &entry : fs::directory_iterator(folderResultsPath)) {
        if (fs::is_directory(entry)) {
            std::unordered_map<std::string, std::string> simData;
            path_name = entry.path().filename().string();
            std::regex regex_pattern(R"(\d+$)");
            std::smatch match;
            std::string sim_id;
            if (std::regex_search(path_name, match, regex_pattern)) {
                sim_id = match.str();
                std::cout << "Extracted Sim ID " << sim_id << std::endl;
            } else {
                std::cout << "No SIM ID found" << std::endl;
            }
            simData["sim_ID"] = sim_id;
            // Read parameters file
            std::ifstream paramFile(entry.path() / "parameters.data");
            if (paramFile.is_open()) {
                std::string line;
                while (std::getline(paramFile, line)) {
                    std::istringstream iss(line);
                    std::string key, value;
                    if (std::getline(iss, key, '=') &&
                        std::getline(iss, value)) {
                        simData[key] = value;
                        if (first_sim_visited) {
                            allKeys.push_back(key);
                        }
                    }
                }
                paramFile.close();
            }

            // Read results file
            std::ifstream resultFile(entry.path() / "results.data");
            if (resultFile.is_open()) {
                std::string line;
                std::getline(resultFile, line);
                if (first_sim_visited) {
                    std::istringstream iss(line);
                    std::string key;
                    while (std::getline(iss, key, ',')) {
                        allKeys.push_back(key);
                        resultKeys.push_back(key);
                    }
                    first_sim_visited = false;
                }
                while (std::getline(resultFile, line)) {
                    int nb_elements = 0;
                    std::istringstream iss(line);
                    std::string value;
                    while (std::getline(iss, value, ',')) {
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
    if (csvFile.is_open()) {
        // Write header
        for (const auto &key : allKeys) {
            csvFile << key << ",";
        }
        csvFile << "\n";

        // Write data
        for (const auto &simData : allSimData) {
            for (const auto &key : allKeys) {
                auto it = simData.find(key);
                if (it != simData.end()) {
                    csvFile << it->second;
                }
                csvFile << ",";
            }
            csvFile << "\n";
        }
        csvFile.close();
        std::cout
            << "All simulation data has been written to all_simulation_data.csv"
            << std::endl;
    } else {
        std::cerr << "Unable to open file for writing CSV data." << std::endl;
    }
}

//-------------------------------------------------LINESPACE STUFF
// Function to generate linspace equivalent
std::vector<double> linspace(double start, double end, int num) {
    std::vector<double> result;
    if (num <= 0) {
        return result;
    }
    if (num == 1) {
        result.push_back(start);
        return result;
    }

    double step = (end - start) / (num - 1);
    for (int i = 0; i < num; ++i) {
        result.push_back(start + i * step);
    }
    return result;
}

// Function to generate linspace equivalent for vectors
std::vector<std::vector<double>> linspace_vector(std::vector<double> X,
                                                 std::vector<double> Y,
                                                 int num) {
    std::vector<std::vector<double>> vectors(X.size(),
                                             std::vector<double>(num));
    for (size_t i = 0; i < X.size(); i++) {
        vectors[i] = linspace(X[i], Y[i], num);
    }
    return transposeMatrix(vectors);
}

std::vector<std::vector<double>> linspace_square_mesh(
    std::vector<std::vector<double>> edge_1,
    std::vector<std::vector<double>> edge_2) {
    std::vector<std::vector<std::vector<double>>> square_mesh(edge_1.size());
    std::vector<std::vector<double>> x_coordinates(edge_1.size());
    std::vector<std::vector<double>> y_coordinates(edge_1.size());
    std::vector<double> incr(edge_1.size());
    std::iota(incr.begin(), incr.end(), 0);

    for (size_t i = 0; i < edge_1.size(); i++) {
        square_mesh[i] = linspace_vector(edge_1[i], edge_2[i], edge_1.size());
        x_coordinates[i] = incr;
        y_coordinates[i] =
            std::vector<double>(edge_1.size(), edge_1.size() - i);
    }
    // coordinates_and_values = // continue here
    return flattenFirstDim3D(square_mesh);
}

//-------------------------------------------------VECTOR FIELD STUFF
/**
 * @brief Compute and save the vector field in rate space (a0,a1).
 *
 * We sample (u0,u1) on a given grid, then compute:
 *    d(u_i)/dt = sum_j [ W[i][j]*v_j ] - leak * u_i
 * where v_j = transfer(u_j).
 *
 * We store lines of the form:
 *    u0  u1  du0_dt  du1_dt
 */
void compute_and_save_potential_vector_field(Network &net,
                                             const std::string &filename,
                                             double u_min, double u_max,
                                             double step) {
    std::ofstream file(filename, std::ios::trunc);
    if (!file.is_open()) {
        std::cerr << "Cannot open file " << filename << std::endl;
        return;
    }
    // For a 2-unit network
    double leak = net.leak;
    // We'll temporarily store the original network state

    // Loop over a grid in activity space
    for (double u0 = u_min; u0 <= u_max; u0 += step) {
        for (double u1 = u_min; u1 <= u_max; u1 += step) {
            // Compute the rate from this activity
            double v0 = net.transfer(u0);
            double v1 = net.transfer(u1);

            // d(u0)/dt = sum_j [ W[0][j] * r_j ] - leak * u0
            // d(u1)/dt = sum_j [ W[1][j] * r_j ] - leak * u1
            double du0 = 0.0;
            double du1 = 0.0;

            // Just do it manually for 2 units:
            du0 = net.weight_matrix[0][0] * v0 + net.weight_matrix[0][1] * v1 -
                  leak * u0;

            du1 = net.weight_matrix[1][0] * v0 + net.weight_matrix[1][1] * v1 -
                  leak * u1;

            // Save: u0, u1, du0, du1
            file << u0 << " " << u1 << " " << du0 << " " << du1 << "\n";
        }
    }

    file.close();
}

/**
 * @brief Compute and save the vector field in RATE space (v0,v1).
 *
 * We sample (v0,v1) in [0,1], then compute:
 *    u_i = logit(v_i) = ln(v_i / (1 - v_i))
 *    d(u_i)/dt = sum_j [ W[i][j] * v_j ] - leak * u_i
 *    d(u_i)/dt = derivative_of_transfer(u_i) * d(u_i)/dt
 *              = v_i * (1 - v_i) * d(u_i)/dt
 *
 * We store lines of the form:
 *    v0  v1  dv0_dt  dv1_dt
 */
void compute_and_save_rate_vector_field(Network &net,
                                        const std::string &filename,
                                        double v_min, double v_max,
                                        double step) {
    std::ofstream file(filename, std::ios::trunc);
    if (!file.is_open()) {
        std::cerr << "Cannot open file " << filename << std::endl;
        return;
    }
    double leak = net.leak;

    for (double v0 = v_min; v0 <= v_max; v0 += step) {
        for (double v1 = v_min; v1 <= v_max; v1 += step) {
            // u0 = logit(v0), u1 = logit(v1)
            // but ensure v0,v1 are strictly between 0 and 1 to avoid blow-up:
            if (v0 <= 0.0 || v0 >= 1.0 || v1 <= 0.0 || v1 >= 1.0) {
                continue;
            }
            double u0 = std::log(v0 / (1.0 - v0));
            double u1 = std::log(v1 / (1.0 - v1));

            // du0_dt = sum_j [ W[0][j] * v_j ] - leak * u0
            double du0 = net.weight_matrix[0][0] * v0 +
                         net.weight_matrix[0][1] * v1 - leak * u0;

            // du1_dt = sum_j [ W[1][j] * v_j ] - leak * u1
            double du1 = net.weight_matrix[1][0] * v0 +
                         net.weight_matrix[1][1] * v1 - leak * u1;

            // dv_i/dt = v_i*(1-v_i)*du_i
            double dv0 = v0 * (1.0 - v0) * du0;
            double dv1 = v1 * (1.0 - v1) * du1;

            // Save: v0, v1, dv0, dv1
            file << v0 << " " << v1 << " " << dv0 << " " << dv1 << "\n";
        }
    }

    file.close();
}


/**
 * @brief Compute and save the 2D-projected derivative field in RATE space
 * for two patterns (affine plane among p1->0 and p2->1).
 *
 * Instead of naive dot-product with p1,p2, we do a local "finite difference"
 * approach to find the tangent directions in the plane at each point, then
 * project the full 10D derivative onto those directions.
 */
void compute_and_save_rate_vector_field_two_pattern_interpolate_plane( double delta,
    Network &net, const std::string &foldername, const std::string &filename,
    const std::vector<double> &pattern_1_rate,
    const std::vector<double> &pattern_2_rate, int nb_step) {
    // Output file
    std::string out_filename =
        foldername + "/vector_field_two_patterns_" + filename + ".txt";
    std::ofstream file(out_filename, std::ios::trunc);
    if (!file.is_open()) {
        std::cerr << "Cannot open file " << out_filename << std::endl;
        return;
    }

    int size = net.size;  // e.g. 10

    // 1) Build edges (pattern1->0, pattern2->1) and the square mesh in 10D
    std::vector<double> fully_inactivated_state(size, 0.02);
    std::vector<double> fully_activated_state(size, 0.98);

    std::vector<std::vector<double>> edge_1 =
        linspace_vector(fully_inactivated_state, pattern_2_rate, nb_step);
    std::vector<std::vector<double>> edge_2 =
        linspace_vector(pattern_1_rate, fully_activated_state, nb_step);

    // Flatten the 2D grid into nb_step^2 points in 10D
    std::vector<std::vector<double>> square_mesh_values =
        linspace_square_mesh(edge_1, edge_2);
    
    // for (size_t k = 0; k < square_mesh_values.size(); k++)
    // {
    //     for (size_t i = 0; i < square_mesh_values[k].size(); i++)
    //     {
    //         std::cout << square_mesh_values[k][i] <<" ";
    //     }
        
    //     std::cout<<std::endl;
    // }
    
    // 3) Helper: a function to index (i,j) => flatten
    auto idx_ij = [&](int i, int j) { return i * nb_step + j; };

    // We'll also define a small function to "safe normalize" a 10D vector:
    auto safe_normalize = [&](std::vector<double> &vec) {
        double norm = 0.0;
        for (auto &x : vec)
            norm += x * x;
        norm = std::sqrt(norm);
        if (norm > 1e-12) {
            for (auto &x : vec)
                x /= norm;
        }
    };

    // 4) For each grid point (i,j), approximate e_x and e_y by finite
    // differences
    //    and project the 10D derivative.
    //    Then write: x_2D, y_2D, dot_x, dot_y, plus the 10D rates.
    for (int i = 0; i < nb_step; i++) {
        for (int j = 0; j < nb_step; j++) {
            int flat_idx = idx_ij(i, j);

            // The 10D state at (i,j)
            const std::vector<double> &V = square_mesh_values[flat_idx];

            // Compute 10D derivative
            net.set_state(V);
            // std::vector<double> dV = net.give_derivative_v(net.give_derivative_u());
            std::vector<double> dV =net.give_derivative_v(net.give_derivative_u(delta));
            // std::cout << "showing the derivative :" << std::endl;
            // std::cout <<i<<" "<<j<< std::endl;

            // for (size_t k = 0; k < size; k++) {
            //     std::cout << V[k] << " ";
            // }
            // std::cout << std::endl;
            // for (size_t k = 0; k < size; k++)
            // {
            //     std::cout<<dV[k]<<" ";
            // }
            // std::cout << std::endl;
            // Approx e_x = neighbor in j-direction
            // If j+1 < nb_step, use forward difference. Else use backward.
            std::vector<double> e_x(size, 0.0);
            if (j + 1 < nb_step) {
                const std::vector<double> &V_right =
                    square_mesh_values[idx_ij(i, j + 1)];
                for (int dd = 0; dd < size; dd++) {
                    e_x[dd] = V_right[dd] - V[dd];
                }
            } else if (j > 0) {
                // fallback: use backward difference
                const std::vector<double> &V_left =
                    square_mesh_values[idx_ij(i, j - 1)];
                for (int dd = 0; dd < size; dd++) {
                    e_x[dd] = V[dd] - V_left[dd];
                }
            }
            safe_normalize(e_x);

            // Approx e_y = neighbor in i-direction
            std::vector<double> e_y(size, 0.0);
            if (i + 1 < nb_step) {
                const std::vector<double> &V_down =
                    square_mesh_values[idx_ij(i + 1, j)];
                for (int dd = 0; dd < size; dd++) {
                    e_y[dd] = V_down[dd] - V[dd];
                }
            } else if (i > 0) {
                // fallback: use backward difference
                const std::vector<double> &V_up =
                    square_mesh_values[idx_ij(i - 1, j)];
                for (int dd = 0; dd < size; dd++) {
                    e_y[dd] = V[dd] - V_up[dd];
                }
            }
            safe_normalize(e_y);

            // Now project dV onto e_x => dot_x, onto e_y => dot_y
            double dot_x = 0.0, dot_y = 0.0;
            for (int dd = 0; dd < size; dd++) {
                dot_x += dV[dd] * e_x[dd];
                dot_y += dV[dd] * e_y[dd];
            }

            // The 2D coordinates in [0..10]
            double xx = j;
            double yy = i;

            // Write to file:
            // x_2D, y_2D, dot_x, dot_y, then the 10D rate vector
            file << xx << " " << yy << " " << dot_x << " " << dot_y << " ";
            for (int dd = 0; dd < size; dd++) {
                file << V[dd];
                if (dd < size - 1)
                    file << " ";
            }
            file << "\n";
        }
    }

    file.close();
    std::cout << "Saved 2D (projected) vector field to " << out_filename
              << std::endl;
}

//---------------------------------------------------------
// Bilinear interpolation in N dimensions
//---------------------------------------------------------
std::vector<double> bilinear_interpolate(
    const std::vector<double>
        &corner_00,  // typically fully_inactivated e.g. 0.02
    const std::vector<double> &corner_10,  // pattern 1
    const std::vector<double> &corner_01,  // pattern 2
    const std::vector<double>
        &corner_11,  // typically fully_activated e.g. 0.98
    double lambda1, double lambda2) {
    const int N = static_cast<int>(corner_00.size());
    std::vector<double> result(N, 0.0);

    for (int i = 0; i < N; i++) {
        double val = (1.0 - lambda1) * (1.0 - lambda2) * corner_00[i] +
                     (lambda1) * (1.0 - lambda2) * corner_10[i] +
                     (1.0 - lambda1) * (lambda2)*corner_01[i] +
                     (lambda1) * (lambda2)*corner_11[i];
        result[i] = val;
    }
    return result;
}

//---------------------------------------------------------
// Compute & save the 2D vector field
// over the bilinear interpolation plane
//---------------------------------------------------------
void compute_and_save_rate_vector_field_two_pattern_bilinear(
    double delta, Network &net, const std::string &foldername,
    const std::string &filename, const std::vector<double> &pattern_1_rate,
    const std::vector<double> &pattern_2_rate, int nb_step) {
    // (1) Create the four corners in ND:
    //     0_N, v^1, v^2, 1_N
    //     (Here we illustrate with 0.02 and 0.98 for "fully
    //     inactivated/activated".)
    int size = net.size;
    std::vector<double> corner_00(size, 0.02);  // ~ fully "off"
    std::vector<double> corner_11(size, 0.98);  // ~ fully "on"
    // pattern_1_rate => corner_10
    // pattern_2_rate => corner_01
    // We'll also define a small function to "safe normalize" a 10D vector:
    auto safe_normalize = [&](std::vector<double> &vec) {
        double norm = 0.0;
        for (auto &x : vec)
            norm += x * x;
        norm = std::sqrt(norm);
        if (norm > 1e-12) {
            for (auto &x : vec)
                x /= norm;
        }
    };
    // (2) Prepare output file
    std::string out_filename =
        foldername + "/vector_field_bilinear_" + filename + ".txt";
    std::ofstream file(out_filename, std::ios::trunc);
    if (!file.is_open()) {
        std::cerr << "Cannot open file " << out_filename << std::endl;
        return;
    }

    // (3) Loop over a grid of (lambda1, lambda2) in [0..1] x [0..1]
    //     (You can decide on nb_step, e.g. 20 or 50, etc.)
    for (int i = 0; i < nb_step; i++) {
        for (int j = 0; j < nb_step; j++) {
            // Convert i,j to two parameters in [0..1]
            double lambda1 = double(i) / (nb_step - 1);
            double lambda2 = double(j) / (nb_step - 1);

            // (4) Bilinear interpolation => 10D rate vector
            std::vector<double> V = bilinear_interpolate(corner_00,       // 0_N
                                                         pattern_1_rate,  // v^1
                                                         pattern_2_rate,  // v^2
                                                         corner_11,       // 1_N
                                                         lambda1, lambda2);

            // (5) Compute 10D derivative for that state
            net.set_state(V);
            std::vector<double> dU = net.give_derivative_u(delta);
            std::vector<double> dV = net.give_derivative_v(dU);

            // (6) For a "2D vector field" in the (lambda1,lambda2) plane,
            //     you can approximate partial derivatives in the ND space,
            //     then do the dot-product with dV, or simply do forward
            //     differences in lambda1/lambda2, etc.
            //
            //     Below, we do the forward-difference approach (like the
            //     original code). Alternatively, you can explicitly compute
            //     ∂/∂λ1 and ∂/∂λ2 of the bilinear formula if you prefer.

            // e_x => difference in the lambda2 direction (like "x direction")
            // Here j+1 means "move in lambda2"
            std::vector<double> e_x(size, 0.0);
            if (j + 1 < nb_step) {
                double next_lambda2 = double(j + 1) / (nb_step - 1);
                std::vector<double> V_right = bilinear_interpolate(
                    corner_00, pattern_1_rate, pattern_2_rate, corner_11,
                    lambda1, next_lambda2);
                for (int dd = 0; dd < size; dd++) {
                    e_x[dd] = V_right[dd] - V[dd];
                }
            } else if (j > 0) {
                double prev_lambda2 = double(j - 1) / (nb_step - 1);
                std::vector<double> V_left = bilinear_interpolate(
                    corner_00, pattern_1_rate, pattern_2_rate, corner_11,
                    lambda1, prev_lambda2);
                for (int dd = 0; dd < size; dd++) {
                    e_x[dd] = V[dd] - V_left[dd];
                }
            }
            safe_normalize(e_x);

            // e_y => difference in the lambda1 direction
            std::vector<double> e_y(size, 0.0);
            if (i + 1 < nb_step) {
                double next_lambda1 = double(i + 1) / (nb_step - 1);
                std::vector<double> V_down = bilinear_interpolate(
                    corner_00, pattern_1_rate, pattern_2_rate, corner_11,
                    next_lambda1, lambda2);
                for (int dd = 0; dd < size; dd++) {
                    e_y[dd] = V_down[dd] - V[dd];
                }
            } else if (i > 0) {
                double prev_lambda1 = double(i - 1) / (nb_step - 1);
                std::vector<double> V_up = bilinear_interpolate(
                    corner_00, pattern_1_rate, pattern_2_rate, corner_11,
                    prev_lambda1, lambda2);
                for (int dd = 0; dd < size; dd++) {
                    e_y[dd] = V[dd] - V_up[dd];
                }
            }
            safe_normalize(e_y);

            // (7) Now project the 10D derivative onto these directions
            double dot_x = 0.0, dot_y = 0.0;
            for (int dd = 0; dd < size; dd++) {
                dot_x += dV[dd] * e_x[dd];
                dot_y += dV[dd] * e_y[dd];
            }

            // (8) Write out:
            //     lambda1, lambda2, dot_x, dot_y, plus the ND rates
            file << lambda1 << " " << lambda2 << " " << dot_x << " " << dot_y
                 << " ";
            for (int dd = 0; dd < size; dd++) {
                file << V[dd] << ((dd < size - 1) ? ' ' : '\n');
            }
        }
    }

    file.close();
    std::cout << "Saved bilinear 2D vector field to " << out_filename
              << std::endl;
}

/**
 * @brief Compute and save the 2D-projected derivative field in potential space
 * for two patterns p1, p2 as U states . We define the plane as:
 *    U(alpha,beta) = alpha * p1 + beta * p2 
 * for alpha,beta in [0..up_lim].
 *
 * Then we do a simple dot product approach for the derivatives:
 *    dot_x = dU . p1
 *    dot_y = dU . p2
 */
void compute_and_save_potential_vector_field_two_pattern(
    double delta, Network &net, const std::string &foldername,
    const std::string &filename,
    const std::vector<double> &p1,  // pattern_1_rate
    const std::vector<double> &p2,  // pattern_2_rate
    int nb_step,                    // how many steps from 0..1
    double up_lim) {
    std::string out_filename =
        foldername + "/vector_field_two_patterns_" + filename + ".txt";
    std::ofstream file(out_filename, std::ios::trunc);
    if (!file.is_open()) {
        std::cerr << "Cannot open file " << out_filename << std::endl;
        return;
    }

    int size = net.size;  // e.g. 10

    // For i=0..nb_step-1, alpha = i/((nb_step-1)/up_lim)
    // For j=0..nb_step-1, beta  = j/((nb_step-1)/up_lim)
    // Then we store x=i, y=j in [0..nb_step-1] so Python can reshape.
    for (int i = 0; i < nb_step; i++) {
        double alpha = double(i) / ((nb_step - 1) / up_lim) -0.25;
        for (int j = 0; j < nb_step; j++) {
            double beta = double(j) / ((nb_step - 1) / up_lim) - 0.25;

            // 1) Build U = alpha*p1 + beta*p2
            std::vector<double> U(size, 0.0);
            for (int k = 0; k < size; k++) {
                double val = alpha * p1[k] + beta * p2[k];
                // clamp to [0,1] if needed
                U[k] = val;
            }

            // 2) Compute derivative dV in 10D
            std::vector<double> V(size,0.0);
            for (size_t k = 0; k < U.size(); k++)
            {
                V[k] = net.transfer(U[k]);
            }
            
            net.set_state(V);
            std::vector<double> dU = net.give_derivative_u(delta);

            double dot_alpha = 0.0, dot_beta = 0.0;
            for (int k = 0; k < size; k++) {
                dot_alpha += dU[k] * p1[k];
                dot_beta += dU[k] * p2[k];
            }
            // 5) Write line: x, y, dot_x, dot_y, V0..V(size-1)
            file << alpha << " " << beta << " " << dot_alpha << " " << dot_beta
                 << " ";
            for (int k = 0; k < size; k++) {
                file << V[k];
                if (k < size - 1)
                    file << " ";
            }
            file << "\n";
        }
    }

    file.close();
    std::cout << "Saved dotproduct-based 2D field to " << out_filename
              << std::endl;
}