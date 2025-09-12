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

void compute_and_save_energy_field_two_pattern(
    double delta, Network &net, const std::string &foldername,
    const std::string &filename,
    const std::vector<double> &p1,  // pattern_1_potential
    const std::vector<double> &p2,  // pattern_2_potential
    int nb_step,                    // how many steps from 0..1
    double up_lim) {
    std::string out_filename =
        foldername + "/energy_field_two_patterns_" + filename + ".txt";
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
        double alpha = double(i) / ((nb_step - 1) / up_lim) - 0.25;
        for (int j = 0; j < nb_step; j++) {
            double beta = double(j) / ((nb_step - 1) / up_lim) - 0.25;

            // 1) Build U = alpha*p1 + beta*p2
            std::vector<double> U(size, 0.0);
            for (int k = 0; k < size; k++) {
                double val = alpha * p1[k] + beta * p2[k];
                U[k] = val;
            }

            // 2) Set the network state based on U
            std::vector<double> V(size, 0.0);
            for (size_t k = 0; k < U.size(); k++) {
                V[k] = net.transfer(U[k]);
            }

            net.set_state(V);

            // 3) Compute the energy at this state
            double energy = net.compute_energy();

            // 4) Write line: alpha, beta, energy, V0..V(size-1)
            file << alpha << " " << beta << " " << energy << " ";
            for (int k = 0; k < size; k++) {
                file << V[k];
                if (k < size - 1)
                    file << " ";
            }
            file << "\n";
        }
    }

    file.close();
    std::cout << "Saved energy field to " << out_filename << std::endl;
}