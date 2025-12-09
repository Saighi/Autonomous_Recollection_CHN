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
#include <map>
#include <cerrno>
#include <cstring>
#include <cstdint>
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
            if (conf.use_full_inhibition)
            {
                net.noisy_full_depressed_iterate(conf.delta, conf.mean, conf.stddev);
            }
            else
            {
                net.noisy_depressed_iterate(conf.delta, conf.mean, conf.stddev);
            }
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

std::vector<std::vector<bool>> generatePatterns(int K, int N, double sparsity, double rho, bool use_old_patterns)
{
    // Clamp rho
    if (rho < 0.0)
    {
        rho = 0.0;
    }
    else if (rho > 1.0)
    {
        rho = 1.0;
    }

    std::vector<std::vector<bool>> patterns;
    patterns.reserve(static_cast<size_t>(K));

    if (use_old_patterns)
    {
        // ---------------- OLD MODE ----------------
        // Interpret sparsity as fraction of ACTIVE units; only supported for sparsity ~= 0.5.
        const double tol = 1e-6;
        if (std::abs(sparsity - 0.5) > tol)
        {
            // Force to 0.5 to avoid inconsistent behavior
            sparsity = 0.5;
        }

        int nb_winners = std::max(1, static_cast<int>(sparsity * N));
        int numFlips = static_cast<int>((1.0 - rho) * nb_winners);

        // Base pattern: first nb_winners ones, rest zeros
        std::vector<bool> base(N, false);
        for (int i = 0; i < nb_winners; ++i)
        {
            base[i] = true;
        }

        while (patterns.size() < static_cast<size_t>(K))
        {
            std::vector<bool> pattern = base;

            for (int f = 0; f < numFlips; ++f)
            {
                // pick a 1 -> 0
                std::vector<int> ones;
                std::vector<int> zeros;
                ones.reserve(N);
                zeros.reserve(N);
                for (int i = 0; i < N; ++i)
                {
                    if (pattern[i])
                    {
                        ones.push_back(i);
                    }
                    else
                    {
                        zeros.push_back(i);
                    }
                }
                if (!ones.empty() && !zeros.empty())
                {
                    int idx_one = ones[std::rand() % static_cast<int>(ones.size())];
                    int idx_zero = zeros[std::rand() % static_cast<int>(zeros.size())];
                    pattern[idx_one] = false;
                    pattern[idx_zero] = true;
                }
            }

            if (!patternExists(patterns, pattern))
            {
                patterns.push_back(std::move(pattern));
            }
        }
    }
    else
    {
        // ---------------- NEW MODE ----------------
        // sparsity = fraction of inactive units (P(0))
        if (sparsity < 0.0)
        {
            sparsity = 0.0;
        }
        else if (sparsity > 1.0)
        {
            sparsity = 1.0;
        }

        // Step 1: generate parent pattern with P(x_i = 0) = sparsity
        std::vector<bool> parent(N, false);
        for (int i = 0; i < N; ++i)
        {
            double u = static_cast<double>(rand()) / (static_cast<double>(RAND_MAX) + 1.0);
            // P(1) = 1 - sparsity, P(0) = sparsity
            parent[i] = (u >= sparsity);
        }

        // Number of positions to redraw per pattern
        int k = static_cast<int>((1.0 - rho) * N);
        if (k < 0)
        {
            k = 0;
        }
        else if (k > N)
        {
            k = N;
        }

        while (patterns.size() < static_cast<size_t>(K))
        {
            // Step 2: start from parent
            std::vector<bool> pattern = parent;

            // Step 3: choose k distinct indices
            if (k > 0)
            {
                std::unordered_set<int> indices;
                while (static_cast<int>(indices.size()) < k)
                {
                    int idx = rand() % N;
                    indices.insert(idx);
                }

                // Step 4: redraw bits at those indices with P(0) = sparsity, P(1) = 1 - sparsity
                for (int idx : indices)
                {
                    double u = static_cast<double>(rand()) / (static_cast<double>(RAND_MAX) + 1.0);
                    pattern[idx] = (u >= sparsity);
                }
            }

            if (!patternExists(patterns, pattern))
            {
                patterns.push_back(std::move(pattern));
            }
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

bool matchesPatternOrConverse(const std::vector<bool>& pattern1, const std::vector<bool>& pattern2)
{
    if (pattern1.size() != pattern2.size())
        return false;

    bool direct_match = true;
    bool converse_match = true;

    for (size_t i = 0; i < pattern1.size(); ++i)
    {
        if (pattern1[i] != pattern2[i])
            direct_match = false;
        if (pattern1[i] == pattern2[i])  // For converse, all bits must be flipped
            converse_match = false;
    }

    return direct_match || converse_match;
}

// ============================================================================
// Heterogeneous Pattern Generation
// ============================================================================

std::pair<std::vector<std::vector<bool>>, PatternMetadata> generatePatternsHeterogeneous(
    int K, int N, double mean_sparsity, double sparsity_width, double rho)
{
    // Clamp parameters to valid ranges
    if (mean_sparsity < 0.0) mean_sparsity = 0.0;
    else if (mean_sparsity > 1.0) mean_sparsity = 1.0;

    if (rho < 0.0) rho = 0.0;
    else if (rho > 1.0) rho = 1.0;

    // Generate parent pattern with P(0) = mean_sparsity
    std::vector<bool> parent(N, false);
    for (int i = 0; i < N; ++i)
    {
        double u = static_cast<double>(rand()) / (static_cast<double>(RAND_MAX) + 1.0);
        parent[i] = (u >= mean_sparsity);  // P(1) = 1 - mean_sparsity
    }

    // Number of positions to redraw per pattern
    int k = static_cast<int>((1.0 - rho) * N);
    if (k < 0) k = 0;
    else if (k > N) k = N;

    std::vector<std::vector<bool>> patterns;
    PatternMetadata metadata;
    metadata.num_patterns = K;
    metadata.network_size = N;
    metadata.generation_method = "heterogeneous";
    metadata.mean_sparsity = mean_sparsity;
    metadata.sparsity_width = sparsity_width;
    metadata.rho = rho;

    while (static_cast<int>(patterns.size()) < K)
    {
        // Sample sparsity for this pattern from Uniform(mean - width/2, mean + width/2)
        double u_width = static_cast<double>(rand()) / (static_cast<double>(RAND_MAX) + 1.0);
        double s_i = mean_sparsity + (u_width - 0.5) * sparsity_width;
        // Clamp to valid range
        if (s_i < 0.01) s_i = 0.01;
        else if (s_i > 0.99) s_i = 0.99;

        // Start from parent
        std::vector<bool> pattern = parent;

        // Choose k distinct indices to redraw
        if (k > 0)
        {
            std::unordered_set<int> indices;
            while (static_cast<int>(indices.size()) < k)
            {
                indices.insert(rand() % N);
            }

            // Redraw bits at those indices with P(0) = s_i
            for (int idx : indices)
            {
                double u = static_cast<double>(rand()) / (static_cast<double>(RAND_MAX) + 1.0);
                pattern[idx] = (u >= s_i);
            }
        }

        // Check uniqueness
        if (!patternExists(patterns, pattern))
        {
            patterns.push_back(std::move(pattern));

            // Compute actual sparsity from the pattern we just added
            int nb_active = 0;
            for (bool b : patterns.back())
            {
                if (b) nb_active++;
            }
            double actual_sparsity = 1.0 - static_cast<double>(nb_active) / N;

            PatternInfo info;
            info.index = static_cast<int>(patterns.size()) - 1;
            info.sparsity = actual_sparsity;
            info.nb_active = nb_active;
            metadata.patterns.push_back(info);
        }
    }

    return {patterns, metadata};
}

void writePatternMetadata(const PatternMetadata& metadata, const std::string& filepath)
{
    std::ofstream file(filepath);
    if (!file.is_open())
    {
        std::cerr << "Warning: Could not open " << filepath << " for writing metadata" << std::endl;
        return;
    }

    file << "{\n";
    file << "  \"version\": " << metadata.version << ",\n";
    file << "  \"num_patterns\": " << metadata.num_patterns << ",\n";
    file << "  \"network_size\": " << metadata.network_size << ",\n";
    file << "  \"generation_method\": \"" << metadata.generation_method << "\",\n";
    file << "  \"global_params\": {\n";
    file << "    \"mean_sparsity\": " << metadata.mean_sparsity << ",\n";
    file << "    \"sparsity_width\": " << metadata.sparsity_width << ",\n";
    file << "    \"rho\": " << metadata.rho << "\n";
    file << "  },\n";
    file << "  \"patterns\": [\n";
    for (size_t i = 0; i < metadata.patterns.size(); ++i)
    {
        const auto& p = metadata.patterns[i];
        file << "    {\"index\": " << p.index
             << ", \"sparsity\": " << p.sparsity
             << ", \"nb_active\": " << p.nb_active << "}";
        if (i < metadata.patterns.size() - 1) file << ",";
        file << "\n";
    }
    file << "  ]\n";
    file << "}\n";

    file.close();
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
// Binary Matrix I/O Functions
// ============================================================================

void writeBinaryMatrix(const std::vector<std::vector<double>>& matrix, std::ostream& out)
{
    uint32_t rows = static_cast<uint32_t>(matrix.size());
    uint32_t cols = rows > 0 ? static_cast<uint32_t>(matrix[0].size()) : 0;
    out.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    out.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
    for (const auto& row : matrix)
    {
        out.write(reinterpret_cast<const char*>(row.data()), cols * sizeof(double));
    }
}

void writeBinaryMatrix(const std::vector<std::vector<double>>& matrix, const std::string& filePath)
{
    std::ofstream out(filePath, std::ios::binary);
    if (!out.is_open())
    {
        std::cerr << "Error opening file for binary writing: " << filePath << std::endl;
        return;
    }
    writeBinaryMatrix(matrix, out);
    out.close();
}

std::vector<std::vector<double>> readBinaryMatrix(std::istream& in)
{
    uint32_t rows, cols;
    in.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    in.read(reinterpret_cast<char*>(&cols), sizeof(cols));

    std::vector<std::vector<double>> matrix(rows, std::vector<double>(cols));
    for (auto& row : matrix)
    {
        in.read(reinterpret_cast<char*>(row.data()), cols * sizeof(double));
    }
    return matrix;
}

std::vector<std::vector<double>> readBinaryMatrix(const std::string& filePath)
{
    std::ifstream in(filePath, std::ios::binary);
    if (!in.is_open())
    {
        std::cerr << "Error opening file for binary reading: " << filePath << std::endl;
        return {};
    }
    auto matrix = readBinaryMatrix(in);
    in.close();
    return matrix;
}

void writeBitpackedBoolMatrix(const std::vector<std::vector<bool>>& matrix, std::ostream& out)
{
    uint32_t rows = static_cast<uint32_t>(matrix.size());
    uint32_t cols = rows > 0 ? static_cast<uint32_t>(matrix[0].size()) : 0;
    out.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    out.write(reinterpret_cast<const char*>(&cols), sizeof(cols));

    uint8_t byte = 0;
    int bit_pos = 0;
    for (const auto& row : matrix)
    {
        for (bool val : row)
        {
            if (val) byte |= (1 << bit_pos);
            if (++bit_pos == 8)
            {
                out.write(reinterpret_cast<const char*>(&byte), 1);
                byte = 0;
                bit_pos = 0;
            }
        }
    }
    // Write remaining bits if any
    if (bit_pos > 0)
    {
        out.write(reinterpret_cast<const char*>(&byte), 1);
    }
}

void writeBitpackedBoolMatrix(const std::vector<std::vector<bool>>& matrix, const std::string& filePath)
{
    std::ofstream out(filePath, std::ios::binary);
    if (!out.is_open())
    {
        std::cerr << "Error opening file for binary writing: " << filePath << std::endl;
        return;
    }
    writeBitpackedBoolMatrix(matrix, out);
    out.close();
}

std::vector<std::vector<bool>> readBitpackedBoolMatrix(std::istream& in)
{
    uint32_t rows, cols;
    in.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    in.read(reinterpret_cast<char*>(&cols), sizeof(cols));

    size_t total_bits = static_cast<size_t>(rows) * cols;
    size_t num_bytes = (total_bits + 7) / 8;
    std::vector<uint8_t> packed(num_bytes);
    in.read(reinterpret_cast<char*>(packed.data()), num_bytes);

    std::vector<std::vector<bool>> matrix(rows, std::vector<bool>(cols));
    size_t bit_idx = 0;
    for (auto& row : matrix)
    {
        for (size_t j = 0; j < cols; j++)
        {
            row[j] = (packed[bit_idx / 8] >> (bit_idx % 8)) & 1;
            bit_idx++;
        }
    }
    return matrix;
}

std::vector<std::vector<bool>> readBitpackedBoolMatrix(const std::string& filePath)
{
    std::ifstream in(filePath, std::ios::binary);
    if (!in.is_open())
    {
        std::cerr << "Error opening file for binary reading: " << filePath << std::endl;
        return {};
    }
    auto matrix = readBitpackedBoolMatrix(in);
    in.close();
    return matrix;
}

std::vector<uint8_t> matrixToBlob(const std::vector<std::vector<double>>& matrix)
{
    std::ostringstream oss(std::ios::binary);
    writeBinaryMatrix(matrix, oss);
    std::string str = oss.str();
    return std::vector<uint8_t>(str.begin(), str.end());
}

std::vector<uint8_t> boolMatrixToBlob(const std::vector<std::vector<bool>>& matrix)
{
    std::ostringstream oss(std::ios::binary);
    writeBitpackedBoolMatrix(matrix, oss);
    std::string str = oss.str();
    return std::vector<uint8_t>(str.begin(), str.end());
}

std::vector<std::vector<double>> blobToMatrix(const std::vector<uint8_t>& blob)
{
    std::istringstream iss(std::string(blob.begin(), blob.end()), std::ios::binary);
    return readBinaryMatrix(iss);
}

std::vector<std::vector<bool>> blobToBoolMatrix(const std::vector<uint8_t>& blob)
{
    std::istringstream iss(std::string(blob.begin(), blob.end()), std::ios::binary);
    return readBitpackedBoolMatrix(iss);
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
    std::map<int, std::unordered_map<std::string, std::string>> finalStates; // Track final state per sim
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
            int sim_id_int = -1;
            if (std::regex_search(path_name, match, regex_pattern))
            {
                sim_id = match.str();
                sim_id_int = std::stoi(sim_id);
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

                // Store final state (simData now contains last iteration's values)
                if (sim_id_int >= 0)
                {
                    finalStates[sim_id_int] = simData;
                }
            }
        }
    }

    // Write all_simulation_data.csv (full time series)
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

    // Write final_results.csv (one row per simulation, final iteration only)
    std::ofstream finalFile(folderResultsPath + "/final_results.csv");
    if (finalFile.is_open())
    {
        // Write header
        for (const auto& key : allKeys)
        {
            finalFile << key << ",";
        }
        finalFile << "\n";

        // Write one row per simulation (sorted by sim_id)
        for (const auto& [sim_id, simData] : finalStates)
        {
            for (const auto& key : allKeys)
            {
                auto it = simData.find(key);
                if (it != simData.end())
                {
                    finalFile << it->second;
                }
                finalFile << ",";
            }
            finalFile << "\n";
        }
        finalFile.close();
        std::cout << "Final results have been written to final_results.csv ("
                  << finalStates.size() << " simulations)" << std::endl;
    }
    else
    {
        std::cerr << "Unable to open file for writing final_results.csv" << std::endl;
    }
}
