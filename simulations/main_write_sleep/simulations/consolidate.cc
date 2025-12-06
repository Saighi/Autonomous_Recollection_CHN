/**
 * consolidate.cc - Consolidate simulation results into SQLite archive
 *
 * Usage: ./consolidate <results_dir> [output.db]
 *
 * Reads all sim_nb_X folders and consolidates them into a single SQLite database
 * with binary-packed matrices for efficient storage and easy archiving.
 */

#include <iostream>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <regex>
#include <sqlite3.h>
#include "utils.hpp"

namespace fs = std::filesystem;

// Convert parameters map to JSON string
std::string paramsToJson(const std::unordered_map<std::string, double>& params)
{
    std::ostringstream oss;
    oss << "{";
    bool first = true;
    for (const auto& [key, value] : params)
    {
        if (!first) oss << ",";
        first = false;
        oss << "\"" << key << "\":" << value;
    }
    oss << "}";
    return oss.str();
}

// Extract simulation ID from folder name (e.g., "sim_nb_42" -> 42)
int extractSimId(const std::string& folder_name)
{
    std::regex pattern(R"(\d+$)");
    std::smatch match;
    if (std::regex_search(folder_name, match, pattern))
    {
        return std::stoi(match.str());
    }
    return -1;
}

// Insert simulation into database
void insertSimulation(sqlite3* db, int sim_id, const std::string& params_json,
                      const std::vector<uint8_t>& weights_blob,
                      const std::vector<uint8_t>& conn_blob,
                      const std::vector<uint8_t>& patterns_blob)
{
    sqlite3_stmt* stmt;
    const char* sql = "INSERT OR REPLACE INTO simulations (sim_id, params, weights, connectivity, patterns) VALUES (?, ?, ?, ?, ?)";

    if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) != SQLITE_OK)
    {
        std::cerr << "Failed to prepare statement: " << sqlite3_errmsg(db) << std::endl;
        return;
    }

    sqlite3_bind_int(stmt, 1, sim_id);
    sqlite3_bind_text(stmt, 2, params_json.c_str(), -1, SQLITE_TRANSIENT);

    if (!weights_blob.empty())
        sqlite3_bind_blob(stmt, 3, weights_blob.data(), weights_blob.size(), SQLITE_TRANSIENT);
    else
        sqlite3_bind_null(stmt, 3);

    if (!conn_blob.empty())
        sqlite3_bind_blob(stmt, 4, conn_blob.data(), conn_blob.size(), SQLITE_TRANSIENT);
    else
        sqlite3_bind_null(stmt, 4);

    if (!patterns_blob.empty())
        sqlite3_bind_blob(stmt, 5, patterns_blob.data(), patterns_blob.size(), SQLITE_TRANSIENT);
    else
        sqlite3_bind_null(stmt, 5);

    if (sqlite3_step(stmt) != SQLITE_DONE)
    {
        std::cerr << "Failed to insert simulation: " << sqlite3_errmsg(db) << std::endl;
    }

    sqlite3_finalize(stmt);
}

// Insert results from CSV file
void insertResultsFromCSV(sqlite3* db, int sim_id, const std::string& results_path)
{
    std::ifstream file(results_path);
    if (!file.is_open())
    {
        return;
    }

    std::string line;
    // Skip header
    std::getline(file, line);

    sqlite3_stmt* stmt;
    const char* sql = "INSERT INTO results (sim_id, query_iter, nb_fnd_pat, nb_spurious, "
                      "nb_iter_biased, nb_iter_free, all_recovered_before_spurious) "
                      "VALUES (?, ?, ?, ?, ?, ?, ?)";

    if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) != SQLITE_OK)
    {
        std::cerr << "Failed to prepare results statement: " << sqlite3_errmsg(db) << std::endl;
        return;
    }

    while (std::getline(file, line))
    {
        std::istringstream iss(line);
        std::string token;
        std::vector<int> values;

        while (std::getline(iss, token, ','))
        {
            try
            {
                values.push_back(std::stoi(token));
            }
            catch (...)
            {
                values.push_back(0);
            }
        }

        if (values.size() >= 6)
        {
            sqlite3_bind_int(stmt, 1, sim_id);
            sqlite3_bind_int(stmt, 2, values[0]);  // query_iter
            sqlite3_bind_int(stmt, 3, values[1]);  // nb_fnd_pat
            sqlite3_bind_int(stmt, 4, values[2]);  // nb_spurious
            sqlite3_bind_int(stmt, 5, values[3]);  // nb_iter_biased
            sqlite3_bind_int(stmt, 6, values[4]);  // nb_iter_free
            sqlite3_bind_int(stmt, 7, values.size() > 5 ? values[5] : 0);  // all_recovered_before_spurious

            if (sqlite3_step(stmt) != SQLITE_DONE)
            {
                std::cerr << "Failed to insert result row: " << sqlite3_errmsg(db) << std::endl;
            }
            sqlite3_reset(stmt);
        }
    }

    sqlite3_finalize(stmt);
    file.close();
}

void consolidate(const std::string& results_dir, const std::string& output_db)
{
    // Create/open database
    sqlite3* db;
    if (sqlite3_open(output_db.c_str(), &db) != SQLITE_OK)
    {
        std::cerr << "Failed to open database: " << sqlite3_errmsg(db) << std::endl;
        return;
    }

    // Create tables
    const char* create_sql = R"(
        CREATE TABLE IF NOT EXISTS simulations (
            sim_id INTEGER PRIMARY KEY,
            params TEXT,
            weights BLOB,
            connectivity BLOB,
            patterns BLOB
        );
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sim_id INTEGER,
            query_iter INTEGER,
            nb_fnd_pat INTEGER,
            nb_spurious INTEGER,
            nb_iter_biased INTEGER,
            nb_iter_free INTEGER,
            all_recovered_before_spurious INTEGER,
            FOREIGN KEY(sim_id) REFERENCES simulations(sim_id)
        );
        CREATE INDEX IF NOT EXISTS idx_results_sim ON results(sim_id);
    )";

    char* err_msg = nullptr;
    if (sqlite3_exec(db, create_sql, nullptr, nullptr, &err_msg) != SQLITE_OK)
    {
        std::cerr << "Failed to create tables: " << err_msg << std::endl;
        sqlite3_free(err_msg);
        sqlite3_close(db);
        return;
    }

    // Begin transaction for faster inserts
    sqlite3_exec(db, "BEGIN TRANSACTION", nullptr, nullptr, nullptr);

    int count = 0;

    // Iterate through sim_nb_X directories
    for (const auto& entry : fs::directory_iterator(results_dir))
    {
        if (!fs::is_directory(entry))
            continue;

        std::string folder_name = entry.path().filename().string();
        if (folder_name.find("sim_nb_") != 0)
            continue;

        std::string sim_dir = entry.path().string();
        int sim_id = extractSimId(folder_name);

        if (sim_id < 0)
        {
            std::cerr << "Could not extract sim_id from: " << folder_name << std::endl;
            continue;
        }

        // Read parameters
        std::unordered_map<std::string, double> params;
        std::string params_path = sim_dir + "/parameters.data";
        if (fs::exists(params_path))
        {
            params = readParametersFile(params_path);
        }
        std::string params_json = paramsToJson(params);

        // Read and convert matrices to blobs
        std::vector<uint8_t> weights_blob;
        std::vector<uint8_t> conn_blob;
        std::vector<uint8_t> patterns_blob;

        std::string weights_path = sim_dir + "/weights.data";
        if (fs::exists(weights_path))
        {
            auto weights = readMatrixFromFile(weights_path);
            if (!weights.empty())
            {
                weights_blob = matrixToBlob(weights);
            }
        }

        std::string conn_path = sim_dir + "/connectivity.data";
        if (fs::exists(conn_path))
        {
            auto connectivity = readBoolMatrixFromFile(conn_path);
            if (!connectivity.empty())
            {
                conn_blob = boolMatrixToBlob(connectivity);
            }
        }

        std::string patterns_path = sim_dir + "/patterns.data";
        if (fs::exists(patterns_path))
        {
            auto patterns = loadPatterns(patterns_path);
            if (!patterns.empty())
            {
                patterns_blob = boolMatrixToBlob(patterns);
            }
        }

        // Insert simulation
        insertSimulation(db, sim_id, params_json, weights_blob, conn_blob, patterns_blob);

        // Insert results
        std::string results_path = sim_dir + "/results.data";
        if (fs::exists(results_path))
        {
            insertResultsFromCSV(db, sim_id, results_path);
        }

        count++;
        if (count % 100 == 0)
        {
            std::cout << "Processed " << count << " simulations..." << std::endl;
        }
    }

    // Commit transaction
    sqlite3_exec(db, "COMMIT", nullptr, nullptr, nullptr);

    sqlite3_close(db);
    std::cout << "Consolidated " << count << " simulations to: " << output_db << std::endl;
}

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <results_dir> [output.db]" << std::endl;
        std::cerr << std::endl;
        std::cerr << "Consolidates all sim_nb_X folders into a single SQLite database." << std::endl;
        std::cerr << "This creates a portable archive that's easy to move and query." << std::endl;
        return 1;
    }

    std::string results_dir = argv[1];
    std::string output_db = argc > 2 ? argv[2] : results_dir + "/experiment.db";

    if (!fs::exists(results_dir))
    {
        std::cerr << "Error: Results directory not found: " << results_dir << std::endl;
        return 1;
    }

    consolidate(results_dir, output_db);
    return 0;
}
