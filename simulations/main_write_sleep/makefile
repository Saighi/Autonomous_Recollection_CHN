CC = g++
CFLAGS = -pthread -g -Wall -std=c++17 -O3 -march=native
INCLUDE_DIR = include
SRC_DIR = src
SIM_DIR = simulations
OBJ_DIR = obj
BIN_DIR = bin

# Core library object files
CORE_OBJS = $(OBJ_DIR)/network.o $(OBJ_DIR)/utils.o $(OBJ_DIR)/discrete_hopfield.o

# Auto-discover simulation sources and derive executable names
SIM_SOURCES = $(wildcard $(SIM_DIR)/*.cc)
SIM_NAMES = $(basename $(notdir $(SIM_SOURCES)))
SIM_BINS = $(addprefix $(BIN_DIR)/, $(SIM_NAMES))

# Default: build all simulations
all: dirs $(SIM_BINS)

# Create directories
dirs:
	@mkdir -p $(OBJ_DIR) $(BIN_DIR)

# Compile core library
$(OBJ_DIR)/network.o: $(SRC_DIR)/network.cc $(INCLUDE_DIR)/network.hpp
	$(CC) -c $(CFLAGS) $< -o $@ -I$(INCLUDE_DIR)

$(OBJ_DIR)/utils.o: $(SRC_DIR)/utils.cc $(INCLUDE_DIR)/utils.hpp $(INCLUDE_DIR)/network.hpp
	$(CC) -c $(CFLAGS) $< -o $@ -I$(INCLUDE_DIR)

$(OBJ_DIR)/discrete_hopfield.o: $(SRC_DIR)/discrete_hopfield.cc $(INCLUDE_DIR)/discrete_hopfield.hpp
	$(CC) -c $(CFLAGS) $< -o $@ -I$(INCLUDE_DIR)

# Build any simulation: bin/X from simulations/X.cc
$(BIN_DIR)/%: $(SIM_DIR)/%.cc $(CORE_OBJS)
	$(CC) $(CFLAGS) $< -o $@ -I$(INCLUDE_DIR) $(CORE_OBJS)

# Special rule for consolidate (requires SQLite)
$(BIN_DIR)/consolidate: $(SIM_DIR)/consolidate.cc $(CORE_OBJS)
	$(CC) $(CFLAGS) $< -o $@ -I$(INCLUDE_DIR) $(CORE_OBJS) -lsqlite3

# Individual targets for convenience
consolidate: $(BIN_DIR)/consolidate

write: $(BIN_DIR)/write

sleep: $(BIN_DIR)/sleep

query: $(BIN_DIR)/query

dhn_train: $(BIN_DIR)/dhn_train

dhn_query: $(BIN_DIR)/dhn_query

# Run targets
run_write: $(BIN_DIR)/write
	./$(BIN_DIR)/write

run_sleep: $(BIN_DIR)/sleep
	./$(BIN_DIR)/sleep

# Clean
clean:
	rm -rf $(OBJ_DIR)/*.o $(BIN_DIR)/*

# Deep clean (removes directories too)
distclean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

# List available simulations
list:
	@echo "Available simulations: $(SIM_NAMES)"
	@echo "Executables will be in: $(BIN_DIR)/"

# Help
help:
	@echo "Usage:"
	@echo "  make            - Build all simulations"
	@echo "  make write      - Build only write simulation (CHN training)"
	@echo "  make sleep      - Build only sleep simulation (CHN retrieval)"
	@echo "  make query      - Build only query simulation (CHN partial cue)"
	@echo "  make dhn_train  - Build DHN training (Hebbian/Storkey)"
	@echo "  make dhn_query  - Build DHN partial cue query"
	@echo "  make consolidate- Build consolidation tool (requires libsqlite3-dev)"
	@echo "  make bin/X      - Build specific simulation X"
	@echo "  make run_write  - Build and run write simulation"
	@echo "  make run_sleep  - Build and run sleep simulation"
	@echo "  make list       - List available simulations"
	@echo "  make clean      - Remove object files and executables"
	@echo "  make distclean  - Remove obj and bin directories"
	@echo ""
	@echo "To add a new simulation:"
	@echo "  1. Create simulations/your_sim.cc"
	@echo "  2. Run 'make' or 'make bin/your_sim'"
	@echo ""
	@echo "Consolidation tool:"
	@echo "  ./bin/consolidate <results_dir> [output.db]"
	@echo "  Merges all sim_nb_X folders into a single SQLite database."

.PHONY: all dirs write sleep query dhn_train dhn_query consolidate run_write run_sleep clean distclean list help
