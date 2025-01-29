CC=mpic++
MPIRUN=/usr/bin/mpirun
SRC := $(shell pwd)/src
INCLUDE_DIR := $(shell pwd)/include/
CFLAGS=-pthread -g -Wall -std=c++17
NPROC=10

all : main

main: obj $(SRC)/simulations.cc $(SRC)/utils.cc $(SRC)/network.cc obj/network.o obj/utils.o 
	$(CC) $(CFLAGS) $(SRC)/simulations.cc -o simulations -I$(INCLUDE_DIR) ./obj/*.o

obj: 
	if [ ! -d "./obj" ]; then mkdir obj; fi

obj/network.o : $(SRC)/network.cc
	$(CC) -c $(CFLAGS) $(SRC)/network.cc -o obj/network.o -I$(INCLUDE_DIR)

obj/utils.o : $(SRC)/utils.cc
	$(CC) -c $(CFLAGS) $(SRC)/utils.cc -o obj/utils.o -I$(INCLUDE_DIR)

run: simulations
	$(MPIRUN) -np $(NPROC) ./simulations

# Show cores then run
run-auto: simulations
	@echo "Available cores: $$(nproc)"
	@echo "MPI will use: $$($(MPIRUN) --use-hwthread-cpus hostname | wc -l) processes"
	@echo "Running with SIM_NAME=$(SIM_NAME), PATH_INPUTS=$(PATH_INPUTS) and PATH_RESULTS=$(PATH_RESULTS)"
	$(MPIRUN) --mca btl_vader_single_copy_mechanism none --use-hwthread-cpus ./simulations $(SIM_NAME) $(PATH_INPUTS) $(PATH_RESULTS)

clean:
	rm -rf obj simulations

.PHONY: all clean obj run

# New target for SLURM
run-slurm: simulations
	@echo "Running slurm with SIM_NAME=$(SIM_NAME), PATH_INPUTS=$(PATH_INPUTS) and PATH_RESULTS=$(PATH_RESULTS)"
	srun --mpi=pmix_v3 ./simulations $(SIM_NAME) $(PATH_INPUTS) $(PATH_RESULTS)