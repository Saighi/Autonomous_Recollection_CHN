#!/bin/bash
#SBATCH --job-name=mpi_simulations              # Job name
#SBATCH --output=job_output_%j.out              # Standard output and error log
#SBATCH --error=job_error_%j.err
#SBATCH --nodes=1                               # Number of nodes
#SBATCH --ntasks-per-node=10                    # Number of tasks per node (adjust as needed)
#SBATCH --time=02:00:00                         # Time limit hrs:min:sec (adjust as needed)
#SBATCH --partition=compute                     # Partition name (adjust based on your cluster)

# Load necessary modules (adjust based on your cluster's module system)
module load mpi/openmpi

# Variables
SIM_NAME="Fig_load_SR_average_MPI_test"
PATH_RESULTS_WRITE="../../../data/all_data_splited/trained_networks_fast/"
PATH_RESULTS_SLEEP="../../../data/all_data_splited/sleep_simulations/"

# Run make and make run-auto in the first subfolder
echo "Starting tasks in the first subfolder..."
make -C ../simulations/Fig_load_SR_average_MPI clean
make -C ../simulations/Fig_load_SR_average_MPI
if [ $? -ne 0 ]; then
    echo "Compilation failed in the first subfolder."
    exit 1
fi

make -C ../simulations/Fig_load_SR_average_MPI run-auto SIM_NAME="$SIM_NAME" PATH_RESULTS="$PATH_RESULTS_WRITE" MPIRUN="srun"
if [ $? -ne 0 ]; then
    echo "An error occurred during execution in the first subfolder."
    exit 1
fi

# Run make and make run-auto in the second subfolder
echo "Starting tasks in the second subfolder..."
make -C ../simulations/Fig_load_SR_average_sleep_MPI clean
make -C ../simulations/Fig_load_SR_average_sleep_MPI
if [ $? -ne 0 ]; then
    echo "Compilation failed in the second subfolder."
    exit 1
fi

make -C ../simulations/Fig_load_SR_average_sleep_MPI run-auto SIM_NAME="$SIM_NAME" PATH_INPUTS="$PATH_RESULTS_WRITE" PATH_RESULTS="$PATH_RESULTS_SLEEP" MPIRUN="srun"
if [ $? -ne 0 ]; then
    echo "An error occurred during execution in the second subfolder."
    exit 1
fi

echo "All tasks completed successfully."
