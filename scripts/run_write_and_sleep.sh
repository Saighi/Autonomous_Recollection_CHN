SIM_NAME="Fig_load_SR_average_100N"
# SIM_NAME="Fig_load_SR_average_MPI_test_master_worker"
PATH_RESULTS_WRITE="../../../data/all_data_splited/trained_networks_fast/"
PATH_RESULTS_SLEEP="../../../data/all_data_splited/sleep_simulations/"
# Run make and make run-auto in the first subfolder
make -C ../simulations/Fig_load_SR_average_MPI
make -C ../simulations/Fig_load_SR_average_MPI run-auto SIM_NAME="$SIM_NAME" PATH_RESULTS="$PATH_RESULTS_WRITE"
# Check if the previous commands were successful
if [ $? -ne 0 ]; then
    echo "An error occurred in the first subfolder."
    exit 1
fi

# Run make and make run-auto in the second subfolder
make -C ../simulations/Fig_load_SR_average_sleep_MPI make clean  # Clean any previous builds
make -C ../simulations/Fig_load_SR_average_sleep_MPI
make -C ../simulations/Fig_load_SR_average_sleep_MPI run-auto SIM_NAME="$SIM_NAME" PATH_INPUTS="$PATH_RESULTS_WRITE" PATH_RESULTS="$PATH_RESULTS_SLEEP"

# Check if the previous commands were successful
if [ $? -ne 0 ]; then
    echo "An error occurred in the second subfolder."
    exit 1
fi

echo "All tasks completed successfully."
