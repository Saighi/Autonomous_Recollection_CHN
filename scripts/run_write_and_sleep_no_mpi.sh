# Run make and make run-auto in the first subfolder
make -C ../simulations/Fig_load_SR_average
# make -C ../simulations/Fig_load_SR_average run
# # Check if the previous commands were successful
# if [ $? -ne 0 ]; then
#     echo "An error occurred in the first subfolder."
#     exit 1
# fi
# # Run make and make run-auto in the second subfolder
# make -C ../simulations/Fig_load_SR_average_sleep make clean  # Clean any previous builds
# make -C ../simulations/Fig_load_SR_average_sleep
# make -C ../simulations/Fig_load_SR_average_sleep run
# # Check if the previous commands were successful
# if [ $? -ne 0 ]; then
#     echo "An error occurred in the second subfolder."
#     exit 1
# fi
# echo "All tasks completed successfully."