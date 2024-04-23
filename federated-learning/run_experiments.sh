#!/bin/bash

# Assign the first command line argument to a variable
CONFIG_DIR=$1
# CONFIG_DIR="/local/vpaloma/fabric-federated-learning/federated-learning/client/config/phase-2/1-ml_mode-dataset-dp/mixed_depression/"


# Navigate to the root directory of your project
cd /local/vpaloma/fabric-federated-learning/federated-learning

# Find all .json files in the ./client/config/ directory and its subdirectories
# For each file found, execute the python command in parallel
# find "$CONFIG_DIR" -type f -name "*.json" | while read config_file; do
# ! -name "bcfl*"
# find "$CONFIG_DIR" -type f -name "fl_0.5*.json"  ! -name "fl_1_*" ! -name "fl_0_*" | while read config_file; do
# find "$CONFIG_DIR" -type f -name "ml_*.json"  ! -name "ml_1_*" ! -name "ml_0_*" | while read config_file; do


# find "$CONFIG_DIR" -type f -name "fl*.json" | while read config_file; do
#     # if [[ "$config_file" != *"/twitter_dep"* ]]; then

# # find "$CONFIG_DIR" -type f -regextype posix-extended -regex ".*\/fl_0\..*\.json|.*\/fl_1.*\.json" | while read config_file; do

#     echo "Executing config file: $config_file"
#     # parallel execution
#     # python -m client.run_train --config_file "$config_file" &
#     # sequential execution (for execution time ablation study)
#     python -m client.run_train --config_file "$config_file"
#     # fi
# done

# Loop through all .json config files in the specified directory
find "$CONFIG_DIR" -type f -name "*.json" | while read config_file; do
# find "$CONFIG_DIR" -type f -name "*.json" ! -name "*_seed3_*.json" ! -name "*_seed4_*.json" ! -name "*_seed5_*.json" ! -name "*_seed6_*.json" ! -name "*_seed7_*.json" ! -name "*_seed8_*.json" ! -name "*_seed9_*.json" ! -name "*_seed10_*.json" | while read config_file; do

    # Extract base name without the "_config.json" part
    base_name=$(basename "$config_file" "_config.json")
    
    # Extract the dataset name from the path, assuming it's the name of the subdirectory in the config path
    dataset_name=$(basename $(dirname "$config_file"))
    
    # Construct the log file path by including the dataset name in the log filename
    log_file=$(echo "$config_file" | sed -e "s#./client/config/#../ablation_study/#" -e "s#_config.json#_${dataset_name}.log#")
    echo $log_file
    # Check if the output log file exists
    if [[ -f "$log_file" ]]; then
        # echo "Skipping execution for: $config_file (Output log exists)"
        echo "Executing config file: $config_file"
        # Run the Python command for training
        python -m client.run_train --config_file "$config_file" 
    else
        echo "Executing config file: $config_file"
        # Run the Python command for training
        python -m client.run_train --config_file "$config_file" 
    fi
done


# Wait for all background processes to finish
wait


# if you need to kill all experiments
# pkill -f "python -m client.run_train"
# killall -r "python -m client.run_train"
