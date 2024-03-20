#!/bin/bash

# Assign the first command line argument to a variable
CONFIG_DIR=$1
# CONFIG_DIR="/local/vpaloma/fabric-federated-learning/federated-learning/client/config/phase-2/1-ml_mode-dataset-dp/mixed_depression/"


# Navigate to the root directory of your project
cd /local/vpaloma/fabric-federated-learning/federated-learning

# Find all .json files in the ./client/config/ directory and its subdirectories
# For each file found, execute the python command in parallel
# find "$CONFIG_DIR" -type f -name "*.json" | while read config_file; do
find "$CONFIG_DIR" -type f -name "*.json" ! -name "bcfl*" | while read config_file; do

# find "$CONFIG_DIR" -type f -name "fed*.json" | while read config_file; do
# find "$CONFIG_DIR" -type f -regextype posix-extended -regex ".*\/fl_0\..*\.json|.*\/fl_1.*\.json" | while read config_file; do

    echo "Executing config file: $config_file"
    # parallel execution
    # python -m client.run_train --config_file "$config_file" &
    # sequential execution (for execution time ablation study)
    python -m client.run_train --config_file "$config_file"
done

# Wait for all background processes to finish
wait


# if you need to kill all experiments
# pkill -f "python -m client.run_train"
# killall -r "python -m client.run_train"
