#!/bin/bash

# Define the base directory
# BASE_DIR="/local/vpaloma/fabric-federated-learning/ablation_study/phase-2/1-ml_mode-dataset-dp"
BASE_DIR="/local/vpaloma/fabric-federated-learning/ablation_study/phase-2/2-fed-data_dist-dataset"


# List of folders to read files from
# FOLDERS=("dreaddit")

FOLDERS=("acl_dep_sad" "mixed_depression" "dreaddit"  "twitter_dep")
# "dreaddit"  "twitter_dep"


# Output file for model names and accuracy values
OUTPUT_FILE="${BASE_DIR}/model_accuracy_summary.txt"
PYTHON_FILE="${BASE_DIR}/calculate-averages.py"

# Check if the output file already exists and remove it to start fresh
if [ -f "$OUTPUT_FILE" ]; then
    rm "$OUTPUT_FILE"
fi

# Navigate to the base directory
cd "$BASE_DIR"

# Function to extract model name and accuracy, then add to array
process_log_files_in_folder() {
    local folder_path=$1
    # Array to hold model_name and accuracy lines for the current folder
    declare -a MODEL_ACCURACY_LINES=()

    # Find all .log files in the folder
    LOG_FILES=$(find "$folder_path" -type f -name "*.log")
    for LOG_FILE in $LOG_FILES; do
        local model_name=$(grep '"model_name":' "$LOG_FILE" | awk -F'"' '{print $4}')
        local accuracy=$(grep 'Accuracy: ' "$LOG_FILE" | tail -1 | awk '{print $2}')
        MODEL_ACCURACY_LINES+=("${model_name}, ${accuracy}")
    done

    # Sort the array alphabetically for the current folder and write to the output file
    echo -e "Dataset: $(basename "$folder_path")\n" >> "$OUTPUT_FILE"
    printf "%s\n" "${MODEL_ACCURACY_LINES[@]}" | sort >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE" # Adding a newline for separation
}

# Iterate over each folder and process its .log files
for FOLDER in "${FOLDERS[@]}"; do
    FOLDER_PATH="${BASE_DIR}/${FOLDER}"
    if [ -d "$FOLDER_PATH" ]; then
        process_log_files_in_folder "$FOLDER_PATH"
    else
        echo "Folder not found: $FOLDER"
    fi
done


python "$PYTHON_FILE"