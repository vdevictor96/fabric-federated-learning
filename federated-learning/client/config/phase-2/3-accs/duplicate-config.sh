#!/bin/bash

# Define the directory containing the files.
# Replace "./phase-2/3-accs/acl_dep_sad" with the actual path to your folder if different.
DIRECTORY="./acl_dep_sad"

# Navigate to the directory
cd "$DIRECTORY"

# Loop through all files matching the pattern "*.json"
for FILE in *.json; do
    # Check if the file exists to avoid error messages
    if [ -f "$FILE" ]; then
        # Generate the new file names by appending "seed2" and "seed3" before "_config.json"
        NEW_FILE_NAME_SEED2=$(echo "$FILE" | sed 's/_config.json/_seed2_config.json/')
        NEW_FILE_NAME_SEED3=$(echo "$FILE" | sed 's/_config.json/_seed3_config.json/')
        
        # Duplicate the file with the new names for seed2 and seed3, if they do not already exist
        if [ ! -f "$NEW_FILE_NAME_SEED2" ]; then
            cp "$FILE" "$NEW_FILE_NAME_SEED2"
        else
            echo "File $NEW_FILE_NAME_SEED2 already exists. Skipping."
        fi
        
        if [ ! -f "$NEW_FILE_NAME_SEED3" ]; then
            cp "$FILE" "$NEW_FILE_NAME_SEED3"
        else
            echo "File $NEW_FILE_NAME_SEED3 already exists. Skipping."
        fi
    fi
done

echo "Files have been duplicated with the new naming conventions."
