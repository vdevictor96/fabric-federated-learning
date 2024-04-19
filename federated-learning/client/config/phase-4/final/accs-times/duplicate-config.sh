#!/bin/bash

# Define the directory containing the files.
# Replace "/path/to/your/folder" with the actual path to your folder.
# DIRECTORY="./phase-2/1-ml_mode-dataset-dp/twitter_dep"
# DIRECTORY="./dreaddit"
DIRECTORY=$1


# Navigate to the directory
cd "$DIRECTORY"

for FILE in *_seed3_*.json; do
    # Check if the file exists to avoid error messages
    if [ -f "$FILE" ]; then
        # Loop to create duplicates with seed4 to seed10
        for SEED in {4..10}; do
            # Generate the new file name by replacing "seed3" with "seed$SEED"
            NEW_FILE_NAME=$(echo "$FILE" | sed "s/seed3/seed$SEED/")
            
            # Check if the new file name already exists
            if [ ! -f "$NEW_FILE_NAME" ]; then
                # Copy the file to the new file with the modified name
                cp "$FILE" "$NEW_FILE_NAME"
            else
                echo "File $NEW_FILE_NAME already exists. Skipping."
            fi
        done
    fi
done

echo "Files have been duplicated with the new naming convention."
