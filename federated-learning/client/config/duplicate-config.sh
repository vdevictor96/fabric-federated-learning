#!/bin/bash

# Define the directory containing the files.
# Replace "/path/to/your/folder" with the actual path to your folder.
DIRECTORY="./phase-2/1-ml_mode-dataset-dp/twitter_dep"

# Navigate to the directory
cd "$DIRECTORY"

# Loop through all files matching the pattern "fl_1_*.json"
for FILE in fl_1_*.json; do
    # Check if the file exists to avoid error messages
    if [ -f "$FILE" ]; then
        # Generate the new file name by replacing "fl_1_" with "fl_0.5_"
        NEW_FILE_NAME=$(echo "$FILE" | sed 's/fl_1_/fl_0.5_/')
        
         # Check if the new file name already exists
        if [ ! -f "$NEW_FILE_NAME" ]; then
            # Copy the file to the new file with the modified name
            cp "$FILE" "$NEW_FILE_NAME"
        else
            echo "File $NEW_FILE_NAME already exists. Skipping."
        fi
    fi
done

echo "Files have been duplicated with the new naming convention."
