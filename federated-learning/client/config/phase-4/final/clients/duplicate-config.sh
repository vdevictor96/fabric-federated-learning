#!/bin/bash

# Define the directory containing the files.
# Use a positional parameter or specify the directory directly.
DIRECTORY=$1

# Navigate to the directory
cd "$DIRECTORY"

# Loop through all relevant files using a wildcard pattern that matches any "seed" or no "seed".
for FILE in fl_small*.json; do
    # Check if the file exists to avoid error messages
    if [ -f "$FILE" ]; then
        # Handle files with a "seed" label
        if [[ "$FILE" == *"_seed"* ]]; then
            # Extract the part before "_config.json" to append suffixes correctly
            BASE_NAME=${FILE%_config.json}
            
            # Generate new file names by appending suffixes
            NEW_FILE1="${BASE_NAME}_c3_config.json"
            NEW_FILE2="${BASE_NAME}_c10_config.json"
        else
            # For files without a "seed" label
            NEW_FILE1=$(echo "$FILE" | sed "s/_config.json/_c3_config.json/")
            NEW_FILE2=$(echo "$FILE" | sed "s/_config.json/_c10_config.json/")
        fi
        
        # Initialize a counter to track successful copies
        copy_count=0

        # Copy the original file to the new file names if they do not already exist
        for NEW_FILE in "$NEW_FILE1" "$NEW_FILE2"; do
            if [ ! -f "$NEW_FILE" ]; then
                cp "$FILE" "$NEW_FILE"
                if [ $? -eq 0 ]; then
                    let copy_count++
                fi
            else
                echo "File $NEW_FILE already exists. Skipping."
            fi
        done

        # If both copies were successful, remove the original file
        if [ $copy_count -eq 2 ]; then
            rm "$FILE"
            echo "Original file $FILE removed after successful duplication."
        fi
    else
        echo "File $FILE does not exist. Skipping."
    fi
done

echo "Files have been duplicated and original files removed where applicable."
