#!/bin/bash

# Define the directory containing the files using a positional parameter or specify the directory directly.
DIRECTORY=$1

# Navigate to the directory
cd "$DIRECTORY"

# Loop through all relevant files using a wildcard pattern that matches any "seed" or no "seed".
for FILE in bcfl_small*.json; do
    # Check if the file exists to avoid error messages
    if [ -f "$FILE" ]; then
        # Check for files with a "seed" label
        if [[ "$FILE" == *"_c10_"* ]]; then
            # Insert the new suffixes before "_seed" in the filename
            NEW_FILE1=$(echo "$FILE" | sed 's/_c10_seed/_c20_seed/')
            # NEW_FILE2=$(echo "$FILE" | sed 's/_seed/_c10_seed/')
        else
            # For files without a "seed" label, append suffixes before "_config.json"
            NEW_FILE1=$(echo "$FILE" | sed 's/_c10_config.json/_c20_config.json/')
            # NEW_FILE2=$(echo "$FILE" | sed 's/_config.json/_c20_config.json/')
        fi
        
        # Initialize a counter to track successful copies
        copy_count=0

        # Copy the original file to the new file names if they do not already exist
        for NEW_FILE in "$NEW_FILE1"; do
            if [ ! -f "$NEW_FILE" ]; then
                cp "$FILE" "$NEW_FILE"
                if [ $? -eq 0 ]; then
                    let copy_count++
                fi
            else
                echo "File $NEW_FILE already exists. Skipping."
            fi
        done

        # # If both copies were successful, remove the original file
        # if [ $copy_count -eq 2 ]; then
        #     rm "$FILE"
        #     echo "Original file $FILE removed after successful duplication."
        # fi
    else
        echo "File $FILE does not exist. Skipping."
    fi
done

echo "Files have been duplicated and original files removed where applicable."
