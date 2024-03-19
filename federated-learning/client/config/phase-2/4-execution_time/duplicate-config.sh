#!/bin/bash

# Define the directory containing the files. The DIRECTORY is set to the current directory "./"
DIRECTORY="./dreaddit"

# Navigate to the directory
cd "$DIRECTORY"

# Loop through all files matching the pattern "fl_*.json"
for FILE in fl_*.json; do
    # Check if the file exists to avoid error messages
    if [ -f "$FILE" ]; then
        # Generate the new file name by prepending "bc" to the original file name
        NEW_FILE_NAME="bc$FILE"
        
        # Duplicate the file with the new name, if it does not already exist
        if [ ! -f "$NEW_FILE_NAME" ]; then
            cp "$FILE" "$NEW_FILE_NAME"
        else
            echo "File $NEW_FILE_NAME already exists. Skipping."
        fi
    fi
done

echo "Files have been duplicated with the new naming convention."
