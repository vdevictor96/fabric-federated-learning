import re
import os
from collections import defaultdict

# Define the path to the folder containing the files
folder_path = '/local/vpaloma/fabric-federated-learning/ablation_study/phase-2/3-accs/'

# Pattern to match lines with epoch loss and accuracy information
pattern = re.compile(r'Epoch \[(\d+)/\d+\] Loss: ([\d.]+), Accuracy: ([\d.]+) %')

# Pattern to match file names starting with "ml", ending with ".log", and not containing "seed"
# file_pattern = re.compile(r'^(ml)[^seed]*\.log$')

# Output file path
output_file_path = '/local/vpaloma/fabric-federated-learning/ablation_study/phase-2/3-accs/accuracy-per-epoch.txt'

# Open the output file in append mode
with open(output_file_path, 'a') as output_file:
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            # print(f"Found file: {filename}")  # Debug print
            # print(filename.startswith("ml") and filename.endswith(".log") and "seed" not in filename)
            if filename.startswith("ml") and filename.endswith(".log") and "seed" not in filename:
                file_path = os.path.join(dirpath, filename)
                if os.path.isfile(file_path):
                    print(f'File: {filename}', file=output_file)
                    with open(file_path, 'r') as file:
                        for line in file:
                            match = pattern.match(line)
                            if match:
                                epoch, loss, accuracy = match.groups()
                                print(f'    Epoch [{epoch}/12] Loss: {loss}, Accuracy: {accuracy} %', file=output_file)
                    print('---- End of File ----\n', file=output_file)