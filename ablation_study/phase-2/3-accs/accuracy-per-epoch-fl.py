import re
import os
from collections import defaultdict

# Define the path to the folder containing the files
folder_path = '/local/vpaloma/fabric-federated-learning/ablation_study/phase-2/3-accs/'

# Pattern to match lines with the required information
pattern = re.compile(r'Client (\d+) of \d+: Local Epoch \[(\d+)/\d+\] Loss: ([\d.]+), Accuracy: ([\d.]+) %')

# Pattern to match file names starting with "fl", ending with ".log", and not containing "seed"
# file_pattern = re.compile(r'^fl[^seed]*\.log$')


# Output file path
output_file_path = '/local/vpaloma/fabric-federated-learning/ablation_study/phase-2/3-accs/accuracy-per-epoch.txt'

# Open the output file in append mode
with open(output_file_path, 'a') as output_file:
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            if filename.startswith("fl") and filename.endswith(".log") and "seed" not in filename:
                file_path = os.path.join(dirpath, filename)
                if os.path.isfile(file_path):
                    # Reset aggregated_data for each file
                    aggregated_data = defaultdict(lambda: [0.0, 0.0, 0])
                    round_number = 0
                    with open(file_path, 'r') as file:
                        for line in file:
                            # Update round_number based on the lines preceding client details
                            if "Round" in line and "of" in line:
                                round_number = int(re.search(r'Round (\d+) of', line).group(1))
                            match = pattern.match(line)
                            if match:
                                client_number, local_epoch, loss, accuracy = match.groups()
                                epoch_index = ((round_number - 1) * 3) + int(local_epoch)  # Adjusting local_epoch to a global index
                                data = aggregated_data[epoch_index]
                                data[0] += float(loss)
                                data[1] += float(accuracy)
                                data[2] += 1

                    # Calculate and print averages for the current file
                    print(f'File: {filename}', file=output_file)
                    for epoch_index, (total_loss, total_accuracy, count) in sorted(aggregated_data.items()):
                        average_loss = total_loss / count
                        average_accuracy = total_accuracy / count
                        print(f'    Global Epoch {epoch_index}: Average Loss: {average_loss:.4f}, Average Accuracy: {average_accuracy:.2f} %', file=output_file)
                    print('---- End of File ----\n', file=output_file)