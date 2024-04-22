import numpy as np


def time_to_seconds(time_str):
    """Convert time string MM:SS to total seconds."""
    minutes, seconds = map(int, time_str.split(':'))
    return minutes * 60 + seconds

def seconds_to_time(seconds):
    """Convert total seconds back to MM:SS format."""
    minutes = seconds // 60
    seconds = seconds % 60
    return f"{int(minutes):02d}:{int(seconds):02d}"



def custom_sort_key(model):
    """Custom sort function to order model names according to a specific sequence."""
    order = {'c3': 1, 'c10': 2}
    base, model, variant = model.split('_')[:3]
    # Extract the number (variant) and map it to the custom order, default to 5 if not found
    return (base, order.get(variant, 5))

# Function to process the data file
def process_data_file(input_file, output_file):
    experiment_data = {}
    
    with open(input_file, 'r') as file:
        dataset = ""
        for line in file:
            line = line.strip()
            if line.startswith('Dataset:'):
                dataset = line.split(': ')[1].strip()
                if dataset not in experiment_data:
                    experiment_data[dataset] = {}
            elif line:
                parts = line.split(', ')
                model_time = parts[2] if len(parts) == 3 else '0:00'
                model, accuracy, time_seconds = parts[0], float(parts[1].rstrip('%')), time_to_seconds(model_time)
                base_model = '_'.join(model.split('_')[:3])
                print(base_model)
                if base_model not in experiment_data[dataset]:
                    experiment_data[dataset][base_model] = []
                experiment_data[dataset][base_model].append((accuracy, time_seconds))

    # Writing results to the output file
    with open(output_file, 'w') as file:
        for dataset, models in experiment_data.items():
            file.write(f"Dataset: {dataset}\n")
            for model, values in sorted(models.items(), key=lambda x: custom_sort_key(x[0])):
                accuracies, times = zip(*values)
                avg_acc = np.mean(accuracies)
                avg_time_seconds = np.mean(times)
                std_dev_acc = np.std(accuracies, ddof=1)
                std_dev_time_seconds = np.std(times, ddof=1)
                file.write(f"{model} : {avg_acc:.2f}% +- {std_dev_acc:.2f}, {seconds_to_time(avg_time_seconds)} +- {seconds_to_time(std_dev_time_seconds)}\n")
            file.write("\n")

# Specify the input and output file paths
input_file_path = '/local/vpaloma/fabric-federated-learning/ablation_study/phase-4/final/clients/model_execution_time_summary.txt'
output_file_path = '/local/vpaloma/fabric-federated-learning/ablation_study/phase-4/final/clients/model_execution_time_averaged_summary.txt'

# Process the input file and write the results to the output file
process_data_file(input_file_path, output_file_path)