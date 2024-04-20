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
    # Define the order for both prefixes and sizes
    prefix_order = {'ml': 1, 'fl': 2, 'bcfl': 3}
    size_order = {'tiny': 1, 'mini': 2, 'small': 3, 'medium': 4}
    
    # Split the model name to extract its parts
    parts = model.split('_')
    # Extract the prefix (e.g., "ml" or "fl")
    prefix = parts[0]
    # Extract the size if available, or set a default value
    size = parts[1] if len(parts) > 1 else ''
    
    # Map the prefix and size to their respective orders, defaulting to a high number if not found
    prefix_order_val = prefix_order.get(prefix, 99)
    size_order_val = size_order.get(size, 99)
    
    # Return a tuple that represents the custom order
    return (prefix_order_val, size_order_val)

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
                base_model = '_'.join(model.split('_')[:2])
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
input_file_path = '/local/vpaloma/fabric-federated-learning/ablation_study/phase-4/final/accs-times/model_execution_time_summary.txt'
output_file_path = '/local/vpaloma/fabric-federated-learning/ablation_study/phase-4/final/accs-times/model_execution_time_averaged_summary.txt'

# Process the input file and write the results to the output file
process_data_file(input_file_path, output_file_path)