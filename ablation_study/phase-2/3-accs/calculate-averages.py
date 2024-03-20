import numpy as np


def custom_sort_key(model):
    """Custom sort function to order model names according to a specific sequence."""
    # Define the order for both prefixes and sizes
    prefix_order = {'ml': 1, 'fl': 2}
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
                if len(parts) == 2:
                    model, accuracy = parts
                    accuracy = float(accuracy.rstrip('%'))
                else:
                    model = parts[0]
                    accuracy = 0.0  # Default to 0 if accuracy value is missing
                base_model = '_'.join(model.split('_')[:2])  # Keep the base model name
                if base_model not in experiment_data[dataset]:
                    experiment_data[dataset][base_model] = []
                experiment_data[dataset][base_model].append(accuracy)

    # Writing results to the output file
    with open(output_file, 'w') as file:
        for dataset, models in experiment_data.items():
            file.write(f"Dataset: {dataset}\n")
            for model, accuracies in sorted(models.items(), key=lambda x: custom_sort_key(x[0])):
            # for model, accuracies in models.items():
                avg_acc = np.mean(accuracies)
                std_dev = np.std(accuracies, ddof=1)  # Use ddof=1 for sample standard deviation
                file.write(f"{model} : {avg_acc:.2f}% +- {std_dev:.2f}\n")
            file.write("\n")  # Extra newline for spacing between datasets

# Specify the input and output file paths
input_file_path = '/local/vpaloma/fabric-federated-learning/ablation_study/phase-2/3-accs/model_accuracy_summary.txt'
output_file_path = '/local/vpaloma/fabric-federated-learning/ablation_study/phase-2/3-accs/model_accuracy_averaged_summary.txt'

# Process the input file and write the results to the output file
process_data_file(input_file_path, output_file_path)