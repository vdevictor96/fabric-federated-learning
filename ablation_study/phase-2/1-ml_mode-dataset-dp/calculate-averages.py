import numpy as np

# Simulated input data
data = """
Dataset: acl_dep_sad
ml_0_acl_dep_sad, 93.54%
ml_0_seed2_acl_dep_sad, 93.85%
ml_0_seed3_acl_dep_sad, 94.31%
ml_10_acl_dep_sad, 85.69%
ml_10_seed2_acl_dep_sad, 81.54%
ml_10_seed3_acl_dep_sad, 87.08%
ml_1_acl_dep_sad, 80.46%
ml_1_seed2_acl_dep_sad, 81.85%
ml_1_seed3_acl_dep_sad, 84.00%
Dataset: dreaddit
ml_0_dreaddit, 73.99%
ml_0_seed2_dreaddit, 75.24%
ml_0_seed3_dreaddit, 75.66%
"""

# Parsing the input data
experiment_data = {}
for line in data.strip().split('\n'):
    if line.startswith('Dataset:'):
        dataset = line.split(': ')[1]
    else:
        model, accuracy = line.split(', ')
        accuracy = float(accuracy.rstrip('%'))
        base_model = '_'.join(model.split('_')[:2]) + '_' + dataset  # Keep the base model name and dataset
        if base_model not in experiment_data:
            experiment_data[base_model] = []
        experiment_data[base_model].append(accuracy)

# Calculating average accuracy and standard deviation
results = {}
for model, accuracies in experiment_data.items():
    avg_acc = np.mean(accuracies)
    std_dev = np.std(accuracies, ddof=1)
    results[model] = (avg_acc, std_dev)

# Prepare the result string for writing to a file
result_str = '\n'.join([f"{model} : {avg_acc:.2f}% +- {std_dev:.2f}" for model, (avg_acc, std_dev) in results.items()])
print(result_str)


def custom_sort_key(model):
    """Custom sort function to order model names according to a specific sequence."""
    order = {'0.5': 1, '1': 2, '10': 3, '0': 4}
    base, variant = model.split('_')[:2]
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
input_file_path = '/local/vpaloma/fabric-federated-learning/ablation_study/phase-2/1-ml_mode-dataset-dp/model_accuracy_summary.txt'
output_file_path = '/local/vpaloma/fabric-federated-learning/ablation_study/phase-2/1-ml_mode-dataset-dp/model_accuracy_averaged_summary.txt'

# Process the input file and write the results to the output file
process_data_file(input_file_path, output_file_path)