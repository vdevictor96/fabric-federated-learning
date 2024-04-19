import os
import torch
import torch.nn as nn
import warnings
import numpy as np
import random
import sys
from torch.optim import AdamW
from transformers import get_scheduler
from .model.bert_tiny import get_bert_tiny_tokenizer, get_bert_tiny_model, load_bert_tiny_model
from .model.bert_mini import get_bert_mini_tokenizer, get_bert_mini_model, load_bert_mini_model
from .model.bert_small import get_bert_small_tokenizer, get_bert_small_model, load_bert_small_model
from .model.bert_medium import get_bert_medium_tokenizer, get_bert_medium_model, load_bert_medium_model
from .model.distilbert_base import get_distilbert_base_tokenizer, get_distilbert_base_model, load_distilbert_base_model
from .model.albert_base import get_albert_base_tokenizer, get_albert_base_model, load_albert_base_model
from .model.bert_base import get_bert_base_tokenizer, get_bert_base_model, load_bert_base_model
from .data.twitter_dep import get_twitter_dep_test_dataloader
from .data.acl_dep_sad import get_acl_dep_sad_test_dataloader
from .data.dreaddit import get_dreaddit_test_dataloader
from .data.mixed_depression import get_mixed_depression_test_dataloader
from .data.deptweet import get_deptweet_test_dataloader

# --- Model utils ---


def weights_init(m):
    if type(m) == nn.Linear:
        m.weight.data.normal_(0.0, 1e-3)
        m.bias.data.fill_(0.)


def weights_zero_init(m):
    for param in m.parameters():
        nn.init.zeros_(param)


def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# def translate_state_dict_keys(state_dict, keyword="_module.", replacement=""):
#     # replace the names in a state dict for all the keys that contain the keyword
#     new_state_dict = {}
#     for key in state_dict:
#         if keyword in key:
#             new_state_dict[key.replace(keyword, replacemente)] = state_dict[key]
#         else:
#             new_state_dict[key] = state_dict[key]
#     return new_state_dict

def translate_state_dict_keys(state_dict, keyword="_module.", replacement=""):
    keys_to_replace = [key for key in state_dict if keyword in key]

    for key in keys_to_replace:
        new_key = key.replace(keyword, replacement)
        state_dict[new_key] = state_dict.pop(key)

    return state_dict


def freeze_layers(model, layers_count):
    layers = []
    # Retrieve the last layers keys
    if hasattr(model, 'classifier'):
        layers.append(('classifier', model.classifier))
        layers_count -= 1
    if layers_count > 0 and hasattr(model.bert, 'pooler'):
        layers.append(('bert.pooler', model.bert.pooler))
        layers_count -= 1
    if layers_count > 0:
        total_encoder_layers = len(model.bert.encoder.layer)
        encoder_layers_to_add = min(
            layers_count, total_encoder_layers)
        start_index = total_encoder_layers - encoder_layers_to_add
        layers.extend([
            (f'bert.encoder.layer.{i}', layer)
            for i, layer in enumerate(model.bert.encoder.layer[-encoder_layers_to_add:], start=start_index)
        ])
    # layers_count -= encoder_layers_to_add
    # TODO TRAIN EMBEDDINGS?
    # while layers_count > 0:
    # if layers_count > 0:
    #     layers.append(model.bert.embeddings.LayerNorm)
    #     tra
    #     embedding_layers = [model.bert.embeddings.word_embeddings, model.bert.embeddings.position_embeddings, model.bert.embeddings.token_type_embeddings, model.bert.embeddings.LayerNorm]
    #     embeddings_layers_to_add = min(layers_count, len(embedding_layers))
    #     layers.extend(model.bert.embeddings[-embeddings_layers_to_add:])

    trainable_params = 0
    # Set requires_grad to False for all layers except the last ones
    for param in model.parameters():
        param.requires_grad = False
    for _, layer in layers:
        for param in layer.parameters():
            param.requires_grad = True
            trainable_params += param.numel()

    return trainable_params, layers


def filter_trainable_weights(state_dict, layers, dp):
    dp_prefix = '_module.'
    if layers is None:
        return state_dict

    # Extract layer names from the layers
    trainable_layer_names = [
        dp_prefix+name if dp else name for name, _ in layers]

    # Filter the state_dict to include only weights from the trainable layers
    trainable_state_dict = {name: weight for name, weight in state_dict.items()
                            if any(name.startswith(trainable_name) for trainable_name in trainable_layer_names)}

    # This is needed when applying DP with Opacus because the layer keys are modified
    # Translate the layers name back to the original model
    if dp:
        trainable_state_dict = translate_state_dict_keys(
            trainable_state_dict, dp_prefix, '')
    return trainable_state_dict


def get_trainable_state_dict_elements(model, num_layers):
    # Function to get parameter names for a given layer
    def get_param_names(key, layer):
        return [key+'.'+name for name, _ in layer.named_parameters()]

    # Collecting all layers' parameter names in reverse order
    all_layers = list(model.named_children())
    param_names = []
    layer_keys = []
    for key, layer in reversed(all_layers):
        if key not in layer_keys:
            layer_keys.append(key)
        param_names.extend(get_param_names(key, layer))
        if len(layer_keys) >= num_layers:
            break

    # Filtering the state_dict for the last N trainable layers
    trainable_state_dict = {
        name: param for name, param in model.state_dict().items() if name in param_names}

    return trainable_state_dict

# Example usage:
# trainable_state_dict = get_trainable_state_dict_elements(model, layers)


def print_parameters(model):
    for param in model.parameters():
        if param.grad is not None:
            print(param.grad.shape)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compare_models(model1, model2):
    for param1, param2 in zip(model1.parameters(), model2.parameters()):
        if param1.shape != param2.shape or not torch.equal(param1, param2):
            return False
    return True

# Function to compare two state_dicts


def compare_state_dicts(dict1, dict2):
    for key in dict1:
        if key not in dict2:
            return False
        if torch.all(torch.eq(dict1[key], dict2[key])) == False:
            return False
    return True


def compare_weights(model1, model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True

# --- Run utils ---

def set_random_seed(device='cuda'):
    seed = random.randint(0, 999999)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("seed set: ", seed)
    return seed
    
def set_seed(seed, device='cuda'):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("seed set: ", seed)


def set_device(device_name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        device = torch.device('cpu')
        if (device_name.startswith('cuda')):
            # if torch.cuda.is_available():
            if is_cuda_device_available(device_name):
                device = torch.device(device_name)
                print(f'{device_name} device selected and available.')
            elif torch.cuda.is_available():
                print(
                    f'{device_name} device given but it does not exist. cuda default device selected instead.')
                device = torch.device('cuda')
            else:
                print(
                    f'{device_name} device given but no CUDA devices available. Using CPU instead.')
        else:
            print('CPU device selected.')
    return device


def is_cuda_device_available(cuda_device_str):
    # Check if CUDA is available
    if not torch.cuda.is_available():
        return False
    if cuda_device_str == 'cuda' and torch.cuda.is_available():
        return True

    # Extract the device index from the string
    try:
        device_index = int(cuda_device_str.split(':')[1])
    except (IndexError, ValueError):
        # If the string format is incorrect or not a number
        return False

    # Check if the device_index is within the range of available devices
    return device_index >= 0 and device_index < torch.cuda.device_count()


def create_model(model_type, device):
    if model_type == 'bert_tiny':
        return get_bert_tiny_model(device)
    elif model_type == 'bert_mini':
        return get_bert_mini_model(device)
    elif model_type == 'bert_small':
        return get_bert_small_model(device)
    elif model_type == 'bert_medium':
        return get_bert_medium_model(device)
    # NOT SUPPORTED. TOO LARGE FOR BLOCKCHAIN-BASED FL
    # elif model_type == 'distilbert_base':
    #     return get_distilbert_base_model(device)
    # elif model_type == 'albert_base':
    #     return get_albert_base_model(device)
    # elif model_type == 'bert_base':
    #     return get_bert_base_model(device)
    else:
        raise ValueError(f"Unknown model type {model_type}.")


def load_model(model_type, model_path, device):
    if model_type == 'bert_tiny':
        return load_bert_tiny_model(model_path, device)
    elif model_type == 'bert_mini':
        return load_bert_mini_model(model_path, device)
    elif model_type == 'bert_small':
        return load_bert_small_model(model_path, device)
    elif model_type == 'bert_medium':
        return load_bert_medium_model(model_path, device)
    # NOT SUPPORTED. TOO LARGE FOR BLOCKCHAIN-BASED FL
    # elif model_type == 'distilbert_base':
    #     return load_distilbert_base_model(model_path, device)
    # elif model_type == 'albert_base':
    #     return load_albert_base_model(model_path, device)
    # elif model_type == 'bert_base':
    #     return load_bert_base_model(model_path, device)
    else:
        raise ValueError(f"Unknown model type {model_type}.")


def create_test_dataloader(dataset_type, tokenizer, test_batch_size, max_length, seed):
    if dataset_type == 'twitter_dep' or dataset_type == 'twitter_dep_balanced':
        dataset_path = get_dataset_path(dataset_type, 'test')
        return get_twitter_dep_test_dataloader(dataset_path, tokenizer, test_batch_size, max_length, seed)
    elif dataset_type == 'acl_dep_sad':
        dataset_path = get_dataset_path(dataset_type, 'test')
        return get_acl_dep_sad_test_dataloader(dataset_path, tokenizer, test_batch_size, max_length, seed)
    elif dataset_type == 'dreaddit':
        dataset_path = get_dataset_path(dataset_type, 'test')
        return get_dreaddit_test_dataloader(dataset_path, tokenizer, test_batch_size, max_length, seed)
    elif dataset_type == 'mixed_depression':
        dataset_path = get_dataset_path(dataset_type, 'test')
        return get_mixed_depression_test_dataloader(dataset_path, tokenizer, test_batch_size, max_length, seed)
    elif dataset_type == 'deptweet':
        dataset_path = get_dataset_path(dataset_type, 'test')
        return get_deptweet_test_dataloader(dataset_path, tokenizer, test_batch_size, max_length, seed)
    else:
        raise ValueError(f"Unknown dataset {dataset_type}.")


def create_tokenizer(model_type):
    if model_type == 'bert_tiny':
        return get_bert_tiny_tokenizer()
    elif model_type == 'bert_mini':
        return get_bert_mini_tokenizer()
    elif model_type == 'bert_small':
        return get_bert_small_tokenizer()
    elif model_type == 'bert_medium':
        return get_bert_medium_tokenizer()
    # NOT SUPPORTED. TOO LARGE FOR BLOCKCHAIN-BASED FL
    # elif model_type == 'distilbert_base':
    #     return get_distilbert_base_tokenizer()
    # elif model_type == 'albert_base':
    #     return get_albert_base_tokenizer()
    # elif model_type == 'bert_base':
    #     return get_bert_base_tokenizer()
    else:
        raise ValueError(f"Unknown tokenizer for model type {model_type}.")


def create_optimizer(optimizer_type, model, lr):
    if optimizer_type.lower() == 'adamw':
        return AdamW(model.parameters(), lr=lr, eps=1e-8)
    else:
        raise ValueError(f"Unknown optimizer {optimizer_type}.")


def create_scheduler(scheduler_type, optimizer, num_training_steps, num_warmup_steps):
    if scheduler_type.lower() == 'linear':
        return get_scheduler(name='linear', optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
    else:
        raise ValueError(f"Unknown scheduler {scheduler_type}.")


def get_dir_path():
    # Get the directory of the current script
    return os.path.dirname(os.path.realpath(__file__))


def get_dataset_path(dataset_name, dataset_type='train'):
    # Get the directory of the current script
    dir_path = get_dir_path()
    balanced = False
    if (dataset_name == 'twitter_dep_balanced'):
            dataset_name = 'twitter_dep'
            balanced = True
    if (dataset_type == 'train'):
        if balanced:
            dataset_path = os.path.join(
                dir_path, "data", "datasets", dataset_name, dataset_name + "_train_balanced.csv")
        else:
            dataset_path = os.path.join(
                dir_path, "data", "datasets", dataset_name, dataset_name + "_train.csv")
    elif (dataset_type == 'test'):
        dataset_path = os.path.join(
            dir_path, "data", "datasets", dataset_name, dataset_name + "_test.csv")
    else:
        raise ValueError(f"Unknown dataset type {dataset_type}.")
    # Check if the file exists
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"The dataset file does not exist: {dataset_path}")

    return dataset_path


def iid_partition(dataset, clients):
    dataset_length = len(dataset)

    # Create an array of indexes from 0 to dataset_length and shuffle it
    indexes = list(range(dataset_length))
    random.shuffle(indexes)

    # Calculate the size of each partition
    partition_size = dataset_length // clients

    # Initialize the dictionary to hold the partitions
    partitions = {}

    for i in range(clients):
        # For each client, assign a slice of the indexes array
        start_index = i * partition_size
        end_index = start_index + partition_size

        # If it's the last client, include any remaining indexes
        if i == clients - 1:
            end_index = dataset_length

        partitions[i] = indexes[start_index:end_index]
    labels = np.array(dataset.get_labels())
    # Print the distribution of labels for each client
    for client in partitions:
        label_0_count = sum(labels[partitions[client]] == 0)
        label_1_count = sum(labels[partitions[client]] == 1)
        print(f"Client {client}: Label 0: {label_0_count}, Label 1: {label_1_count}")

    return partitions

def non_iid_partition_varying_samples(dataset, clients, max_label_ratio=0.2, sample_allocation_strategy=None):
    labels = np.array(dataset.get_labels())
    indices_label_0 = np.where(labels == 0)[0]
    indices_label_1 = np.where(labels == 1)[0]

    np.random.shuffle(indices_label_0)
    np.random.shuffle(indices_label_1)

    # If no specific strategy is provided, use a simple random allocation
    if sample_allocation_strategy is None:
        sample_allocation_strategy = np.random.rand(clients)
        sample_allocation_strategy /= sample_allocation_strategy.sum()  # Normalize

    total_samples = len(labels)
    total_label_0 = len(indices_label_0)
    total_label_1 = len(indices_label_1)
    partitions = {i: [] for i in range(clients)}
    min_samples_per_label = 0

    allocated_indices_0 = 0
    allocated_indices_1 = 0

        # Initial allocation of min_samples_per_label of each label to every client
    for i in range(clients):
        partitions[i].extend(indices_label_0[i * min_samples_per_label:(i + 1) * min_samples_per_label])
        partitions[i].extend(indices_label_1[i * min_samples_per_label:(i + 1) * min_samples_per_label])
    
    # Update indices to remove those already allocated
    indices_label_0 = indices_label_0[clients * min_samples_per_label:]
    indices_label_1 = indices_label_1[clients * min_samples_per_label:]
    
    for i in range(clients):
        # Calculate samples for each client based on the strategy
        client_samples = int(sample_allocation_strategy[i] * total_samples)
        max_label_0_samples = int(max_label_ratio * client_samples)
        max_label_1_samples = client_samples - max_label_0_samples
        # max_label_1_samples = int(max_label_ratio * client_samples)

        # Allocate primarily label 0 to the client up to max_label_ratio
        label_0_alloc = min(len(indices_label_0) - allocated_indices_0, max_label_0_samples)
        partitions[i].extend(indices_label_0[allocated_indices_0:allocated_indices_0 + label_0_alloc])
        allocated_indices_0 += label_0_alloc

        # Fill the rest with label 1
        label_1_alloc = min(client_samples - label_0_alloc, len(indices_label_1) - allocated_indices_1)
        partitions[i].extend(indices_label_1[allocated_indices_1:allocated_indices_1 + label_1_alloc])
        allocated_indices_1 += label_1_alloc

    # Allocate any remaining samples of each label
    remaining_indices_label_0 = np.setdiff1d(indices_label_0, np.concatenate([partitions[i] for i in partitions]))
    remaining_indices_label_1 = np.setdiff1d(indices_label_1, np.concatenate([partitions[i] for i in partitions]))

    # Distribute remaining label 0 samples
    np.random.shuffle(remaining_indices_label_0)
    for i, idx in enumerate(remaining_indices_label_0):
        partitions[i % clients].append(idx)

    # Distribute remaining label 1 samples
    np.random.shuffle(remaining_indices_label_1)
    for i, idx in enumerate(remaining_indices_label_1):
        partitions[(i + len(remaining_indices_label_0)) % clients].append(idx)
        
    # Shuffle partitions to ensure randomness
    for client in partitions:
        np.random.shuffle(partitions[client])

    # Print distribution of labels for each client
    total_counted_samples = 0
    total_counted_label_0 = 0
    total_counted_label_1 = 0
    for client in partitions:
        label_0_count = sum(labels[partitions[client]] == 0)
        label_1_count = sum(labels[partitions[client]] == 1)
        total_counted_samples += label_0_count + label_1_count
        total_counted_label_0 += label_0_count
        total_counted_label_1 += label_1_count
        print(f"Client {client}: Label 0: {label_0_count}, Label 1: {label_1_count}")
    print(f"Total samples: {total_samples}")
    print(f"Total counted samples: {total_counted_samples}")
    print(f"Total label 0: {total_label_0}")
    print(f"Total label 1: {total_label_1}")
    print(f"Total counted label 0: {total_counted_label_0}")
    print(f"Total counted label 1: {total_counted_label_1}")

    return partitions



def non_iid_partition(dataset, clients, max_label_ratio=0.2):
    labels = np.array(dataset.get_labels())
    indices_label_0 = np.where(labels == 0)[0]
    indices_label_1 = np.where(labels == 1)[0]

    # Halve the data for each label
    # debugging fedprox in benchmarking
    # indices_label_0 = indices_label_0[:len(indices_label_0) // 2]
    # indices_label_1 = indices_label_1[:len(indices_label_1) // 2]
    
    np.random.shuffle(indices_label_0)
    np.random.shuffle(indices_label_1)

    partitions = {i: [] for i in range(clients)}
    # total_samples = len(indices_label_0) + len(indices_label_1)  # Adjusted total samples
    # samples_per_client = total_samples // clients
    # print(total_samples)
    # max_label_0_samples = int(max_label_ratio * samples_per_client)
    # max_label_1_samples = samples_per_client - max_label_0_samples
    total_samples = len(labels)
    samples_per_client = total_samples // clients
    max_label_0_samples = int(max_label_ratio * samples_per_client)
    max_label_1_samples = int(max_label_ratio * samples_per_client)
    # max_label_1_samples = samples_per_client - max_label_0_samples

    # Split clients into two groups
    half_clients = clients // 2

    # Allocate primarily label 0 to the first half
    for i in range(half_clients):
        # Allocate label 0 up to max_label_ratio
        label_0_alloc = min(len(indices_label_0), max_label_0_samples)
        partitions[i].extend(indices_label_0[:label_0_alloc])
        indices_label_0 = indices_label_0[label_0_alloc:]

        # Fill the rest with label 1
        remaining_slots = samples_per_client - label_0_alloc
        label_1_alloc = min(len(indices_label_1), remaining_slots)
        partitions[i].extend(indices_label_1[:label_1_alloc])
        indices_label_1 = indices_label_1[label_1_alloc:]

    # Allocate primarily label 1 to the second half
    for i in range(half_clients, clients):
        # Allocate label 1 up to max_label_ratio
        label_1_alloc = min(len(indices_label_1), max_label_1_samples)
        partitions[i].extend(indices_label_1[:label_1_alloc])
        indices_label_1 = indices_label_1[label_1_alloc:]

        # Fill the rest with label 0
        remaining_slots = samples_per_client - label_1_alloc
        label_0_alloc = min(len(indices_label_0), remaining_slots)
        partitions[i].extend(indices_label_0[:label_0_alloc])
        indices_label_0 = indices_label_0[label_0_alloc:]

    # Ensure all indices are allocated by distributing any remaining samples evenly
    # This step is crucial if the total samples aren't perfectly divisible by clients or if max_label_ratio constraints lead to unallocated samples
    remaining_indices = np.concatenate((indices_label_0, indices_label_1))
    np.random.shuffle(remaining_indices)
    while len(remaining_indices) > 0:
        for i in range(clients):
            if len(remaining_indices) == 0:
                break
            partitions[i].append(remaining_indices[0])
            remaining_indices = remaining_indices[1:]

    # Shuffle partitions to ensure randomness
    for client in partitions:
        np.random.shuffle(partitions[client])

    # Print the distribution of labels for each client
    for client in partitions:
        label_0_count = sum(labels[partitions[client]] == 0)
        label_1_count = sum(labels[partitions[client]] == 1)
        print(f"Client {client}: Label 0: {label_0_count}, Label 1: {label_1_count}")

    return partitions


# ------------------ Output utils ------------------

# Function to duplicated output to a file
def duplicate_output_to_file(filename):
    sys.stdout = DualOutput(filename)


class DualOutput:
    def __init__(self, filename):
        self.file = open(filename, 'w')
        self.stdout = sys.stdout

    def write(self, message):
        self.file.write(message)
        self.stdout.write(message)

    def flush(self):  # Needed for compatibility with the flush method of sys.stdout
        self.file.flush()
        self.stdout.flush()

    def close(self):  # Method to close the file when done
        self.file.close()