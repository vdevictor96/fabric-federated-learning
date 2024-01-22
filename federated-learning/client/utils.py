import os
import torch
import torch.nn as nn
import warnings
import numpy as np
import random
from torch.optim import AdamW
from transformers import get_scheduler
from .model.bert_tiny import get_bert_tiny_tokenizer, get_bert_tiny_model, load_bert_tiny_model
from .model.bert_mini import get_bert_mini_tokenizer, get_bert_mini_model, load_bert_mini_model
from .model.bert_small import get_bert_small_tokenizer, get_bert_small_model, load_bert_small_model
from .model.bert_medium import get_bert_medium_tokenizer, get_bert_medium_model, load_bert_medium_model
from .model.distilbert_base import get_distilbert_base_tokenizer, get_distilbert_base_model, load_distilbert_base_model
from .model.albert_base import get_albert_base_tokenizer, get_albert_base_model, load_albert_base_model
from .model.bert_base import get_bert_base_tokenizer, get_bert_base_model, load_bert_base_model


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


def set_seed(seed, device='cuda'):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_device(device_name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        device = torch.device('cpu')
        if (device_name == 'cuda'):
            try:
                if torch.cuda.is_available():
                    device = torch.device('cuda')
                    print('CUDA device selected and available.')
                else:
                    print('CUDA device selected but not available. Using CPU instead.')
            except:
                print('CUDA device selected but not working. Using CPU instead.')

        else:
            print('CPU device selected.')
    return device


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
        return AdamW(model.parameters(), lr=lr)
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
    if (dataset_type == 'train'):
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

    return partitions

def non_iid_partition(dataset, clients, min_label_ratio=0.2):
    # Retrieve the labels for the dataset
    labels = np.array(dataset.get_labels())

    # Identify indices of data points for each label
    indices_label_0 = np.where(labels == 0)[0]
    indices_label_1 = np.where(labels == 1)[0]

    # Shuffle the indices
    np.random.shuffle(indices_label_0)
    np.random.shuffle(indices_label_1)

    # Initialize the dictionary to hold the partitions
    partitions = {i: [] for i in range(clients)}

    # Calculate the minimum number of samples per label per client
    min_samples_label_0 = int(min_label_ratio * len(indices_label_0) / clients)
    min_samples_label_1 = int(min_label_ratio * len(indices_label_1) / clients)

    # Distribute the minimum required samples of each label to each client
    for i in range(clients):
        partitions[i].extend(indices_label_0[i * min_samples_label_0 : (i + 1) * min_samples_label_0])
        partitions[i].extend(indices_label_1[i * min_samples_label_1 : (i + 1) * min_samples_label_1])

    # Distribute remaining samples in a non-IID manner
    remaining_label_0 = indices_label_0[clients * min_samples_label_0 :]
    remaining_label_1 = indices_label_1[clients * min_samples_label_1 :]

    np.random.shuffle(remaining_label_0)
    np.random.shuffle(remaining_label_1)

    extra_per_client = len(remaining_label_0) // clients
    for i in range(clients):
        start_index = i * extra_per_client
        end_index = start_index + extra_per_client
        if i == clients - 1:
            end_index = len(remaining_label_0)
        partitions[i].extend(remaining_label_0[start_index:end_index])

    extra_per_client = len(remaining_label_1) // clients
    for i in range(clients):
        start_index = i * extra_per_client
        end_index = start_index + extra_per_client
        if i == clients - 1:
            end_index = len(remaining_label_1)
        partitions[i].extend(remaining_label_1[start_index:end_index])

    # Optionally shuffle the indices for each client
    for client in partitions:
        random.shuffle(partitions[client])

    return partitions