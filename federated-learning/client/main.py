# !/usr/bin/env python
import json
import argparse
import sys
import torch
import os
import warnings
from .model.bert_tiny import get_bert_tiny_tokenizer, get_bert_tiny_model
from .data.reddit_dep import get_reddit_dep_dataloaders

DEFAULT_CONFIG_FILE = "config/default_config.json"

def parse_args():
    parser = argparse.ArgumentParser(description="Parse the path to the JSON config file.")
    parser.add_argument('--config_file', type=str, default=None, help='Optional: Path to the JSON configuration file.')
    parser.add_argument('--show_config', action='store_true', 
                        help='Show configurable parameters and default values.')
    args = parser.parse_args()
    
    if args.show_config:
        print_config_help()
        sys.exit(0)

    return args

def print_config_help():
    default_config_info = load_default_config()
    print("Configurable parameters and their current settings:")
    for key, param_info in default_config_info.items():
        print(f"\nParameter: {key}")
        print(f"  Description: {param_info['description']}")
        print(f"  Default Value: {param_info['value']}")
        print(f"  Valid Values: {param_info['valid_values']}")
    # Extract only the 'value' field from each entry
    default_config = {key: value_info["value"] for key, value_info in default_config_info.items()}
    pretty_default_config = json.dumps(default_config, indent=4, sort_keys=True)
    print("\n\nDefault configuration JSON:")
    print(pretty_default_config)
    
    
def load_default_config():
    # Construct the absolute path to the default config file
    dir_path = _get_dir_path()
    default_config_path = os.path.join(dir_path, DEFAULT_CONFIG_FILE)
    
    # Default values
    with open(default_config_path, 'r') as default_file:
        default_config = json.load(default_file)
    return default_config

def load_config(config_file):
    config = {}  # Initialize empty config
    # Load default config
    default_config = load_default_config()
    # Extract only the 'value' field from each entry
    config = {key: value_info["value"] for key, value_info in default_config.items()}
        
    # Override defaults with values from JSON file
    if config_file is not None:
        try:
            with open(config_file, 'r') as file:
                user_config = json.load(file)
                config.update(user_config)
        except FileNotFoundError:
            print(f"Warning: Config file {config_file} not found. Using default settings.")
    else:
        print("No configuration file provided. Using default settings.")
        
    return config

def print_config(config):
    pretty_config = json.dumps(config, indent=4, sort_keys=True)
    print("Current configuration:")
    print(pretty_config)

def main():
    args = parse_args()
    
    print('\n-------- Loading configuration --------')
    config = load_config(args.config_file)
    print_config(config)
    print('-------- Configuration loaded --------')
    
    print('\n-------- Setting device --------')
    device = _set_device(config['device'])
    print('-------- Device set --------')
    
    print('\n-------- Creating Model --------')
    model = _create_model(config['model'], device)
    print('-------- Model created --------')
    
    print('\n-------- Creating Tokenizer --------')
    tokenizer = _create_tokenizer(config['model'])
    print('-------- Tokenizer created --------')
    
    print('\n-------- Creating Train and Eval Dataloaders --------')
    train_loader, eval_loader = _create_dataloaders(config['dataset'], tokenizer, config['train_size'], config['eval_size'], config['train_batch_size'], config['eval_batch_size'], config['max_length'], config['seed'])
    print(f'Train Loader: {len(train_loader.dataset)} total sentences. {len(train_loader)} batches of size {config["train_batch_size"]}.')
    print(f'Eval Loader: {len(eval_loader.dataset)} total sentences. {len(eval_loader)} batches of size {config["eval_batch_size"]}.')
    print('-------- Train and Eval Dataloaders created --------')
    
    
    
    return

def _set_device(device_name):
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

def _create_model(model_type, device):
    if model_type == 'bert_tiny':
        return get_bert_tiny_model(device)
    else:
        raise ValueError(f"Unknown model type {model_type}.")
        
def _create_tokenizer(model_type):
    if model_type == 'bert_tiny':
        return get_bert_tiny_tokenizer()
    else:
        raise ValueError(f"Unknown tokenizer for model type {model_type}.")
    
def _create_dataloaders(dataset_type, tokenizer, train_size, eval_size, train_batch_size, eval_batch_size, max_length, seed):
    if dataset_type == 'reddit_dep':
        dataset_path = _get_dataset_path(dataset_type)
        return get_reddit_dep_dataloaders(dataset_path, tokenizer, train_size, eval_size, train_batch_size, eval_batch_size, max_length, seed)
    else:
        raise ValueError(f"Unknown dataset {dataset_type}.")
                 
def _get_dir_path():
    # Get the directory of the current script
    return os.path.dirname(os.path.realpath(__file__))

def _get_dataset_path(dataset_name):
    # Get the directory of the current script
    dir_path = _get_dir_path()
    dataset_path = os.path.join(dir_path, "data", "datasets", dataset_name, dataset_name + "_train.csv")
    
    # Check if the file exists
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"The dataset file does not exist: {dataset_path}")
    
    return dataset_path

if __name__ == "__main__":
    main()
