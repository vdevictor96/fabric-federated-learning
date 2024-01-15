# !/usr/bin/env python
import json
import argparse
import sys
import torch
import os
import warnings
from .data.reddit_dep import get_reddit_dep_test_dataloader
from .data.acl_dep_sad import get_acl_dep_sad_test_dataloader
from .data.dreaddit import get_dreaddit_test_dataloader
from .data.mixed_depression import get_mixed_depression_test_dataloader
from .test import test_text_class
from .utils import set_seed, set_device, create_tokenizer, get_dir_path, get_dataset_path, load_model


DEFAULT_CONFIG_FILE = "config/default_test_config.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parse the path to the JSON config file.")
    parser.add_argument('--config_file', type=str, default=None,
                        help='Optional: Path to the JSON configuration file.')
    parser.add_argument('--show_config', action='store_true',
                        help='Show configurable parameters and default values.')
    args = parser.parse_args()

    if args.show_config:
        print_config_help()
        sys.exit(0)

    return args


def print_config_help():
    default_config_info = load_default_config(True)
    print("Configurable parameters and their current settings:")
    for key, param_info in default_config_info.items():
        print(f"\nParameter: {key}")
        print(f"  Description: {param_info['description']}")
        print(f"  Default Value: {param_info['default_value']}")
        print(f"  Valid Values: {param_info['valid_values']}")
    # Extract only the 'default_value' field from each entry
    default_config = {key: value_info["default_value"]
                      for key, value_info in default_config_info.items()}
    pretty_default_config = json.dumps(
        default_config, indent=4, sort_keys=True)
    print("\n\nDefault configuration JSON:")
    print(pretty_default_config)


def load_default_config(info=False):
    # Construct the absolute path to the default config file
    dir_path = get_dir_path()
    default_config_path = os.path.join(dir_path, DEFAULT_CONFIG_FILE)

    # Default values
    with open(default_config_path, 'r') as default_file:
        default_config_info = json.load(default_file)
    if info:
        return default_config_info
    # Extract only the 'value' field from each entry
    default_config = {key: value_info["default_value"]
                      for key, value_info in default_config_info.items()}
    return default_config


def load_config(config_file):
    config = {}  # Initialize empty config
    # Load default config
    config = load_default_config(info=False)

    # Override defaults with values from JSON file
    if config_file is not None:
        try:
            with open(config_file, 'r') as file:
                user_config = json.load(file)
                config.update(user_config)
        except FileNotFoundError:
            print(
                f"Warning: Config file {config_file} not found. Using default settings.")
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
    if args.config_file == DEFAULT_CONFIG_FILE:
        config = load_default_config(False)
    else:
        config = load_config(args.config_file)
    print_config(config)
    print('-------- Configuration loaded --------')

    print('\n-------- Setting device --------')
    device = set_device(config['device'])
    print('-------- Device set --------')

    print('\n-------- Setting seed --------')
    set_seed(config['seed'])
    print('-------- Seed set --------')

    print('\n-------- Loading model from model_path --------')
    model = load_model(config['model'], config['model_path'], device)
    print('-------- Model loaded --------')

    print('\n-------- Creating Tokenizer --------')
    tokenizer = create_tokenizer(config['model'])
    print('-------- Tokenizer created --------')

    print('\n-------- Creating Test Dataloader --------')
    test_loader = create_test_dataloader(
        config['dataset'], tokenizer, config['test_batch_size'], config['max_length'], config['seed'])
    print(
        f'Test Loader: {len(test_loader.dataset)} total sentences. {len(test_loader)} batches of size {config["test_batch_size"]}.')
    print('-------- Train and Eval Dataloaders created --------')

    print('\n-------- Testing --------')
    test_text_class(model, test_loader, device)
    print('-------- Testing finished --------')
    return


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


def create_test_dataloader(dataset_type, tokenizer, test_batch_size, max_length, seed):
    if dataset_type == 'reddit_dep':
        dataset_path = get_dataset_path(dataset_type, 'test')
        return get_reddit_dep_test_dataloader(dataset_path, tokenizer, test_batch_size, max_length, seed)
    elif dataset_type == 'acl_dep_sad':
        dataset_path = get_dataset_path(dataset_type, 'test')
        return get_acl_dep_sad_test_dataloader(dataset_path, tokenizer, test_batch_size, max_length, seed)
    elif dataset_type == 'dreaddit':
        dataset_path = get_dataset_path(dataset_type, 'test')
        return get_dreaddit_test_dataloader(dataset_path, tokenizer, test_batch_size, max_length, seed)
    elif dataset_type == 'mixed_depression':
        dataset_path = get_dataset_path(dataset_type, 'test')
        return get_mixed_depression_test_dataloader(dataset_path, tokenizer, test_batch_size, max_length, seed)
    else:
        raise ValueError(f"Unknown dataset {dataset_type}.")


if __name__ == "__main__":
    main()
