# !/usr/bin/env python
import json
import argparse
import sys
import torch
import os
from os.path import join as pjoin
import warnings
from .data.twitter_dep import get_twitter_dep_dataloaders
from .data.acl_dep_sad import get_acl_dep_sad_dataloaders
from .data.dreaddit import get_dreaddit_dataloaders
from .data.mixed_depression import get_mixed_depression_dataloaders
from .data.deptweet import get_deptweet_dataloaders
from .train import train_text_class, train_text_class_fl
from .utils import set_seed, set_device, create_model, create_tokenizer, get_dir_path, get_dataset_path, create_optimizer, create_scheduler, freeze_layers, get_trainable_state_dict_elements, create_test_dataloader, load_model, duplicate_output_to_file
from .test import test_text_class


DEFAULT_CONFIG_FILE = "config/default_config.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parse the path to the JSON config file.")
    parser.add_argument('--config_file', type=str, default=None,
                        help='Optional: Path to the JSON configuration file.')
    parser.add_argument('--show_config', action='store_true',
                        help='Shows configurable parameters and default values.')
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
    # Parse arguments
    args = parse_args()
    print('\n-------- Loading configuration --------')
    config = load_config(args.config_file)
    # Set path to save model
    model_save_path = None
    if config['save_model']:
        # Save the best model at the end
        if not os.path.isdir(config['models_path']):
            os.makedirs(config['models_path'])
        # Set path to save model
        model_save_path = pjoin(
            config['models_path'], config['model_name'])
        # Duplicate the output to a file
        log_file = model_save_path + '.log'
        duplicate_output_to_file(log_file)
    print_config(config)
    print('-------- Configuration loaded --------')

    print('\n-------- Setting device --------')
    device = set_device(config['device'])
    print('-------- Device set --------')

    print('\n-------- Setting seed --------')
    set_seed(config['seed'], device)
    print('-------- Seed set --------')

    print('\n-------- Creating Model --------')
    model = create_model(config['model'], device)
    print('-------- Model created --------')

    print('\n-------- Setting Trainable Layers --------')
    total_params = sum(p.numel() for p in model.parameters())
    # Train only the last 'layers' layers
    total_params = sum(p.numel() for p in model.parameters())
    if config['layers'] <= 0:
        print('The layers parameter must be positive. Training only the last layer classifier.')
        config['layers'] = 1
    elif config['layers'] > 4:
        print('The layers parameter must be less than or equal to 4. Training the last 4 layers.')
        config['layers'] = 4
    else:
        print(
            f'Training the last {config["layers"]} layers.')

    trainable_params, layers = freeze_layers(
        model, config['layers'])

    # Comented because embeddings are not trainable

    # Freeze non-compatible layers with Opacus for differential privacy
    # print('-------- Freezing non-compatible layers with Opacus --------')
    # non_layers = [
    #     ('bert.embeddings.position_embeddings', model.bert.embeddings.position_embeddings)]
    # filtered_non_layers = non_layers
    # if layers is not None:
    #     # Filter out layers from layers that are in non_layers
    #     layers = [
    #         (name, layer) for name, layer in layers
    #         if not any(layer is ntl[1] for ntl in non_layers)
    #     ]
    #     # Update filtered_non_layers
    #     filtered_non_layers = [
    #         (name, layer) for name, layer in non_layers if any(layer is tl[1] for tl in layers)
    #     ]

    # # Freeze non-trainable layers
    # for name, layer in filtered_non_layers:
    #     print(f'Freezing layer {name}: {layer}')
    #     for p in layer.parameters():
    #         p.requires_grad = False
    #         trainable_params -= p.numel()

    # print('-------- Non-compatible layers with Opacus frozen --------')

    print(f"\nTotal parameters count: {total_params}")
    print(f"Trainable parameters count: {trainable_params}")

    print('-------- Trainable Layers set --------')

    print('\n-------- Creating Tokenizer --------')
    tokenizer = create_tokenizer(config['model'])
    print('-------- Tokenizer created --------')

    print('\n-------- Creating Train and Eval Dataloaders --------')
    train_loader, eval_loader = create_dataloaders(config['dataset'], tokenizer, config['train_size'], config['eval_size'],
                                                   config['train_batch_size'], config['eval_batch_size'], config['max_length'], config['seed'])
    print(
        f'Train Loader: {len(train_loader.dataset)} total sentences. {len(train_loader)} batches of size {config["train_batch_size"]}.')
    print(
        f'Eval Loader: {len(eval_loader.dataset)} total sentences. {len(eval_loader)} batches of size {config["eval_batch_size"]}.')
    print('-------- Train and Eval Dataloaders created --------')

    ml_mode = config['ml_mode']

    if ml_mode == 'ml':
        print('\n-------- Creating Optimizer --------')
        optimizer = create_optimizer(
            config['optimizer'], model, config['learning_rate'])
        print('-------- Optimizer created --------')
    else:
        # optimizer will be created for each local client
        pass

    if ml_mode == 'ml':
        print('\n-------- Creating Scheduler --------')
        num_training_steps = config['num_epochs'] * len(train_loader)
        scheduler = create_scheduler(
            config['scheduler'], optimizer, num_training_steps, config['scheduler_warmup_steps'])
        print('-------- Scheduler created --------')
    else:
        # scheduler will be created for each local client
        pass

    print('\n-------- Training --------')
    last_model = None
    # Setting ML mode
    ml_mode_string = 'Centralised Machine Learning' if ml_mode == 'ml' else 'Federated Learning' if ml_mode == 'fl' else 'Blockchain-Based Federated Learning'
    print("Training with {} technology.".format(ml_mode_string))

    # Setting differential privacy
    if config['dp_epsilon'] == 0:
        print('Training without differential privacy.')
    elif config['dp_epsilon'] < 0:
        print(
            'Epsilon value must be positive. Training without differential privacy.')
    else:
        # Suppress specific warnings when using Opacus
        warnings.filterwarnings(
            'ignore', message="Secure RNG turned off.*")
        warnings.filterwarnings(
            'ignore', message="Optimal order is the largest alpha.*")
        warnings.filterwarnings(
            'ignore', message="invalid value encountered in log")
        warnings.filterwarnings(
            'ignore', message="Using a non-full backward hook.*")
        print('Training with differential privacy.')
        if config['dp_epsilon'] > 10:
            print('Training without differential privacy.')
            print('Epsilon value is too large. Reducing its value to 10.')
            config['dp_epsilon'] = 10

    if ml_mode == 'ml':
        if config['concurrency_flag']:
            print(
                "Concurrency flag is set to True, but ml mode is selected. Concurrency flag will be ignored.")
        last_model = train_text_class(model, model_save_path, train_loader, eval_loader, optimizer,
                                      config['learning_rate'], scheduler, config['num_epochs'], device, config['eval_flag'], config['save_model'], config['progress_bar_flag'], config['dp_epsilon'], config['dp_delta'])

    elif ml_mode == 'fl' or ml_mode == 'bcfl':
        # Setting federated algorithm
        if config['fed_alg'] == 'fedavg':
            print("Federated averaging algorithm selected.")
        elif config['fed_alg'] == 'fedprox':
            print("Federated proximal algorithm selected.")
            if config['mu'] <= 0:
                print("Mu value must be positive. Using value of 0.001.")
                config['mu'] = 0.001
            elif config['mu'] > 10:
                print("Mu value must be less than or equal to 10. Using value of 10.")
                config['mu'] = 1
            else:
                print("Using mu value of {}.".format(config['mu']))
        else:
            print("Unknown federated algorithm selected. Using federated averaging.")
            config['fed_alg'] = 'fedavg'

        last_model = train_text_class_fl(model, config['ml_mode'], config['fed_alg'], config['mu'], config['model_name'], model_save_path, layers, train_loader, eval_loader, config['optimizer'],
                                         config['learning_rate'], config['scheduler'], config[
            'scheduler_warmup_steps'], config['num_epochs'], config['concurrency_flag'], device, config['eval_flag'], config['save_model'],
            config['progress_bar_flag'], config['num_rounds'], config['num_clients'],
            config['dp_epsilon'], config['dp_delta'], config['data_distribution'])

    else:
        raise ValueError(f"Unknown learning mode {ml_mode}.")

    print('-------- Training finished --------')

    if config['test_flag']:

        if config['save_model'] is False:
            print('\nsave_model flag is False. Testing skipped.')
            return

        print('\nTest flag enabled. Testing the model')
        config['test_batch_size'] = 8

        if last_model is not None:
            print('\n-------- Loading last model --------')
            print('-------- Last model loaded --------')
        else:
            print('\n-------- Loading best model from model_path --------')
            test_model_path = model_save_path + '_best.ckpt'
            test_model = load_model(config['model'], test_model_path, device)
            print('-------- Best model loaded --------')

        print('\n-------- Creating Test Dataloader --------')
        test_loader = create_test_dataloader(
            config['dataset'], tokenizer, config['test_batch_size'], config['max_length'], config['seed'])
        print(
            f'Test Loader: {len(test_loader.dataset)} total sentences. {len(test_loader)} batches of size {config["test_batch_size"]}.')
        print('-------- Test Dataloader created --------')

        print('\n-------- Testing --------')
        test_text_class(test_model, test_loader, device,
                        config['progress_bar_flag'])
        print('-------- Testing finished --------')

    # Restore original stdout and close the file
    sys.stdout.close()
    sys.stdout = sys.__stdout__

    return


def create_dataloaders(dataset_type, tokenizer, train_size, eval_size, train_batch_size, eval_batch_size, max_length, seed):
    if dataset_type == 'twitter_dep':
        dataset_path = get_dataset_path(dataset_type)
        return get_twitter_dep_dataloaders(dataset_path, tokenizer, train_size, eval_size, train_batch_size, eval_batch_size, max_length, seed)
    elif dataset_type == 'acl_dep_sad':
        dataset_path = get_dataset_path(dataset_type)
        return get_acl_dep_sad_dataloaders(dataset_path, tokenizer, train_size, eval_size, train_batch_size, eval_batch_size, max_length, seed)
    elif dataset_type == 'dreaddit':
        dataset_path = get_dataset_path(dataset_type)
        return get_dreaddit_dataloaders(dataset_path, tokenizer, train_size, eval_size, train_batch_size, eval_batch_size, max_length, seed)
    elif dataset_type == 'mixed_depression':
        dataset_path = get_dataset_path(dataset_type)
        return get_mixed_depression_dataloaders(dataset_path, tokenizer, train_size, eval_size, train_batch_size, eval_batch_size, max_length, seed)
    elif dataset_type == 'deptweet':
        dataset_path = get_dataset_path(dataset_type)
        return get_deptweet_dataloaders(dataset_path, tokenizer, train_size, eval_size, train_batch_size, eval_batch_size, max_length, seed)
    else:
        raise ValueError(f"Unknown dataset {dataset_type}.")


if __name__ == "__main__":
    main()
