import os
import base64
import json
import torch
import torch.nn as nn

def get_file_path(relative_path):
    # Get the directory of the script
    script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in

    # Join the relative path with the script directory
    return os.path.abspath(os.path.join(script_dir, relative_path))

def serialize_model(state_dict):
    # Convert the state dict to a format that can be JSON serialized
    serializable_state_dict = {k: v.tolist() for k, v in state_dict.items()}
    # Convert to JSON
    json_data = json.dumps(serializable_state_dict)
    return json_data

def encode_file_to_base64(filepath):
    with open(filepath, 'rb') as file:
        encoded_string = base64.b64encode(file.read()).decode('utf-8')
    return encoded_string

def decode_base64_to_file(encoded_string, output_filepath):
    with open(output_filepath, 'wb') as file:
        file.write(base64.b64decode(encoded_string))


def weights_init(m):
    if type(m) == nn.Linear:
        m.weight.data.normal_(0.0, 1e-3)
        m.bias.data.fill_(0.)

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    