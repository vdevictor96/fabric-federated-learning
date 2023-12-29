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


def load_model_from_json(model, params):
    new_state_dict = model.state_dict()

    for key in new_state_dict.keys():
        # The key in your JSON might not exactly match the key in the state_dict
        # Adjust the following line if necessary to map the keys correctly
        param_key = key

        if param_key in params:
            # Convert array to tensor
            param_value = torch.tensor(params[param_key])
            # Check if the shape matches
            if param_value.shape == new_state_dict[key].shape:
                new_state_dict[key] = param_value
            else:
                raise ValueError(
                    f"Shape mismatch at {key}: {param_value.shape} vs {new_state_dict[key].shape}")
        else:
            raise KeyError(f"Missing key {key} in the provided parameters")

    model.load_state_dict(new_state_dict)


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