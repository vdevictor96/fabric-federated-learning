
import os
import base64
import json
import zlib
import torch
import torch.nn as nn
import sys
import io
import msgpack
import numpy as np


# --- Gateway Utils ---
def get_file_path(relative_path):
    # Get the directory of the script
    script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
    # Join the relative path with the script directory
    return os.path.abspath(os.path.join(script_dir, relative_path))


def serialize_model_json(state_dict, first_n=0, last_n=0):
    # Convert the state dict to a format that can be JSON serialized, but only for the first 10 and last 2 layers
    keys = list(state_dict.keys())
    if (first_n == 0 and last_n == 0):
        selected_keys = keys
    else:
        selected_keys = keys[:first_n] + keys[-last_n:]
    serializable_state_dict = {
        k: state_dict[k].tolist() for k in selected_keys}

    # Convert to JSON
    json_data = json.dumps(serializable_state_dict)
    return json_data


def serialize_model_numpy(state_dict, first_n=10, last_n=2):
    # Convert the state dict to a format that can be JSON serialized, but only for the first 10 and last 2 layers
    keys = list(state_dict.keys())
    if (first_n == 0 and last_n == 0):
        selected_keys = keys
    else:
        selected_keys = keys[:first_n] + keys[-last_n:]

    serializable_state_dict = {
        k: state_dict[k].tolist() for k in selected_keys}

    # Convert the state dictionary to a format that can be serialized by MessagePack
    # serializable_state_dict = {k: state_dict[k].cpu().numpy().tolist() for k in selected_keys}

    # 1. Serialize using Numpy
    buffer = io.BytesIO()
    np.savez_compressed(buffer, serializable_state_dict)
    buffer.seek(0)
    # 2. Compress using zlib
    compressed_data = zlib.compress(buffer.getvalue())
    # 3. Encode as base64
    encoded_str = base64.b64encode(compressed_data).decode('utf-8')

    return encoded_str


def deserialize_model_numpy(encoded_str):
    # 1. Decode from base64
    decoded_data = base64.b64decode(encoded_str)
    # 2. Decompress from zlib
    decompressed_data = zlib.decompress(decoded_data)
    # 3. Deserialize using Numpy
    buffer = io.BytesIO(decompressed_data)
    unpacked_data = np.load(buffer, allow_pickle=True)

    # 4. Convert to torch tensors
    raise NotImplementedError
    return {key: torch.tensor(value) for key, value in unpacked_data.items()}


def serialize_model_msgpack(state_dict, first_n=10, last_n=2):
    # Convert the state dict to a format that can be JSON serialized, but only for the first 10 and last 2 layers
    keys = list(state_dict.keys())
    if (first_n == 0 and last_n == 0):
        selected_keys = keys
    else:
        selected_keys = keys[:first_n] + keys[-last_n:]

    serializable_state_dict = {
        k: state_dict[k].tolist() for k in selected_keys}

    # Convert the state dictionary to a format that can be serialized by MessagePack
    # serializable_state_dict = {k: state_dict[k].cpu().numpy().tolist() for k in selected_keys}
    # 1. Serialize using MessagePack
    packed_data = msgpack.packb(serializable_state_dict)
    # 2. Compress using zlib
    compressed_data = zlib.compress(packed_data)
    # 3. Encode as base64
    encoded_str = base64.b64encode(compressed_data).decode('utf-8')

    return encoded_str


def deserialize_model_msgpack(encoded_str):
    # 1. Decode from base64
    decoded_data = base64.b64decode(encoded_str)
    # 2. Decompress from zlib
    decompressed_data = zlib.decompress(decoded_data)
    # 3. Deserialize using MessagePack
    unpacked_data = msgpack.unpackb(decompressed_data, raw=False)

    # 4. Convert to torch tensors
    return {key: torch.tensor(value) for key, value in unpacked_data.items()}


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
        
        


def print_layer_size(state_dict, first_n=10, last_n=2, size_type='bytes', format='json'):
    keys = list(state_dict.keys())
    if (first_n == 0 and last_n == 0):
        selected_keys = keys
    else:
        selected_keys = keys[:first_n] + keys[-last_n:]

    serializable_state_dict = {
        k: state_dict[k].tolist() for k in selected_keys}
    if format == 'json':
        serialized_data = json.dumps(serializable_state_dict)
        total_size = sys.getsizeof(serialized_data)
    elif format == 'base64':
        # TODO change to do it with msgpack
        serialized_data = base64.b64encode(json.dumps(
            serializable_state_dict).encode())
        total_size = sys.getsizeof(serialized_data)
    else:
        total_size = sys.getsizeof(serializable_state_dict)
    if (size_type == 'mb'):
        total_size = total_size / 1024 / 1024
    elif (size_type == 'kb'):
        total_size = total_size / 1024
    print(f"Total size: {total_size} {size_type}")
    # Print the size of each value in the dictionary
    for key, value in serializable_state_dict.items():
        if format == 'json':
            serialized_value = json.dumps(value)
        elif format == 'base64':
            serialized_value = base64.b64encode(json.dumps(value).encode())
        else:
            serialized_value = value
        size = sys.getsizeof(serialized_value)
        if (size_type == 'mb'):
            size = size / 1024 / 1024
        elif (size_type == 'kb'):
            size = size / 1024
        print(f"Size of value for key '{key}': {size} {size_type}")