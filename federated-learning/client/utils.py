import os
import torch
import torch.nn as nn

def get_file_path(relative_path):
    # Get the directory of the script
    script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in

    # Join the relative path with the script directory
    return os.path.abspath(os.path.join(script_dir, relative_path))


def weights_init(m):
    if type(m) == nn.Linear:
        m.weight.data.normal_(0.0, 1e-3)
        m.bias.data.fill_(0.)

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    