import numpy as np
import torch
import torch.nn as nn

def federated_aggregate(weights):
    """Compute simple average of model weights from state dictionaries."""
    # Initialize a dictionary to store the aggregated weights
    agg_weights = {}

    # Iterate over each parameter in the state dictionary
    for key in weights[0]:
        # Sum the weights of this parameter across all state dictionaries
        agg_weights[key] = sum(state_dict[key] for state_dict in weights) / len(weights)

    return agg_weights

# def federated_aggregate(models):
#     """Compute simple average of model weights."""
#     # Initialize a dictionary to store the aggregated weights
#     agg_weights = {}

#     # Iterate over each parameter in the model's state_dict
#     for key in models[0].state_dict():
#         # Sum the weights of this parameter across all models
#         agg_weights[key] = sum(model.state_dict()[key] for model in models) / len(models)

#     # Create a new model to hold the aggregated weights
#     aggregated_model = type(models[0])()  # Assuming all models are of the same type
#     aggregated_model.load_state_dict(agg_weights)

#     return aggregated_model


# def federated_aggregate(models):
#     """Compute simple average of model weights."""
#     # Calculate the average weights of each layer
#     agg_weights = [
#         np.mean(layer, axis=0) for layer in zip(*models)
#     ]
#     return agg_weights