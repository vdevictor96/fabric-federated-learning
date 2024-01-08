from .utils import update_lr
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import sys
from os.path import join as pjoin


def test(model, input_size, test_loader, device='cuda'):
    # Test the model
    # In test phase, we don't need to compute gradients (for memory efficiency)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            ####################################################

            # reshape images to input size
            images = images.reshape(-1, input_size).to(device)
            # set the model for evaluation
            output = model(images)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if total == 1000:
                break

    print('Accuracy of the network on the {} test images: {} %'.format(
        total, 100 * correct / total))