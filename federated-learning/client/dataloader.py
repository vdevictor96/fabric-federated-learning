import numpy as np
import torch
import math

import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import sys

def get_cifar10(root='./data'):
  norm_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                     ])
  cifar_dataset = datasets.CIFAR10(root,
                                           train=True,
                                           transform=norm_transform,
                                           download=True)

  test_dataset = datasets.CIFAR10(root,
                                          train=False,
                                          transform=norm_transform
                                          )

  
  return cifar_dataset, test_dataset


def prepare_train_val(dataset, num_training=45000, num_validation=5000):
  mask = list(range(num_training))
  train_dataset = torch.utils.data.Subset(dataset, mask)
  mask = list(range(num_training, num_training + num_validation))
  val_dataset = torch.utils.data.Subset(dataset, mask)
  return train_dataset, val_dataset

def get_cifar10_datasets(root, num_training=45000, num_validation=5000):
  cifar, test_dataset = get_cifar10(root)
  train_dataset, val_dataset = prepare_train_val(cifar, num_training, num_validation)
  return train_dataset, val_dataset, test_dataset


def get_cifar10_dataloaders(root, batch_size = 200, num_training=45000, num_validation=5000, device='cuda'):
  cifar, test_dataset = get_cifar10(root)
  train_dataset, val_dataset = prepare_train_val(cifar, num_training, num_validation)
  
  train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                             generator=torch.Generator(device=device),
                                           batch_size=batch_size,
                                           shuffle=True)

  val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             generator=torch.Generator(device=device),
                                            batch_size=batch_size,
                                            shuffle=False)

  test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                             generator=torch.Generator(device=device),
                                          batch_size=batch_size,
                                          shuffle=False)
  return train_loader, val_loader, test_loader