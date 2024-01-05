from .utils import update_lr
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import sys
from os.path import join as pjoin


def train(model, modelpath, modelname, dataloaders, criterion, optimizer, learning_rate, learning_rate_decay, input_size, num_epochs, device):

    # Train the model
    lr = learning_rate
    train_loader = dataloaders['train']
    val_loader = dataloaders['validation']
    test_loader = dataloaders['test']

    total_step = len(train_loader)
    for epoch in range(num_epochs):
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(train_loader):
            # if(i>2):
            #     break
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)

            # reshape images to input size
            images = images.reshape(-1, input_size).to(device)

            # set the model to train
            model.train()

            optimizer.zero_grad()

            # forward pass
            output = model(images)
            _, predicted = torch.max(output.data, 1)
            # calculate loss
            loss = criterion(output, labels)

            # backpropagation
            loss.backward()
            optimizer.step()

            # track train accuracy
            if (i+1) % 1 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Train accuracy is: {} %'.format(100 * correct / total))
        # Code to update the lr
        # lr *= learning_rate_decay
        # update_lr(optimizer, lr)
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                # reshape images to input size
                images = images.reshape(-1, input_size).to(device)
                # set the model for evaluation
                output = model(images)
                _, predicted = torch.max(output.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print('Validation accuracy is: {} %'.format(100 * correct / total))

    # Save the model checkpoint
    torch.save(model.state_dict(), pjoin(modelpath, modelname + '.ckpt'))
    # torch.save(last_model, pjoin(modelpath, 'last_model_{}_{}.pt'.format(model_subpath, args.num_labeled)))
