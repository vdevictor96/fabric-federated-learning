from .utils import update_lr
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import sys
from os.path import join as pjoin
# progress bar
from tqdm.auto import tqdm

def train_text_class(model, modelpath, modelname, train_loader, eval_loader, optimizer, lr, lr_scheduler, num_epochs, device, progress_bar_flag=True):
    model.train()
    total_steps_per_epoch = len(train_loader)
    total_steps = num_epochs * total_steps_per_epoch
    if progress_bar_flag:
        progress_bar = tqdm(range(total_steps))
    
    for epoch in range(num_epochs):
        accumulated_loss = 0
        steps = 0
        correct = 0
        total = 0
        for i, batch in enumerate(train_loader):
            # Move batch to GPU
            ids = batch['input_ids'].to(device=device, dtype=torch.long)
            mask = batch['attention_mask'].to(device=device, dtype=torch.long)
            targets = batch['label'].to(device=device, dtype=torch.long)

            optimizer.zero_grad()
            
            outputs = model(ids, mask, labels=targets)
            logits = outputs.logits
            loss = outputs.loss
            predicted = torch.argmax(logits, dim=-1)
            # backpropagation
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            # if progress_bar_flag:
            #     progress_bar.update(1)
            accumulated_loss += loss.item()
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            steps += 1
            # track train accuracy
            if (i+1) % 100 == 0:
                loss_step = accumulated_loss/steps
                accuracy_step = 100 * correct / total
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f} %'
                        .format(epoch+1, num_epochs, i+1, total_steps_per_epoch, loss_step, accuracy_step))
        loss_epoch = accumulated_loss/steps
        accuracy_epoch = 100 * correct / total
        print('Epoch [{}/{}], Loss is: {} %'.format(epoch+1, num_epochs, loss_epoch))
        print('Epoch [{}/{}], Train accuracy is: {:.2f} %'.format(epoch+1, num_epochs, accuracy_epoch))
        print("-------------------------------")

    # Save the model checkpoint
    torch.save(model.state_dict(), pjoin(modelpath, modelname + '.ckpt'))
    # torch.save(last_model, pjoin(modelpath, 'last_model_{}_{}.pt'.format(model_subpath, args.num_labeled)))

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
