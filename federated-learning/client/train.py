from .utils import iid_partition, non_iid_partition, create_optimizer, create_scheduler
from .aggregators import federated_aggregate
import torch
import torch.nn as nn
import torchvision
import sys
import os
import copy
from torch.utils.data import DataLoader, SubsetRandomSampler
from datetime import datetime
from os.path import join as pjoin
# progress bar
from tqdm.auto import tqdm

def train_text_class(model, modelpath, modelname, train_loader, eval_loader, optimizer, lr, lr_scheduler, num_epochs, device='cuda', eval_flag=True, progress_bar_flag=True):
    total_steps = num_epochs * len(train_loader)
    if eval_flag:
        total_steps += num_epochs * len(eval_loader)
    if progress_bar_flag:
        progress_bar = tqdm(range(total_steps))
    # Save the best model at the end
    if eval_flag and not os.path.isdir(modelpath):
        os.makedirs(modelpath)
    # Initialize variables to track the best model
    best_val_accuracy = 0.0
    best_model_state = None
    best_epoch = 0
    current_date = datetime.now().strftime("%d-%m-%Y %H:%M")
    for epoch in range(num_epochs):
        model.train()
        accumulated_loss, steps, correct, total = 0, 0, 0, 0
        for i, batch in enumerate(train_loader):
            # ids = batch['input_ids']
            # mask = batch['attention_mask']
            # targets = batch['label']
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
            if progress_bar_flag:
                progress_bar.update(1)
            accumulated_loss += loss.item()
            total += targets.size(0)
            correct += (predicted == targets).cpu().sum().item()
            steps += 1
            # track train accuracy
            if (i+1) % 100 == 0:
                loss_step = accumulated_loss/steps
                accuracy_step = 100 * correct / total
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f} %'
                        .format(epoch+1, num_epochs, i+1, total_steps_per_epoch, loss_step, accuracy_step))
        loss_epoch = accumulated_loss/steps
        accuracy_epoch = 100 * correct / total
        print("-------------------------------")
        print('Epoch [{}/{}] finished, Loss: {:.4f}, Accuracy: {:.2f} %'.format(epoch+1, num_epochs, loss_epoch, accuracy_epoch))
        print("-------------------------------")
        # ---------------------- Validation ----------------------
        if eval_flag:
            model.eval()
            val_loss, val_correct, val_total = 0, 0, 0
            with torch.no_grad():
                for i, batch in enumerate(eval_loader):
                    ids = batch['input_ids'].to(device=device, dtype=torch.long)
                    mask = batch['attention_mask'].to(device=device, dtype=torch.long)
                    targets = batch['label'].to(device=device, dtype=torch.long)

                    outputs = model(ids, mask, labels=targets)
                    loss = outputs.loss
                    predicted = torch.argmax(outputs.logits, dim=-1)

                    val_loss += loss.item()
                    val_total += targets.size(0)
                    val_correct += (predicted == targets).cpu().sum().item()
                    if progress_bar_flag:
                        progress_bar.update(1)
            val_accuracy = 100 * val_correct / val_total
            print('Validation Loss: {:.4f}, Validation Accuracy: {:.2f} %'.format(val_loss / len(eval_loader), val_accuracy))
            print("-------------------------------")
            # Check if this is the best model based on validation accuracy
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_model_state = model.state_dict().copy()
                best_model = {
                    'epoch': epoch+1,
                    'lr': lr,
                    'optimizer': optimizer.__class__.__name__,
                    'tr_acc': accuracy_epoch,
                    'val_acc': best_val_accuracy,
                    'date': current_date,
                    'model_state_dict': model.state_dict().copy(),
                    'lr_scheduler_dict': lr_scheduler.state_dict().copy(),
                    'optimizer_dict': optimizer.state_dict().copy(),
                    
                }
                best_epoch = epoch + 1
                print(f"Updated best model in epoch {best_epoch} saved with Validation Accuracy: {best_val_accuracy:.2f} %")
                print("-------------------------------")
                torch.save(best_model, pjoin(modelpath, modelname + '_best.ckpt'))
                
    # ---------------------- Saving Models ----------------------

    if best_model_state is not None:
        print(f"Best model in epoch {best_epoch} saved with Validation Accuracy: {best_val_accuracy:.2f} %")
        # return best_model_state
    else: 
        # Save the last model checkpoint
        last_model = {
            'epoch': epoch+1,
            'lr': lr,
            'optimizer': optimizer.__class__.__name__,
            'tr_acc': accuracy_epoch,
            'val_acc': best_val_accuracy,
            'date': current_date,
            'model_state_dict': model.state_dict().copy(),
            'lr_scheduler_dict': lr_scheduler.state_dict().copy(),
            'optimizer_dict': optimizer.state_dict().copy(),
        }
        torch.save(last_model, pjoin(modelpath, modelname + '_last.ckpt'))
        print(f"Last model in Epoch {epoch+1} saved with Training Accuracy: {accuracy_epoch:.2f} %")
        # return model.state_dict()


def train_text_class_fl(model, modelpath, modelname, train_loader, eval_loader, optimizer_type, lr, scheduler_type, scheduler_warmup_steps, num_epochs, device='cuda', eval_flag=True, progress_bar_flag=True, num_rounds=10, num_clients=5, dp_epsilon=0.0, data_distribution='iid'):
    # Set the progress bar
    total_steps = num_rounds * num_epochs * len(train_loader)
    if eval_flag:
        total_steps += num_rounds * len(eval_loader)
    if progress_bar_flag:
        progress_bar = tqdm(range(total_steps))
    else:
        progress_bar = None
        
    # partition the training dataset
    if data_distribution == 'iid':
        partitioned_indexes = iid_partition(train_loader.dataset, num_clients)
    else: # non-iid
        partitioned_indexes = non_iid_partition(train_loader.dataset, num_clients)
    
    # Save the best model at the end
    if eval_flag and not os.path.isdir(modelpath):
        os.makedirs(modelpath)
    # Initialize variables to track the best model
    best_val_accuracy = 0.0
    best_model_state = None
    best_round = 0
    current_date = datetime.now().strftime("%d-%m-%Y %H:%M")
    
    global_model = model
    # outer training loop
    for round in range(num_rounds):
        weights, local_loss, local_acc = [], [], []
        print("-------------------------------")
        print(f"Round {round+1} of {num_rounds}")
        # inner training loop
        for client in range(num_clients):
            print("-------------------------------")
            print(f"Client {client+1} of {num_clients}")
            # train the local model on the partitioned dataset
            c_weights, c_local_loss, c_local_acc = train_text_class_fl_inner(global_model, train_loader, partitioned_indexes[client], optimizer_type, lr, scheduler_type, scheduler_warmup_steps, num_epochs, device, progress_bar_flag, progress_bar)
            # append the weights, local loss and local accuracy
            weights.append(copy.deepcopy(c_weights))
            local_loss.append(copy.deepcopy(c_local_loss))
            local_acc.append(copy.deepcopy(c_local_acc))
        # loss and accuracy metrics from average of local loss and accuracy
        loss_avg = sum(local_loss) / len(local_loss)
        acc_avg = sum(local_acc) / len(local_acc)
        print("-------------------------------")
        print('Round [{}/{}] finished, Average Local Loss: {:.4f}, Average Local Accuracy: {:.2f} %'.format(round+1, num_rounds, loss_avg, acc_avg))
        print("-------------------------------")
        # aggregate the models and update the global model
        global_weights = {}
        global_weights = federated_aggregate(weights)
        global_model.load_state_dict(global_weights)
        # ---------------------- Validation ----------------------
        if eval_flag:
            # validation loss and accuracy of the global model
            global_model.eval()
            val_loss, val_correct, val_total = 0, 0, 0
            with torch.no_grad():
                for i, batch in enumerate(eval_loader):
                    ids = batch['input_ids'].to(device=device, dtype=torch.long)
                    mask = batch['attention_mask'].to(device=device, dtype=torch.long)
                    targets = batch['label'].to(device=device, dtype=torch.long)

                    outputs = global_model(ids, mask, labels=targets)
                    loss = outputs.loss
                    predicted = torch.argmax(outputs.logits, dim=-1)

                    val_loss += loss.item()
                    val_total += targets.size(0)
                    val_correct += (predicted == targets).cpu().sum().item()
                    if progress_bar_flag:
                        progress_bar.update(1) 
            val_accuracy = 100 * val_correct / val_total
            print("-------------------------------")
            print('Round [{}/{}] finished, Global Model Validation Loss: {:.4f}, Validation Accuracy: {:.2f} %'.format(round+1, num_rounds, val_loss / len(eval_loader), val_accuracy))
            print("-------------------------------")
            # Check if this is the best model based on validation accuracy
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_model_state = global_model.state_dict().copy()
                best_model = {
                    'round': round+1,
                    'lr': lr,
                    'optimizer': optimizer_type,
                    'tr_acc': acc_avg,
                    'val_acc': best_val_accuracy,
                    'date': current_date,
                    'model_state_dict': global_model.state_dict().copy(),
                    # could be averaged as the model_state_dict
                    # 'lr_scheduler_dict': lr_scheduler.state_dict().copy(),
                    # 'optimizer_dict': optimizer.state_dict().copy(),
                    
                }
                best_round = round + 1
                print(f"Updated best model in round {best_round} saved with Validation Accuracy: {best_val_accuracy:.2f} %")
                print("-------------------------------")
                torch.save(best_model, pjoin(modelpath, modelname + '_best.ckpt'))
    
    # ---------------------- Saving Models ----------------------
    if best_model_state is not None:
        print(f"Best model in round {best_round} saved with Validation Accuracy: {best_val_accuracy:.2f} %")
        # return best_model_state
    else: 
        # Save the last model checkpoint
        last_model = {
            'round': round+1,
            'lr': lr,
            'optimizer': optimizer_type,
            'tr_acc': acc_avg,
            'val_acc': best_val_accuracy,
            'date': current_date,
            'model_state_dict': global_model.state_dict().copy(),
            # could be averaged as the model_state_dict
            # 'lr_scheduler_dict': lr_scheduler.state_dict().copy(),
            # 'optimizer_dict': optimizer.state_dict().copy(),
        }
        torch.save(last_model, pjoin(modelpath, modelname + '_last.ckpt'))
        print(f"Last model in round {round+1} saved with Training Accuracy: {acc_avg:.2f} %")
        # return model.state_dict()
        

def train_text_class_fl_inner(global_model, train_loader, indexes, optimizer_type, lr, scheduler_type, scheduler_warmup_steps, num_epochs, device='cuda', progress_bar_flag=True, progress_bar=None):
    # Make a deep copy of the global model to ensure the original global model is not modified
    model = copy.deepcopy(global_model).to(device)
    
    # Create a new DataLoader that only samples from the specified indexes
    sampler = SubsetRandomSampler(indexes)
    train_loader_subset = DataLoader(train_loader.dataset, batch_size=train_loader.batch_size, sampler=sampler, drop_last=train_loader.drop_last)
        
    # Initialize the optimizer for the new local model
    optimizer = create_optimizer(
            optimizer_type, model, lr)
    
    # Initialize the learning rate scheduler for the new local optimizer
    num_training_steps = num_epochs * len(train_loader_subset)
    lr_scheduler = create_scheduler(
        scheduler_type, optimizer, num_training_steps, scheduler_warmup_steps)
        
    # inner training loop
    for epoch in range(num_epochs):
        model.train()
        accumulated_loss, steps, correct, total = 0, 0, 0, 0
        # train on the partitioned dataset
        for i, batch in enumerate(train_loader_subset):
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
            if progress_bar_flag:
                progress_bar.update(1)
            accumulated_loss += loss.item()
            total += targets.size(0)
            correct += (predicted == targets).cpu().sum().item()
            steps += 1
            # track train accuracy
            # if (i+1) % 100 == 0:
            #     loss_step = accumulated_loss/steps
            #     accuracy_step = 100 * correct / total
            #     print('Local Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f} %'
            #             .format(epoch+1, num_epochs, i+1, total_steps_per_epoch, loss_step, accuracy_step))
        loss_epoch = accumulated_loss/steps
        accuracy_epoch = 100 * correct / total
        print('Local Epoch [{}/{}] finished, Loss: {:.4f}, Accuracy: {:.2f} %'.format(epoch+1, num_epochs, loss_epoch, accuracy_epoch))
    # return the last epoch weights, local loss and local accuracy
    return model.state_dict(), loss_epoch, accuracy_epoch

# TRAINING FOR CIFAR DATASETS
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
