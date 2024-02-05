from .utils import iid_partition, non_iid_partition, create_optimizer, create_scheduler, translate_state_dict_keys, filter_trainable_weights
from .services.utils import deserialize_model_msgpack
from .services.gateway_client import submit_model, aggregate_models, get_model
from .aggregators import federated_aggregate
import torch
import torch.nn as nn
import torchvision
import sys
import os
import copy
import concurrent.futures
from opacus import PrivacyEngine
from torch.utils.data import DataLoader, SubsetRandomSampler
from datetime import datetime
from os.path import join as pjoin
# progress bar
from tqdm.auto import tqdm


def train_text_class(model, model_save_path, train_loader, eval_loader, optimizer, lr, lr_scheduler, num_epochs, device='cuda', eval_flag=True, save_model=True, progress_bar_flag=True, dp_epsilon=0.0, dp_delta=3e-3):
    total_steps_per_epoch = len(train_loader)
    total_steps = num_epochs * total_steps_per_epoch
    if eval_flag:
        total_steps += num_epochs * len(eval_loader)

    if progress_bar_flag:
        progress_bar = tqdm(range(total_steps))
    
    # Initialize variables to track the best model
    best_val_accuracy = 0.0
    best_model_state = None
    best_epoch = 0
    current_date = datetime.now().strftime("%d-%m-%Y %H:%M")

    model.train()
    # Initialize the differential privacy engine for the new local optimizer
    if dp_epsilon > 0.0:
        # Create differential privacy engine if differential privacy is enabled
        privacy_engine = PrivacyEngine()
        # Integrate the privacy engine with the model, optimizer and data loader
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            target_delta=dp_delta,
            target_epsilon=dp_epsilon,
            epochs=num_epochs,
            max_grad_norm=0.1,
            poisson_sampling=False,
        )

    for epoch in range(num_epochs):
        model.train()
        accumulated_loss, steps, correct, total = 0, 0, 0, 0
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
        print('Epoch [{}/{}] Loss: {:.4f}, Accuracy: {:.2f} %'.format(
            epoch+1, num_epochs, loss_epoch, accuracy_epoch))
        print("-------------------------------")
        # ---------------------- Validation ----------------------
        if eval_flag:
            print('-------- Validation --------')
            model.eval()
            val_loss, val_correct, val_total = 0, 0, 0
            with torch.no_grad():
                for i, batch in enumerate(eval_loader):
                    ids = batch['input_ids'].to(
                        device=device, dtype=torch.long)
                    mask = batch['attention_mask'].to(
                        device=device, dtype=torch.long)
                    targets = batch['label'].to(
                        device=device, dtype=torch.long)

                    outputs = model(ids, mask, labels=targets)
                    loss = outputs.loss
                    predicted = torch.argmax(outputs.logits, dim=-1)

                    val_loss += loss.item()
                    val_total += targets.size(0)
                    val_correct += (predicted == targets).cpu().sum().item()
                    if progress_bar_flag:
                        progress_bar.update(1)
            val_accuracy = 100 * val_correct / val_total
            print('Validation Loss: {:.4f}, Validation Accuracy: {:.2f} %'.format(
                val_loss / len(eval_loader), val_accuracy))
            print('-------- Validation finished --------')
            # Check if this is the best model based on validation accuracy
            if val_accuracy >= best_val_accuracy:
                best_val_accuracy = val_accuracy
                # Translate model state keys in case it was trained with DP
                best_model_state = translate_state_dict_keys(model.state_dict().copy())
                best_model = {
                    'ml_mode': 'ml',
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
                print(
                    f"Updated best model in epoch {best_epoch} saved with Validation Accuracy: {best_val_accuracy:.2f} %")
                print("-------------------------------")
                if save_model:
                    torch.save(best_model, model_save_path + '_best.ckpt')

    # ---------------------- Saving Models ----------------------

    if best_val_accuracy != 0.0:
        print(
            f"Best model in epoch {best_epoch} saved with Validation Accuracy: {best_val_accuracy:.2f} %")
        # return best_model_state
    else:
        # Save the last model checkpoint
        last_model = {
            'ml_mode': 'ml',
            'epoch': epoch+1,
            'lr': lr,
            'optimizer': optimizer.__class__.__name__,
            'tr_acc': accuracy_epoch,
            'val_acc': best_val_accuracy,  # 0.0
            'date': current_date,
            # Translate model state keys in case it was trained with DP
            'model_state_dict': translate_state_dict_keys(model.state_dict().copy()),
            # 'lr_scheduler_dict': lr_scheduler.state_dict().copy(),
            # 'optimizer_dict': optimizer.state_dict().copy(),
        }
        if save_model:
            torch.save(last_model, model_save_path + '_last.ckpt')
        print(
            f"Last model in Epoch {epoch+1} saved with Training Accuracy: {accuracy_epoch:.2f} %")
        return model if eval_flag is False else None

def train_text_class_fl(model, fl_mode, fed_alg, mu, model_name, model_save_path, layers, train_loader, eval_loader, optimizer_type, lr, scheduler_type, scheduler_warmup_steps, num_epochs, concurrency_flag, device='cuda', eval_flag=True, save_model=True, progress_bar_flag=True, num_rounds=10, num_clients=5, dp_epsilon=0.0, dp_delta=3e-3, data_distribution='iid'):
    # Set the progress bar
    total_steps = num_rounds * num_epochs * len(train_loader)
    if eval_flag:
        total_steps += num_rounds * len(eval_loader)

    if concurrency_flag is False:
        if progress_bar_flag:
            progress_bar = tqdm(range(total_steps))
        else:
            progress_bar = None
    else:  # deactivate progress bar if concurrency is enabled
        progress_bar_flag = False
        progress_bar = None

    # partition the training dataset
    if data_distribution == 'iid':
        partitioned_indexes = iid_partition(train_loader.dataset, num_clients)
    else:  # non-iid
        partitioned_indexes = non_iid_partition(
            train_loader.dataset, num_clients)
    # Initialize variables to track the best model
    best_val_accuracy = 0.0
    best_model_state = None
    best_round = 0
    current_date = datetime.now().strftime("%d-%m-%Y %H:%M")

    global_model = model
    # outer training loop
    for round in range(num_rounds):
        trainable_weights, local_loss, local_acc = [], [], []
        print(f"\nRound {round+1} of {num_rounds}")
        print("-------------------------------")
        # inner training loop
        # Parallel training for each client
        if concurrency_flag:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(train_text_class_fl_inner, global_model, model_name, fl_mode, fed_alg, mu, layers, client, num_clients, train_loader, partitioned_indexes[client], optimizer_type, lr,
                                           scheduler_type, scheduler_warmup_steps, dp_epsilon, dp_delta, num_epochs, device, progress_bar_flag, progress_bar) for client in range(num_clients)]
                for future in concurrent.futures.as_completed(futures):
                    c_weights, c_local_loss, c_local_acc = future.result()
                    if fl_mode == 'fl':
                        # if fl_mode is bcfl, the local weights have been sent to the blockchain
                        # trainable_weights.append(copy.deepcopy(c_weights))
                        trainable_weights.append(c_weights)
                    local_loss.append(c_local_loss)
                    local_acc.append(c_local_acc)
        else:  # sequential
            for client in range(num_clients):
                c_weights, c_local_loss, c_local_acc = train_text_class_fl_inner(
                    global_model, model_name, fl_mode, fed_alg, mu, layers, client, num_clients, train_loader, partitioned_indexes[client], optimizer_type, lr, scheduler_type, scheduler_warmup_steps, dp_epsilon, dp_delta, num_epochs, device, progress_bar_flag, progress_bar)
                if fl_mode == 'fl':
                    # if fl_mode is bcfl, the local weights have been sent to the blockchain
                    # trainable_weights.append(copy.deepcopy(c_weights))
                    trainable_weights.append(c_weights)
                local_loss.append(c_local_loss)
                local_acc.append(c_local_acc)
        # loss and accuracy metrics from average of local loss and accuracy
        loss_avg = sum(local_loss) / len(local_loss)
        acc_avg = sum(local_acc) / len(local_acc)
        print("-------------------------------")
        print('Round [{}/{}] Average Local Loss: {:.4f}, Average Local Accuracy: {:.2f} %'.format(
            round+1, num_rounds, loss_avg, acc_avg))
        print("-------------------------------")

        # aggregate the models and update the global model
        global_trainable_weights = {}
        if fl_mode == 'fl':
            global_trainable_weights = federated_aggregate(trainable_weights)
        else:  # bcfl
            # the local weights have been sent to the blockchain
            # triggering the federated aggregation
            aggregate_models(model_name, round+1)
            global_model_data = get_model(model_name + '_round_' + str(round+1))
            global_trainable_weights = deserialize_model_msgpack(
                global_model_data['modelParams'])

        # Overwrite the trainable layers of the global weights with the global trainable weights before loading the state dict
        global_weights = global_model.state_dict()
        for layer in global_trainable_weights:
            if layer in global_weights:
                global_weights[layer] = global_trainable_weights[layer]
            else:
                print(
                    f"Warning: '{layer}' not found in global model's state_dict.")

        # Load the updated state dict back into the model
        global_model.load_state_dict(global_weights)
        # ---------------------- Validation ----------------------
        if eval_flag:
            best_val_accuracy, best_round = eval_text_class_fl(global_model, model_save_path, eval_loader, best_val_accuracy, best_round, round, num_rounds, lr, optimizer_type, acc_avg, current_date,
                                                               save_model, device, progress_bar_flag, progress_bar)
    # ---------------------- Saving Models ----------------------
    if best_val_accuracy != 0.0:
        print(
            f"Best model in round {best_round} saved with Validation Accuracy: {best_val_accuracy:.2f} %")
        # return best_model_state
    elif save_model:
        save_model_text_class_fl(global_model, model_save_path,
                                 num_rounds, lr, optimizer_type, acc_avg, current_date, device)
    
    return global_model if eval_flag is False else None


def train_text_class_fl_inner(global_model, model_name, fl_mode, fed_alg, mu, layers, client, num_clients, train_loader, indexes, optimizer_type, lr, scheduler_type, scheduler_warmup_steps, dp_epsilon, dp_delta, num_epochs, device='cuda', progress_bar_flag=True, progress_bar=None):
    # print("-------------------------------")
    # print(f"Client {client+1} of {num_clients}")
    # Make a deep copy of the global model to ensure the original global model is not modified
    model = copy.deepcopy(global_model).to(device)
    model.train()

    # Create a new DataLoader that only samples from the specified indexes
    sampler = SubsetRandomSampler(indexes)

    train_loader_subset = DataLoader(
        train_loader.dataset, batch_size=train_loader.batch_size, sampler=sampler, drop_last=train_loader.drop_last)
    # Initialize the optimizer for the new local model
    optimizer = create_optimizer(
        optimizer_type, model, lr)

    # Initialize the differential privacy engine for the new local optimizer
    if dp_epsilon > 0.0:
        # Create differential privacy engine if differential privacy is enabled
        privacy_engine = PrivacyEngine()
        # Integrate the privacy engine with the model, optimizer and data loader
        model, optimizer, train_loader_subset = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader_subset,
            target_delta=dp_delta,
            target_epsilon=dp_epsilon,
            epochs=num_epochs,
            max_grad_norm=0.1,
            poisson_sampling=False,
        )
        # model, optimizer, train_loader_subset = privacy_engine.make_private(
        #     module=model,
        #     optimizer=optimizer,
        #     data_loader=train_loader_subset,
        #     noise_multiplier=0.2,
        #     max_grad_norm=0.1,
        #     poisson_sampling=False,
        # )

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

            # FedProx Modification
            if fed_alg == 'fedprox':
                # Calculate the proximal term
                proximal_term = 0
                for param, global_param in zip(model.parameters(), global_model.parameters()):
                    proximal_term += (param - global_param).norm(2)
                proximal_term = (mu / 2.0) * proximal_term

                # Include proximal term in the loss
                loss += proximal_term
            else:  # fedavg
                pass

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
        if dp_epsilon > 0.0:
            eps = privacy_engine.get_epsilon(dp_delta)
            print('Client {} of {}: Local Epoch [{}/{}] Loss: {:.4f}, Accuracy: {:.2f} %, Epsilon: {:.2f}, Delta: {:.4f}'.format(
                client+1, num_clients, epoch+1, num_epochs, loss_epoch, accuracy_epoch, eps, dp_delta))
        else:
            print('Client {} of {}: Local Epoch [{}/{}] Loss: {:.4f}, Accuracy: {:.2f} %'.format(
                client+1, num_clients, epoch+1, num_epochs, loss_epoch, accuracy_epoch))

    if dp_epsilon > 0.0:
        model.remove_hooks()

    # Only the trainable layers are added to the weights array
    trainable_weights = {}
    trainable_weights = filter_trainable_weights(
        model.state_dict(), layers, dp_epsilon > 0.0)

    if fl_mode == 'bcfl':
        # send the local weights to the blockchain
        submit_model(model_name + '_client_' + str(client+1), trainable_weights)
        return None, loss_epoch, accuracy_epoch

    # return the last epoch weights, local loss and local accuracy
    return trainable_weights, loss_epoch, accuracy_epoch


def eval_text_class_fl(model, model_save_path, eval_loader, best_val_accuracy, best_round, round, num_rounds, lr, optimizer_type, acc_avg, current_date, save_model=True, device='cuda', progress_bar_flag=True, progress_bar=None):
    # validation loss and accuracy of the model
    model.eval()
    print('-------- Validation --------')
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for i, batch in enumerate(eval_loader):
            ids = batch['input_ids'].to(
                device=device, dtype=torch.long)
            mask = batch['attention_mask'].to(
                device=device, dtype=torch.long)
            targets = batch['label'].to(
                device=device, dtype=torch.long)

            outputs = model(ids, mask, labels=targets)
            loss = outputs.loss
            predicted = torch.argmax(outputs.logits, dim=-1)

            val_loss += loss.item()
            val_total += targets.size(0)
            val_correct += (predicted == targets).cpu().sum().item()
            if progress_bar_flag:
                progress_bar.update(1)
    val_accuracy = 100 * val_correct / val_total
    print('Round [{}/{}] Global Model Validation Loss: {:.4f}, Validation Accuracy: {:.2f} %'.format(
        round+1, num_rounds, val_loss / len(eval_loader), val_accuracy))
    print('-------- Validation finished --------')
    # Check if this is the best model based on validation accuracy
    if val_accuracy >= best_val_accuracy:
        best_val_accuracy = val_accuracy
        # Translate model state keys in case it was trained with DP
        best_model_state = translate_state_dict_keys(model.state_dict().copy())
        best_model = {
            'ml_mode': 'fl',
            'round': round+1,
            'lr': lr,
            'optimizer': optimizer_type,
            'tr_acc': acc_avg,
            'val_acc': best_val_accuracy,
            'date': current_date,
            'model_state_dict': best_model_state,
            # TODO could be averaged as the model_state_dict
            # 'lr_scheduler_dict': lr_scheduler.state_dict().copy(),
            # 'optimizer_dict': optimizer.state_dict().copy(),

        }
        best_round = round + 1
        print(
            f"Updated best model in round {best_round} saved with Validation Accuracy: {best_val_accuracy:.2f} %")
        print("-------------------------------")
        if save_model:
            torch.save(best_model, model_save_path + '_best.ckpt')
    return best_val_accuracy, best_round


def save_model_text_class_fl(model, model_save_path, num_rounds, lr, optimizer_type, acc_avg, current_date, device):
    # Save the last model checkpoint
    last_model = {
        'ml_mode': 'fl',
        'round': num_rounds,
        'lr': lr,
        'optimizer': optimizer_type,
        'tr_acc': acc_avg,
        'val_acc': 0.0,
        'date': current_date,
        # Translate model state keys in case it was trained with DP
        'model_state_dict': translate_state_dict_keys(model.state_dict().copy())
        # could be averaged as the model_state_dict
        # 'lr_scheduler_dict': lr_scheduler.state_dict().copy(),
        # 'optimizer_dict': optimizer.state_dict().copy(),
    }
    torch.save(last_model, model_save_path + '_last.ckpt')
    print(
        f"Last model in round {num_rounds} saved with Training Accuracy: {acc_avg:.2f} %")
    # return model.state_dict()


# TRAINING FOR CIFAR DATASETS
def train(model, models_path, model_name, dataloaders, criterion, optimizer, learning_rate, learning_rate_decay, input_size, num_epochs, device):

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
    torch.save(model.state_dict(), pjoin(models_path, model_name + '.ckpt'))
    # torch.save(last_model, pjoin(models_path, 'last_model_{}_{}.pt'.format(model_subpath, args.num_labeled)))
