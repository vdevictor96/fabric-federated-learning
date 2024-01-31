import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
# progress bar
from tqdm.auto import tqdm


def calculate_precision(true_labels, predicted_labels):
    return precision_score(true_labels, predicted_labels, average='macro')

def calculate_recall(true_labels, predicted_labels):
    return recall_score(true_labels, predicted_labels, average='macro')

def calculate_f1_score(true_labels, predicted_labels):
    return f1_score(true_labels, predicted_labels, average='macro')


def test_text_class(model, test_loader, device='cuda', progress_bar_flag=True):
    # In test phase, we don't need to compute gradients (for memory efficiency)
    model.eval()
    correct = 0
    total = 0
    if progress_bar_flag:
        progress_bar = tqdm(range(len(test_loader)))
    # Initialize lists to store predictions and actual labels
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            # ids = batch['input_ids']
            # mask = batch['attention_mask']
            # targets = batch['label']
            # Move batch to GPU
            ids = batch['input_ids'].to(device=device, dtype=torch.long)
            mask = batch['attention_mask'].to(device=device, dtype=torch.long)
            targets = batch['label'].to(device=device, dtype=torch.long)

            # Get model predictions
            outputs = model(ids, mask)
            logits = outputs.logits
            predicted = torch.argmax(logits, dim=-1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            # Convert predictions to CPU and numpy for metric calculation
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(targets.cpu().numpy())
            if progress_bar_flag:
                progress_bar.update(1)

    
    

    test_accuracy = (correct / total) * 100
    precision = calculate_precision(true_labels, predictions)
    recall = calculate_recall(true_labels, predictions)
    f1 = calculate_f1_score(true_labels, predictions)
    
    print(f'Nº of test samples: {total}')
    print(f'Accuracy: {test_accuracy:.2f}%')
    print(f'Precision: {precision*100:.2f}%')
    print(f'Recall: {recall*100:.2f}%')
    print(f'F1 Score: {f1*100:.2f}%')
    
    # Other alternative: Calculating accuracy
    # predicted_labels = np.argmax(predictions, axis=1)
    # accuracy = np.mean(predicted_labels == true_labels)
    # print(f"Test Accuracy: {accuracy}")
        
    return test_accuracy, precision, recall, f1

    
    
def test_cifar(model, input_size, test_loader, device='cuda'):
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