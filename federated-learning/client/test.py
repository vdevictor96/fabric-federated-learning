import torch
import numpy as np

def test_text_class(model, test_loader, device='cuda'):
    # In test phase, we don't need to compute gradients (for memory efficiency)
    model.eval()
    correct = 0
    total = 0
    
    # Initialize lists to store predictions and actual labels
    # predictions = []
    # true_labels = []
    
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
            # predictions.extend(logits.detach().cpu().numpy())
            # true_labels.extend(targets.detach().cpu().numpy())

    
    

    test_accuracy = (correct / total) * 100
    print('Accuracy of the network on the {} test sentences: {:.2f} %'.format(
        total, test_accuracy))

    # Other alternative: Calculating accuracy
    # predicted_labels = np.argmax(predictions, axis=1)
    # accuracy = np.mean(predicted_labels == true_labels)
    # print(f"Test Accuracy: {accuracy}")
        
    return test_accuracy

    
    
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