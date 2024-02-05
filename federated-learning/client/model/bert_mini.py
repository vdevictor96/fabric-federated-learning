import torch
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from os.path import join as pjoin



def get_bert_mini_model(device='cuda'):
    model = AutoModelForSequenceClassification.from_pretrained(
    "prajjwal1/bert-mini")  # v1 and v2
    model.to(device)
    return model

def load_bert_mini_model(model_path, device='cuda'):
    model = AutoModelForSequenceClassification.from_pretrained(
    "prajjwal1/bert-mini")  # v1 and v2
    # Load the saved best model state
    model_info = torch.load(model_path)
    ml_mode = model_info['ml_mode']
    ml_mode_string = 'Centralised Machine Learning' if ml_mode == 'ml' else 'Federated Learning' if ml_mode == 'fl' else 'Blockchain-Based Federated Learning'
    if ml_mode == 'ml':
        print(f"Loaded model bert-mini from date {model_info['date']}. Trained with {ml_mode_string} technology.\nEpoch {model_info['epoch']}, lr: {model_info['lr']}, optimizer: {model_info['optimizer']}\nTrain accuracy: {model_info['tr_acc']:.2f} %, Validation accuracy: {model_info['val_acc']:.2f} %")
    else:
        print(f"Loaded model bert-mini from date {model_info['date']}. Trained with {ml_mode_string} technology.\nRound {model_info['round']}, lr: {model_info['lr']}, optimizer: {model_info['optimizer']}\nAverage train accuracy: {model_info['tr_acc']:.2f} %, Validation accuracy: {model_info['val_acc']:.2f} %")
    model_state = model_info['model_state_dict']
    
    # Translate model state keys in case it was trained with DP
    keyword="_module."
    replacement=""
    keys_to_replace = [key for key in model_state if keyword in key]

    for key in keys_to_replace:
        new_key = key.replace(keyword, replacement)
        model_state[new_key] = model_state.pop(key)
        
    # Load the state dictionary into the model
    model.load_state_dict(model_state)
    model.to(device)
    return model

def get_bert_mini_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-mini")
    return tokenizer
