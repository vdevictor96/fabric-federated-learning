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
    if ml_mode == 'standalone':
        print(f"Loaded model from date {model_info['date']}. Epoch {model_info['epoch']}, lr: {model_info['lr']}, optimizer: {model_info['optimizer']}\nTrain accuracy: {model_info['tr_acc']:.2f} %, Validation accuracy: {model_info['val_acc']:.2f} %")
    else:
        print(f"Loaded model from date {model_info['date']}. Round {model_info['round']}, lr: {model_info['lr']}, optimizer: {model_info['optimizer']}\nTrain accuracy: {model_info['tr_acc']:.2f} %, Validation accuracy: {model_info['val_acc']:.2f} %")
    model_state = model_info['model_state_dict']
    # Load the state dictionary into the model
    model.load_state_dict(model_state)
    model.to(device)
    return model

def get_bert_mini_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-mini")
    return tokenizer
