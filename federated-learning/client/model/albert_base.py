import torch
from transformers import AlbertForSequenceClassification
from transformers import AlbertTokenizer
from os.path import join as pjoin


def get_albert_base_model(device='cuda'):
    model = AlbertForSequenceClassification.from_pretrained(
        'albert-base-v2')
    model.to(device)
    return model


def load_albert_base_model(model_path, device='cuda'):
    model = AlbertForSequenceClassification.from_pretrained(
        'albert-base-v2')
    # Load the saved best model state
    model_info = torch.load(model_path)
    print(f"Loaded model from date {model_info['date']}. Epoch {model_info['epoch']}, lr: {model_info['lr']}, optimizer: {model_info['optimizer']}\nTrain accuracy: {model_info['tr_acc']:.2f} %, Validation accuracy: {model_info['val_acc']:.2f} %")
    model_state = model_info['model_state_dict']
    # Load the state dictionary into the model
    model.load_state_dict(model_state)
    model.to(device)
    return model


def get_albert_base_tokenizer():
    tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
    return tokenizer
