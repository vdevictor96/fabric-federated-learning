import torch
from transformers import DistilBertForSequenceClassification
from transformers import DistilBertTokenizer
from os.path import join as pjoin


def get_distilbert_base_model(device='cuda'):
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased")
    model.to(device)
    return model


def load_distilbert_base_model(model_path, device='cuda'):
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased")
    # Load the saved best model state
    model_info = torch.load(model_path)
    print(f"Loaded model from date {model_info['date']}. Epoch {model_info['epoch']}, lr: {model_info['lr']}, optimizer: {model_info['optimizer']}\nTrain accuracy: {model_info['tr_acc']:.2f} %, Validation accuracy: {model_info['val_acc']:.2f} %")
    model_state = model_info['model_state_dict']
    # Load the state dictionary into the model
    model.load_state_dict(model_state)
    model.to(device)
    return model


def get_distilbert_base_tokenizer():
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    return tokenizer
