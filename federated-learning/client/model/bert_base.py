import torch
from transformers import BertForSequenceClassification
from transformers import BertTokenizer
from os.path import join as pjoin


def get_bert_base_model(device='cuda'):
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased")
    model.to(device)
    return model


def load_bert_base_model(model_path, device='cuda'):
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased")
    # Load the saved best model state
    model_info = torch.load(model_path)
    print(f"Loaded model from date {model_info['date']}. Epoch {model_info['epoch']}, lr: {model_info['lr']}, optimizer: {model_info['optimizer']}\nTrain accuracy: {model_info['tr_acc']:.2f} %, Validation accuracy: {model_info['val_acc']:.2f} %")
    model_state = model_info['model_state_dict']
    # Load the state dictionary into the model
    model.load_state_dict(model_state)
    model.to(device)
    return model


def get_bert_base_tokenizer():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    return tokenizer
