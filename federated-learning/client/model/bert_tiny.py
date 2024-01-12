import torch
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from os.path import join as pjoin



def get_bert_tiny_model(device='cuda'):
    model = AutoModelForSequenceClassification.from_pretrained(
    "prajjwal1/bert-tiny")  # v1 and v2
    model.to(device)
    return model

def load_bert_tiny_model(modelpath, modelname, device='cuda'):
    model = AutoModelForSequenceClassification.from_pretrained(
    "prajjwal1/bert-tiny")  # v1 and v2
    # Load the saved best model state
    model_state = torch.load(pjoin(modelpath, modelname))
    # Load the state dictionary into the model
    model.load_state_dict(model_state)
    model.to(device)
    return model

def get_bert_tiny_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
    return tokenizer
