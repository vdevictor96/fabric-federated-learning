from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer



def get_bert_tiny_model(device='cuda'):
    model = AutoModelForSequenceClassification.from_pretrained(
    "prajjwal1/bert-tiny")  # v1 and v2
    model.to(device)
    return model

def get_bert_tiny_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
    return tokenizer
