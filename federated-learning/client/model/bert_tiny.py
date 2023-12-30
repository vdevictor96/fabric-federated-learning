from transformers import AutoModel  # For BERTs
# For models fine-tuned on MNLI
from transformers import AutoModeForSequenceClassification
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")  # v1 and v2
model = AutoModelForSequenceClassification.from_pretrained(
    "prajjwal1/bert-tiny")  # v1 and v2
