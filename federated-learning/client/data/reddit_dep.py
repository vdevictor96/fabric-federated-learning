import pandas as pd
import torch
import transformers
from torch.utils.data import Dataset, DataLoader


def get_reddit_dep_dataloaders(root, tokenizer, train_size=0.8, eval_size=0.1, test_size=0.1, train_batch_size=4, eval_batch_size=2, test_batch_size=2, max_len=512, device='cuda'):
    train_dataset, eval_dataset, test_dataset = get_reddit_dep_datasets(
        root, tokenizer, train_size, eval_size, test_size, max_len)

    train_params = {'batch_size': train_batch_size,
                    'generator': torch.Generator(device=device),
                    'shuffle': True,
                    'num_workers': 0
                    }
    eval_params = {'batch_size': eval_batch_size,
                   'generator': torch.Generator(device=device),
                   'shuffle': False,
                   'num_workers': 0
                   }

    test_params = {'batch_size': test_batch_size,
                   'generator': torch.Generator(device=device),
                   'shuffle': False,
                   'num_workers': 0
                   }

    train_loader = DataLoader(train_dataset, **train_params)
    eval_loader = DataLoader(eval_dataset, **eval_params)
    test_loader = DataLoader(test_dataset, **test_params)

    return train_loader, eval_loader, test_loader


def get_reddit_dep_datasets(root, tokenizer, train_size=0.8, eval_size=0.1, test_size=0.1, max_len=512):
    train_dataframe, eval_dataframe, test_dataframe = get_reddit_dep_dataframes(
        root, train_size, eval_size, test_size)
    train_dataset = RedditDepression(train_dataframe, tokenizer, max_len)
    eval_dataset = RedditDepression(eval_dataframe, tokenizer, max_len)
    test_dataset = RedditDepression(test_dataframe, tokenizer, max_len)
    return train_dataset, eval_dataset, test_dataset


def get_reddit_dep_dataframes(root, train_size=0.8, eval_size=0.1, test_size=0.1):
    # Check if the sizes add up to 1
    if train_size + eval_size + test_size != 1:
        raise ValueError(
            "The sum of train_size, eval_size, and test_size must be 1.")
        
    # Check if train_size and eval_size sum up to 0, which would cause division by zero
    if train_size + eval_size == 0:
        raise ValueError("The sum of train_size and eval_size must not be 0 to avoid division by zero.")

    # Load the dataset
    # Load the dataset
    df = pd.read_csv(root)
    # Create train, eval, test splits
    train_eval_split = df.sample(frac=train_size+eval_size, random_state=200)
    test_dataframe = df.drop(train_eval_split.index).reset_index(drop=True)
    train_dataframe = train_eval_split.sample(
        frac=train_size/(train_size+eval_size), random_state=200)
    eval_dataframe = train_eval_split.drop(
        train_dataframe.index).reset_index(drop=True)
    train_dataframe = train_dataframe.reset_index(drop=True)

    return train_dataframe, eval_dataframe, test_dataframe


# credits for the dataset to https://github.com/whopriyam/Benchmarking-Differential-Privacy-and-Federated-Learning-for-BERT-Models
class RedditDepression(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        text = str(self.data.tweet[index])
        text = " ".join(text.split())
        inputs = self.tokenizer(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
            # return_tensors='pt'
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'target': torch.tensor(self.data.target[index])
            # , dtype=torch.long)
        }

    def __len__(self):
        return self.len
