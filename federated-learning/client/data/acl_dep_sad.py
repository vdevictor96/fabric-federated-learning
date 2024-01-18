import pandas as pd
import torch
import transformers
from torch.utils.data import Dataset, DataLoader


def get_acl_dep_sad_dataloaders(root, tokenizer, train_size=0.8, eval_size=0.2, train_batch_size=4, eval_batch_size=2, max_len=512, seed=200):
    train_dataset, eval_dataset = get_acl_dep_sad_datasets(
        root, tokenizer, train_size, eval_size, max_len, seed)

    train_params = {'batch_size': train_batch_size,
                    # 'generator': torch.Generator(device=device),
                    'shuffle': True,
                    'num_workers': 0
                    }
    eval_params = {'batch_size': eval_batch_size,
                   #    'generator': torch.Generator(device=device),
                   'shuffle': False,
                   'num_workers': 0
                   }

    train_loader = DataLoader(train_dataset, **train_params)
    eval_loader = DataLoader(eval_dataset, **eval_params)

    return train_loader, eval_loader


def get_acl_dep_sad_test_dataloader(root, tokenizer, test_batch_size=2, max_len=512, seed=200):
    test_dataset = get_acl_dep_sad_test_dataset(
        root, tokenizer, max_len, seed)

    test_params = {'batch_size': test_batch_size,
                   #    'generator': torch.Generator(device=device),
                   'shuffle': False,
                   'num_workers': 0
                   }
    test_loader = DataLoader(test_dataset, **test_params)
    return test_loader


def get_acl_dep_sad_datasets(root, tokenizer, train_size=0.8, eval_size=0.2, max_len=512, seed=200):
    train_dataframe, eval_dataframe = get_acl_dep_sad_dataframes(
        root, train_size, eval_size, seed)
    train_dataset = AclDepressionSadness(train_dataframe, tokenizer, max_len)
    eval_dataset = AclDepressionSadness(eval_dataframe, tokenizer, max_len)
    return train_dataset, eval_dataset


def get_acl_dep_sad_dataframes(root, train_size=0.8, eval_size=0.2, seed=200):
    # Check if the sizes add up to 1
    if train_size + eval_size != 1:
        raise ValueError(
            "The sum of train_size and eval_size must be 1.")

    # Load the dataset
    df = pd.read_csv(root)
    # Create train, eval splits
    train_dataframe = df.sample(
        frac=train_size, random_state=seed)
    eval_dataframe = df.drop(
        train_dataframe.index).reset_index(drop=True)
    train_dataframe = train_dataframe.reset_index(drop=True)

    return train_dataframe, eval_dataframe


def get_acl_dep_sad_test_dataset(root, tokenizer, max_len=512, seed=200):
    # Load the dataset
    test_df = pd.read_csv(root)
    test_dataframe = test_df.reset_index(drop=True)

    test_dataset = AclDepressionSadness(test_dataframe, tokenizer, max_len)
    return test_dataset


# credits for the dataset to https://github.com/tiasa2/Interpretability-of-Federated-Learning-for-Fine-grained-Classification-of-Sadness-and-Depression
class AclDepressionSadness(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def get_labels(self):
        labels = []
        for i in range(self.len):
            labels.append(self.data.iloc[i].target)
        return labels
    
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
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'attention_mask': torch.tensor(mask, dtype=torch.long),
            'label': torch.tensor(self.data.target[index])
            # , dtype=torch.long)
        }

    def __len__(self):
        return self.len
