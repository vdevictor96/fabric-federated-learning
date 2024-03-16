import pandas as pd
import numpy as np
import os


def split_csv_dataset_train_test(file_path, train_val_path, test_path, test_size=0.2, seed=200):
    # Check if the sizes add up to 1
    if test_size < 0 or test_size > 1:
        raise ValueError(
            "The test_size must be between 0 and 1.")

    # Determine the file extension
    _, ext = os.path.splitext(file_path)

    # Load the dataset based on file extension
    if ext.lower() in ['.xls', '.xlsx']:
        df = pd.read_excel(file_path)
    elif ext.lower() == '.csv':
        df = pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported file type. Please provide a .csv or .xlsx file.")
    
    # Separating the data based on the 'target' values
    df_0 = df[df['target'] == 0]
    df_1 = df[df['target'] >= 1]


    # Splitting each subset into training/validation and test sets for each target value
    train_val_0, test_0 = split_data(df_0, test_size=0.2, seed=seed)
    train_val_1, test_1 = split_data(df_1, test_size=0.2, seed=seed)

    # Combining the training/validation sets and the test sets from both subsets
    train_val = pd.concat([train_val_0, train_val_1])
    test = pd.concat([test_0, test_1])

    # Shuffle the datasets
    train_val = train_val.sample(frac=1, random_state=seed).reset_index(drop=True)
    test = test.sample(frac=1, random_state=seed).reset_index(drop=True)

    # You can now use 'train_val' for training and validation, and 'test' for testing
    train_val.to_csv(train_val_path, index=False)
    test.to_csv(test_path, index=False)


# Function to split data into training/validation and test sets
def split_data(df, test_size=0.2, seed=200):
    # Shuffle the data
    shuffled_df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    # Calculate the number of test samples
    n_test = int(len(shuffled_df) * test_size)
    # Split the data
    test_data = shuffled_df.iloc[:n_test]
    train_val_data = shuffled_df.iloc[n_test:]
    return train_val_data, test_data

def balance_train_split(train_csv_path, balanced_csv_path=None):
    """
    Ensures there are equal numbers of samples for each target in the training dataset.
    If balanced_csv_path is not provided, the original file is overwritten.
    
    Parameters:
    - train_csv_path: Path to the training split CSV file.
    - balanced_csv_path: Optional; Path to save the balanced CSV file. If None, overwrites the original file.
    """
    # Load the dataset
    df = pd.read_csv(train_csv_path)
    
    # Count the instances of each class
    count_class_0, count_class_1 = df.target.value_counts()
    
    # Divide by class
    df_class_0 = df[df['target'] == 0]
    df_class_1 = df[df['target'] == 1]
    
    # Balance the dataset
    if count_class_0 > count_class_1:
        df_class_0_under = df_class_0.sample(count_class_1)
        df_balanced = pd.concat([df_class_0_under, df_class_1], axis=0)
    else:
        df_class_1_under = df_class_1.sample(count_class_0)
        df_balanced = pd.concat([df_class_0, df_class_1_under], axis=0)
    
    # Shuffle the dataset
    df_balanced = df_balanced.sample(frac=1, random_state=200).reset_index(drop=True)
    
    # Determine the path for saving the balanced dataset
    save_path = balanced_csv_path if balanced_csv_path else train_csv_path
    
    # Save the balanced dataset
    df_balanced.to_csv(save_path, index=False)

    print(f"Balanced dataset saved to {save_path}")