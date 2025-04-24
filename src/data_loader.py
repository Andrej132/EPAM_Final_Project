import os
import pandas as pd

def load_data(train_file: str, test_file: str, data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads train and test datasets from the specified directory.

    Args:
        train_file (str): Name of the train dataset file.
        test_file (str): Name of the test dataset file.
        data_dir (str): Path to the directory containing the datasets. Default is '../data/raw/'.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the train and test datasets as pandas DataFrames.
    """
    train_path = os.path.join(data_dir, train_file)
    test_path = os.path.join(data_dir, test_file)

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    return train_data, test_data
