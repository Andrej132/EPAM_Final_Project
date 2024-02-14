import pandas as pd


def load_data():
    train_dataset = pd.read_csv(f"vol/data/raw/train.csv")
    test_dataset = pd.read_csv(f"vol/data/raw/test.csv")

    return train_dataset, test_dataset