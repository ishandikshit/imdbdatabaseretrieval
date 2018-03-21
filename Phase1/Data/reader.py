import pandas as pd


def read_data(filename):
    return pd.read_csv(filename).drop_duplicates().dropna()
