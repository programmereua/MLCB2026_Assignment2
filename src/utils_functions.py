

import pandas as pd
import numpy as np

def load_data(path):
    return pd.read_csv(path)

def missing_values(df):
    return df.isnull().sum()

def clean_data(df):
    df = df.copy()
    df = df.fillna(df.median(numeric_only=True))
    df = df.fillna(df.mode().iloc[0])
    return df
