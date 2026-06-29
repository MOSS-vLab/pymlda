import pandas as pd

def load_excel(path, sheet_name):
    return pd.read_excel(path, sheet_name=sheet_name)

def save_dataframe(df, path):
    df.to_csv(path, index=False)