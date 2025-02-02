import pandas as pd

def load_data(file_path):
    """Loads data from a CSV file."""
    return pd.read_csv(file_path)


def save_data(df, file_path):
    """Saves DataFrame to a CSV file."""
    df.to_csv(file_path, index=False)