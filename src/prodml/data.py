import pandas as pd
from pathlib import Path
from prodml import settings
from sklearn.model_selection import train_test_split

def load_data() -> pd.DataFrame:
    """
    Load data from a Parquet file into a pandas DataFrame.

    Returns:
        pd.DataFrame: A DataFrame containing the loaded data.
    """
    
    df = pd.read_parquet(settings.data_path)
    return df

def split_data(df) -> tuple:
    df_train, df_val = train_test_split(df, test_size=settings.validation_size, random_state=settings.random_state)
    return df_train, df_val

def main():
    df = load_data()
    df_train, df_val = split_data(df)
if __name__ == "__main__":
    main()