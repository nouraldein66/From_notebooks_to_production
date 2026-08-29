import pandas as pd
from pathlib import Path
import config

def load_data(file_path: Path) -> pd.DataFrame:
    """
    Load data from a Parquet file into a pandas DataFrame.

    Args:
        file_path (Path): The path to the Parquet file.

    Returns:
        pd.DataFrame: A DataFrame containing the loaded data.
    """
    df = pd.read_parquet(file_path)
    return df

def main():
    root_dir = Path(__file__).resolve().parents[2]
    target_path = config.data_path
    print(target_path)
    file_path = root_dir / target_path
    df = load_data(file_path)
    print(df.head(5))
    
if __name__ == "__main__":
    main()