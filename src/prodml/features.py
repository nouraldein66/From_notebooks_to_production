from sklearn.feature_extraction import DictVectorizer
import pandas as pd
from pathlib import Path
from prodml import settings, load_data, split_data

def add_duration(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'duration' column to the DataFrame, calculated as the difference between dropoff and pickup times in minutes.

    Args:
        df (pd.DataFrame): The input DataFrame containing 'lpep_pickup_datetime' and 'lpep_dropoff_datetime'
    Returns:
        pd.DataFrame: The DataFrame with the added 'duration' column.
    """
    df['duration'] = (df['lpep_dropoff_datetime'] - df['lpep_pickup_datetime']).dt.total_seconds() / 60
    return df

def filter_rows(df: pd.DataFrame) -> pd.DataFrame:
    duration_filter = (df['duration'] >= settings.min_duration) & (df['duration'] <= settings.max_duration)
    trip_distance_filter = (df['trip_distance'] >= settings.min_trip_distance) & (df['trip_distance'] <= settings.max_trip_distance)
    df = df[duration_filter & trip_distance_filter].copy()
    return df

def pu_do(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a new column 'PU_DO' by concatenating 'PULocationID' and 'DOLocationID'.

    Args:
        df (pd.DataFrame): The input DataFrame containing 'PULocationID' and 'DOLocationID'.
    
    Returns:
        pd.DataFrame: The DataFrame with the added 'PU_DO' column.
    """
    result = df.copy()
    result['PU_DO'] = result['PULocationID'].astype(str) + '_' + result['DOLocationID'].astype(str)
    return result


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare features for model training or prediction by adding duration, filtering rows, and creating the 'PU_DO' column.

    Args:
        df (pd.DataFrame): The input DataFrame.
    Returns:
        pd.DataFrame: The DataFrame with prepared features.
    """
    result = add_duration(df)
    result = filter_rows(result)
    result = pu_do(result)
    return result

def main():
    df = load_data()
    df = prepare_features(df)

if __name__ == "__main__":   
    main()


