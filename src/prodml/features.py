from sklearn.feature_extraction import DictVectorizer
from data import load_data
import pandas as pd
from pathlib import Path
import config

def extract_features(df: pd.DataFrame, vectorizer: DictVectorizer = None) -> tuple:
    """
    Extract features and target variable from the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        vectorizer (DictVectorizer, optional): An instance of DictVectorizer for feature extraction. If None, a new instance will be created.
    
    Returns:
        X (sparse matrix): The feature matrix.
        y (numpy array): The target variable (duration).
    """
    if vectorizer is None:
        vectorizer = DictVectorizer(sparse=True)

    df['duration'] = (df['lpep_dropoff_datetime'] - df['lpep_pickup_datetime']).dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    df_features = df[['trip_distance', 'PULocationID', 'DOLocationID']]
    X = vectorizer.fit_transform(df_features.to_dict(orient='records'))
    y = df.duration.values

    return X, y

def main():
    root_dir = Path(__file__).resolve().parents[2]
    target_path = config.data_path
    file_path = root_dir / target_path
    df = load_data(file_path)
    dict_vect = DictVectorizer(sparse=True)
    X, y = extract_features(df, dict_vect)
    print(X.shape)
    print(y.shape)

if __name__ == "__main__":   
    main()


