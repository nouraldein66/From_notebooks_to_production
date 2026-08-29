from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
from data import load_data
from features import extract_features
from sklearn.feature_extraction import DictVectorizer
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error
import config


def train_model(X_train, y_train):
    """
    Train a linear regression model using the provided features and target variable.

    Args:
        X_train (sparse matrix): The feature matrix for training.
        y_train (numpy array): The target variable for training (duration).

    Returns:
        model (LinearRegression): The trained linear regression model.
    """

    # Create a linear regression model
    model = LinearRegression()

    # Train the model on the training data
    model.fit(X_train, y_train)

    return model

def main():
    # Example usage

    root_dir = Path(__file__).resolve().parents[2]
    target_path = config.data_path
    file_path = root_dir / target_path
    df = load_data(file_path)
    dict_vect = DictVectorizer(sparse=True)
    X, y = extract_features(df, dict_vect)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = train_model(X_train, y_train)
    with open("models/trained_model.pkl", "wb") as file:
        pickle.dump(model, file)
    
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)

    print(f"Linear Regression RMSE: {rmse:.2f} minutes")
    print(f"Mean Absolute Error Score: {mae:.2f}")

    print("Model trained & saved successfully.")


if __name__ == "__main__":
    main()