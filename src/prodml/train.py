from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
from prodml import data, logging_conf, features, config
from sklearn.feature_extraction import DictVectorizer
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging

logger = logging.getLogger(__name__)

def train_model(train_df):
    """
    Train a linear regression model using the provided features and target variable.

    Args:
        X_train (sparse matrix): The feature matrix for training.
        y_train (numpy array): The target variable for training (duration).

    Returns:
        model (LinearRegression): The trained linear regression model.
    """
    # Extract features and target variable from the training DataFrame
    dict_vect = DictVectorizer(sparse=True)
    X_train = dict_vect.fit_transform(train_df[["PU_DO", "trip_distance"]].to_dict(orient='records'))
    y_train = train_df["duration"].values
    
    # Create a linear regression model
    model = LinearRegression()

    # Train the model on the training data
    model.fit(X_train, y_train)

    return model, dict_vect

def evaluate_model(model, dict_vect, val_df):
    """
    Evaluate the trained model on the validation set.

    Args:
        model (LinearRegression): The trained linear regression model.
        dict_vect (DictVectorizer): The DictVectorizer used for feature extraction.
        val_df (pd.DataFrame): The validation DataFrame containing features and target variable.

    Returns:
        dict: A dictionary containing evaluation metrics (RMSE and MAE).
    """
    # Extract features and target variable from the validation DataFrame
    X_val = dict_vect.transform(val_df[["PU_DO", "trip_distance"]].to_dict(orient='records'))
    y_val = val_df["duration"].values

    # Make predictions on the validation set
    y_pred = model.predict(X_val)

    # Calculate evaluation metrics
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)

    return rmse, mae

def save_model(model, dict_vect):
    """
    Save the trained model and DictVectorizer to a file.

    Args:
        model (LinearRegression): The trained linear regression model.
        dict_vect (DictVectorizer): The DictVectorizer used for feature extraction.
    """
    with open(config.settings.model_path, "wb") as file:
        pickle.dump((model, dict_vect), file)

def main():
    # Example usage
    logging_conf.setup_logging(log_level=logging.DEBUG)

    df = data.load_data()
    df_train, df_val = data.split_data(df)
    logger.debug(
        f"Training set size: {len(df_train)}, Validation set size: {len(df_val)}"
    )

    df_train = features.prepare_features(df_train)

    model, dict_vect = train_model(df_train)
    save_model(model, dict_vect)

    # Evaluate the model on the validation set
    df_val = features.prepare_features(df_val)
    rmse, mae = evaluate_model(model, dict_vect, df_val)

    logger.info(f"Validation RMSE: {rmse:.4f}, Validation MAE: {mae:.4f}")
    
    

if __name__ == "__main__":
    main()