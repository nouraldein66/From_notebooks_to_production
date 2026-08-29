import pickle
from sklearn.feature_extraction import DictVectorizer
from data import load_data
from features import extract_features
from pathlib import Path
import config

class Predictor(object):
    def __init__(self, model_path: str = None, model = None):
        """
        Initialize the Predictor with a trained model.

        Args:
            model_path (str): The path to the trained model file.
        """
        if model_path is not None:
            self.model_path = model_path
            self.model = self.load_model()
        elif model is not None:
            self.model = model
        else:
            raise ValueError("Either model_path or model must be provided.")

    def load_model(self):
        """
        Load the trained model from the specified path.

        Returns:
            model: The loaded trained model.
        """
        with open(self.model_path, "rb") as file:
            model = pickle.load(file)
        return model

    def predict(self, X):
        """
        Make predictions using the trained model.

        Args:
            X: The feature matrix for prediction.

        Returns:
            numpy array: The predicted values.
        """
        return self.model.predict(X)


def main():
    # Example usage
    model_path = "models/trained_model.pkl"
    predictor = Predictor(model_path=model_path)

    # Load data and extract features for prediction
    root_dir = Path(__file__).resolve().parents[2]
    target_path = config.data_path
    file_path = root_dir / target_path
    df = load_data(file_path)
    dict_vect = DictVectorizer(sparse=True)
    X, y = extract_features(df, dict_vect)

    # Make predictions
    predictions = predictor.predict(X)
    print(predictions[:5])  # Print the first 5 predictions

if __name__ == "__main__":
    main()