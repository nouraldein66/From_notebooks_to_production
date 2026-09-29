import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from prodml import data, features, config, logging_conf
from pathlib import Path
import logging

FeatureDict = dict[str, str | float]
logger = logging.getLogger(__name__)

def load_model(model_path: Path) -> tuple[LinearRegression, DictVectorizer]:
    """
    Load a trained model and DictVectorizer from a file.

    Args:
        model_path (Path): The path to the model file.
    Returns:
        tuple: A tuple containing the loaded LinearRegression model and DictVectorizer.
    """
    with open(model_path, "rb") as file:
        model, dict_vect = pickle.load(file)
    return model, dict_vect

class Predictor(object):
    def __init__(self,
                 model: LinearRegression,
                 vectorizer: DictVectorizer):
        """
        Initialize the Predictor with a trained model.
        """
        self.model = model
        self.dict_vect = vectorizer


    def predict_single(self, features: FeatureDict) -> float:
        logger.debug(f"Predicting for features: {features}")
        trip_distance = features.get("trip_distance")
        if trip_distance > 100:
            logger.warning(f"Trip distance {trip_distance} is unusually high. Prediction may be unreliable.")

        X = self.dict_vect.transform([features])
        prediction = self.model.predict(X)
        return prediction[0]

    def predict_batch(self, features_list: list[FeatureDict]) -> list[float]:
        logger.debug(f"Predicting for batch of features: {features_list}")
        X = self.dict_vect.transform(features_list)
        predictions = self.model.predict(X)
        return predictions.tolist()


def main():
    # Example usage
    logging_conf.setup_logging(log_level=logging.DEBUG)
    model, dict_vect = load_model(config.settings.model_path)

    predictor = Predictor(model, dict_vect)

    # Load data and extract features for prediction
    df = data.load_data()
    _, df_val = data.split_data(df)
    df_val = features.prepare_features(df_val)
    featuresDict = df_val[["PU_DO", "trip_distance"]].to_dict(orient='records')

    # Make predictions for the validation set
    predictions = predictor.predict_batch(featuresDict)
    logger.info(f"Predictions for validation set: {predictions[:5]}")  # Log first 5 predictions

    # Make a single prediction
    single_features = {"PU_DO": "1_2", "trip_distance": 120.0}
    single_prediction = predictor.predict_single(single_features)
    logger.info(f"Single prediction for features {single_features}: {single_prediction}")

if __name__ == "__main__":
    main()