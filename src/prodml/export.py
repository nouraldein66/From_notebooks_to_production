from onnxruntime import InferenceSession
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, StringTensorType
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle

from prodml.data import load_data, split_data
from prodml.features import prepare_features
from prodml.config import settings


def build_pipeline(model, dict_vect) -> Pipeline:
    """
    Wrap the DictVectorizer and LinearRegression into a single sklearn Pipeline
    so that skl2onnx can trace the full computation graph in one shot.

    Args:
        model: The trained LinearRegression model.
        dict_vect: The fitted DictVectorizer.
    Returns:
        Pipeline: A sklearn pipeline combining dict_vect and model.
    """
    return Pipeline(steps=[
        ("dict_vect", dict_vect),
        ("model", model),
    ])


def convert_model_to_onnx(model: LinearRegression, onnx_path, size) -> None:
    """
    Convert a trained sklearn Pipeline (DictVectorizer + LinearRegression)
    to ONNX format and save it to a file.

    The pipeline receives a list of dicts with two keys:
      - "PU_DO"          : str   → one-hot encoded by DictVectorizer
      - "trip_distance"  : float → passed through as-is by DictVectorizer

    skl2onnx needs `initial_types` to know the name, dtype, and shape of
    every input feature **before** any transformation. Because DictVectorizer
    operates on raw Python dicts, skl2onnx maps each dict key to a separate
    ONNX input tensor:

      - StringTensorType([None, 1])  for "PU_DO"
            shape [None, 1] means (batch_size, 1 string value per row)
      - FloatTensorType([None, 1])   for "trip_distance"
            shape [None, 1] means (batch_size, 1 float value per row)

    None in the first dimension means "dynamic batch size" (any number of rows).

    Args:
        pipeline: The sklearn Pipeline to convert.
        onnx_path: The path where the ONNX model will be saved.
    """
    initial_types = [
        ("features",         FloatTensorType([None, size]))
    ]

    onnx_model = convert_sklearn(
        model,
        initial_types=initial_types,
    )

    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    print(f"ONNX model saved to: {onnx_path}")


def load_onnx_model(onnx_path) -> InferenceSession:
    """
    Load an ONNX model from a file.

    Args:
        onnx_path: The path to the ONNX model file.
    Returns:
        InferenceSession: An ONNX Runtime inference session for the loaded model.
    """
    session = InferenceSession(str(onnx_path))
    return session


def load_pickle_model(model_path):
    """
    Load a trained model and DictVectorizer from a pickle file.

    Args:
        model_path: The path to the pickle model file.
    Returns:
        tuple: (LinearRegression model, DictVectorizer)
    """
    with open(model_path, "rb") as f:
        model, dict_vect = pickle.load(f)
    return model, dict_vect


def prepare_onnx_inputs(features_dicts: list[dict]) -> dict[str, np.ndarray]:
    """
    Convert a list of feature dicts into the dict-of-arrays format that
    onnxruntime expects, matching the `initial_types` declared during export.

    Each key becomes a separate numpy array with shape (N, 1).

    Args:
        features_dicts: List of dicts with keys "PU_DO" and "trip_distance".
    Returns:
        dict mapping input names to numpy arrays.
    """
    pu_do = np.array(
        [[row["PU_DO"]] for row in features_dicts], dtype=object
    )
    trip_distance = np.array(
        [[row["trip_distance"]] for row in features_dicts], dtype=np.float32
    )
    return {
        "PU_DO": pu_do,
        "trip_distance": trip_distance,
    }


def main():
    # ── 1. Load the pickle model ──────────────────────────────────────────────
    model, dict_vect = load_pickle_model(settings.model_path)
    
    # ── 2. Build and export the pipeline to ONNX ─────────────────────────────
    # pipeline = build_pipeline(model, dict_vect)
    convert_model_to_onnx(model, settings.onnx_path, len(dict_vect.feature_names_))

    # ── 3. Load validation data for smoke-testing ─────────────────────────────
    raw_data = load_data()
    _, df_val = split_data(raw_data)
    df_val = prepare_features(df_val)
    features_dicts = df_val[["PU_DO", "trip_distance"]].to_dict(orient="records")

    # ── 4. Run sklearn inference (ground truth) ───────────────────────────────
    X_val = dict_vect.transform(features_dicts)
    sklearn_preds = model.predict(X_val)

    # ── 5. Run ONNX inference ─────────────────────────────────────────────────
    sess = load_onnx_model(settings.onnx_path)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    

    # onnx_inputs = prepare_onnx_inputs(features_dicts)
    onnx_preds = sess.run([output_name], {input_name: X_val.toarray().astype(np.float32)})[0].flatten()

    # ── 6. Compare results ────────────────────────────────────────────────────
    max_diff = np.max(np.abs(sklearn_preds - onnx_preds))
    print(f"Sklearn predictions (first 5): {sklearn_preds[:5]}")
    print(f"ONNX    predictions (first 5): {onnx_preds[:5]}")
    print(f"Max absolute difference:       {max_diff:.6f}")

    if max_diff < 1e-4:
        print("✅ ONNX export validated successfully.")
    else:
        print("⚠️  Predictions differ — check the export pipeline.")


if __name__ == "__main__":
    main()