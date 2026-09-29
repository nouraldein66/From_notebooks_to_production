from pathlib import Path
from pydantic_settings import BaseSettings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
class Settings(BaseSettings):
    data_path: Path = PROJECT_ROOT / "Dataset" / "green_tripdata_2026-01.parquet"
    model_path: Path = PROJECT_ROOT / "models" / "trained_model.pkl"
    onnx_path: Path = PROJECT_ROOT / "models" / "trained_model.onnx"
    validation_size: float = 0.2
    random_state: int = 42

    min_duration: float = 1.0
    max_duration: float = 60.0

    min_trip_distance: float = 0.0
    max_trip_distance: float = 50.0

settings = Settings()