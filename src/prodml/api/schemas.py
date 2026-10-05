from pydantic import BaseModel, ConfigDict, Field

class PredictionRequest(BaseModel):
    trip_distance: float = Field(gt=0, lt=200, description="Distance of the trip in miles")
    PU_DO: str = Field(min_length=3, description="Pickup and dropoff location")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "PU_DO": "75_76",
                "trip_distance": 5.2,
            }
        }
    )

class PredictionResponse(BaseModel):
    prediction: float
    model_version: str
    correlation_id: str
    latency_ms: float


class BatchPredictionRequest(BaseModel):
    predictions: list[PredictionRequest]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "predictions": [
                    {"PU_DO": "75_76", "trip_distance": 5.2},
                    {"PU_DO": "75_76", "trip_distance": 3.1},
                ]
            }
        }
    )

class BatchPredictionResponse(BaseModel):
    predictions: list[float]
    model_version: str
    correlation_id: str
    latency_ms: float

class HealthCheckResponse(BaseModel):
    status: str
    model_loaded: bool


class MetadataResponse(BaseModel):
    model_version: str
    training_date: str
    feature_names: list[str]
    feature_count: int
    framework: str
    artifact_hash: str
