import hashlib
import logging
import time
import traceback
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from prodml.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthCheckResponse,
    MetadataResponse,
    PredictionRequest,
    PredictionResponse,
)
from prodml import settings, setup_logging, correlation_id_ctx, load_model, Predictor
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging(log_level=logging.INFO)

    logger.info("Loading model from %s", settings.model_path)
    model, vectorizer = load_model(settings.model_path)

    with open(settings.model_path, "rb") as f:
        artifact_hash = hashlib.sha256(f.read()).hexdigest()

    app.state.predictor = Predictor(model, vectorizer)
    app.state.model_metadata = {
        "model_version": "0.1.0",
        "training_date": datetime.fromtimestamp(
            settings.model_path.stat().st_mtime, tz=timezone.utc
        ).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "feature_names": vectorizer.feature_names_,
        "feature_count": len(vectorizer.feature_names_),
        "framework": "scikit-learn",
        "artifact_hash": artifact_hash,
    }
    logger.info("Model loaded successfully (%d features)", len(vectorizer.feature_names_))
    yield
    logger.info("Shutting down – releasing model resources")


app = FastAPI(title="ProdML Prediction API", version="0.1.0", lifespan=lifespan)


# ── Exception handlers ──────────────────────────────────────────────────────

@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    logger.warning("Validation error on %s: %s", request.url.path, exc.errors())
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Validation failed",
            "errors": exc.errors(),
        },
    )


@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception):
    logger.error("Unhandled error on %s:\n%s", request.url.path, traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# ── Correlation-ID middleware ────────────────────────────────────────────────

@app.middleware("http")
async def add_correlation_id(request: Request, call_next):
    import uuid

    corr_id = request.headers.get("X-Correlation-ID") or str(uuid.uuid4())
    token = correlation_id_ctx.set(corr_id)
    request.state.correlation_id = corr_id

    response = await call_next(request)
    response.headers["X-Correlation-ID"] = corr_id

    correlation_id_ctx.reset(token)
    return response


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthCheckResponse)
async def health():
    model_loaded = hasattr(app.state, "predictor") and app.state.predictor is not None
    status_code = 200 if model_loaded else 503
    return JSONResponse(
        status_code=status_code,
        content={"status": "healthy" if model_loaded else "model not loaded", "model_loaded": model_loaded},
    )


@app.get("/metadata", response_model=MetadataResponse)
async def metadata():
    return app.state.model_metadata


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: Request, body: PredictionRequest):
    start = time.perf_counter()
    predictor: Predictor = app.state.predictor

    features = {"PU_DO": body.PU_DO, "trip_distance": body.trip_distance}
    prediction = predictor.predict_single(features)

    latency_ms = (time.perf_counter() - start) * 1000
    return PredictionResponse(
        prediction=round(prediction, 4),
        model_version=app.state.model_metadata["model_version"],
        correlation_id=request.state.correlation_id,
        latency_ms=round(latency_ms, 2),
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: Request, body: BatchPredictionRequest):
    start = time.perf_counter()
    predictor: Predictor = app.state.predictor

    features_list = [
        {"PU_DO": item.PU_DO, "trip_distance": item.trip_distance}
        for item in body.predictions
    ]
    predictions = predictor.predict_batch(features_list)

    latency_ms = (time.perf_counter() - start) * 1000
    return BatchPredictionResponse(
        predictions=[round(p, 4) for p in predictions],
        model_version=app.state.model_metadata["model_version"],
        correlation_id=request.state.correlation_id,
        latency_ms=round(latency_ms, 2),
    )
