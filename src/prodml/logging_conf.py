import logging
from pythonjsonlogger.json import JsonFormatter
import uuid
from contextvars import ContextVar
from fastapi import FastAPI, Request


# 1. Initialize a ContextVar to store the ID safely across async tasks
correlation_id_ctx: ContextVar[str] = ContextVar("correlation_id", default="")

app = FastAPI()

# 2. Define the middleware
@app.middleware("http")
async def add_correlation_id(request: Request, call_next):
    # Check if client sent an existing correlation ID, otherwise generate one
    corr_id = request.headers.get("X-Correlation-ID") or str(uuid.uuid4())
    
    # Store it in contextvars for logging and request state for route access
    token = correlation_id_ctx.set(corr_id)
    request.state.correlation_id = corr_id
    
    # Process the request
    response = await call_next(request)
    
    # Return the correlation ID back to the client in response headers
    response.headers["X-Correlation-ID"] = corr_id
    
    # Reset context var after request finishes
    correlation_id_ctx.reset(token)
    
    return response

class CorrelationIdFilter(logging.Filter):
    def filter(self, record):
        record.correlation_id = correlation_id_ctx.get()
        print(f"Correlation ID in log record: {record.correlation_id}")  # Debugging line
        return True


def setup_logging(log_level=logging.INFO):
    """
    Sets up logging configuration with JSON formatting and correlation ID support.
    """
    logger = logging.getLogger()
    logger.setLevel(log_level)

    handler = logging.StreamHandler()
    handler.addFilter(CorrelationIdFilter())

    handler.setFormatter(JsonFormatter(["asctime", "levelname", "name", "message", "correlation_id"],
                                       datefmt="%Y-%m-%d %H:%M:%S",
                                       rename_fields={"asctime": "timestamp", "levelname": "level", "name": "logger"}))
    logger.addHandler(handler)


