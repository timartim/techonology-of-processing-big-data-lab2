import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from redis.asyncio import from_url

from src.api.routes import router
from src.api.repositories.prediction_repository import PredictionRepository
from src.api.services.prediction_service import PredictionService
from src.models.CatVDogModel import CatVDogModel


@asynccontextmanager
async def lifespan(app: FastAPI):
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    model_version = os.getenv("MODEL_VERSION", "1.0.0")
    device = os.getenv("MODEL_DEVICE", "cpu")
    classifier_key = os.getenv("CLASSIFIER_KEY", "LOG_REG")

    redis = from_url(redis_url, decode_responses=True)
    await redis.ping()

    model_service = CatVDogModel(config_path="config.ini", show_log=True)
    model_service.set_device(device)
    model_service.load_classifier(classifier_key)

    repository = PredictionRepository(redis)

    prediction_service = PredictionService(
        ml_service=model_service,
        repository=repository,
        model_version=model_version,
    )

    app.state.redis = redis
    app.state.prediction_service = prediction_service

    yield

    await redis.aclose()


app = FastAPI(
    title="Dog vs Cat API",
    lifespan=lifespan,
)

app.include_router(router)