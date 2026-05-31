from fastapi import FastAPI

from src.api.routers.data import router as data_router
from src.api.routers.models import router as models_router
from src.api.routers.metrics import router as metrics_router
from src.api.routers.train import router as training_router
from src.api.routers.predict import router as inference_router

app = FastAPI(
    title="Chest X-ray Classification API",
    description="API for managing data pipelines, training models, and classifying chest X-ray images.",
    version="1.2.0",
)

app.include_router(data_router)
app.include_router(models_router)
app.include_router(metrics_router)
app.include_router(training_router)
app.include_router(inference_router)
