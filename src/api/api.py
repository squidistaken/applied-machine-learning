from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel #NOTE this is not our local base model class, will this give issues?
from typing import Optional, Dict, Any
from src.api.schema import ModelType, ModelObject, ModelMetrics


app = FastAPI(
    title="Chest X-ray Classification API",
    description="API for training models and classifying chest X-ray images.",
    version="0.1.0",
)

@app.get("/")
async def root():
    """
    To check if the API is running
    """
    return {
        "message": "Chest X-ray ML API is running",
        "available_endpoints": [
            "/health",
            "/models",
            "/train",
            "/predict",
            "/metrics"
        ],
    }

@app.get("/health")
async def health_check():
    """
    Health check endpoint to test whether the server is alive.
    """
    return {"status": "ok"}


@app.get("/models")
async def get_available_models():
    """
    Retrieve available model names.
    """
    return {
        "models": [
            "lightgbm",
            "cnn",
            "resnet"
        ]
    }


@app.get("/models/{model_name}")
async def get_model_info(model_name: str):
    """
    Retrieve basic information about a specific model.
    """

    available_models = {
        "lightgbm": "LightGBM model trained on extracted image features.",
        "cnn": "Custom convolutional neural network for chest X-ray classification.",
        "resnet": "ResNet-based deep learning model."
    }

    if model_name not in available_models:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found."
        )

    return {
        "model_name": model_name,
        "description": available_models[model_name],
    }



class TrainRequest(BaseModel): 
    model_name: str
    epochs: Optional[int] = 10
    learning_rate: Optional[float] = 0.001


@app.post("/train/")
async def train_model(request: TrainRequest):
    """
    Request model training.
    call the training code from src/training/train.py.
    """

    return {
        "message": "Training request received",
        "model_name": request.model_name,
        "epochs": request.epochs,
        "learning_rate": request.learning_rate,
        "status": "" #TODO implement status updates using Trainingstatus schema
    }



@app.post("/predict/")
async def predict_image(
    model_name: str = "cnn",
    file: UploadFile = File(...),#FIXME I don't think this is a very intuitive way to do this 
):
    """
    Classify a chest X-ray image.
    This should:
    1. Read the uploaded image.
    2. Preprocess it.
    3. Load the selected model.
    4. Return the predicted class and probabilities.
    """
    allowed_extensions = ["png", "jpg", "jpeg"]

    filename = file.filename 
    extension = filename.split(".")[-1].lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload a PNG, JPG, or JPEG image.",
        )

    # TODO implement separate processing based on the model
    return {
        "message": "Image received successfully",
        "filename": filename,
        "model_name": model_name,
        "prediction": "not implemented yet", #TODO actually implement this I suppose?
        "class_probabilities": {
            "BACTERIA": None,
            "NORMAL": None,
            "VIRUS": None,
        },
    }


@app.get("/metrics")
async def get_metrics(model_name: str = "cnn"):
    """
    Retrieve model evaluation metrics.
    """

    return {
        "model_name": model_name, #TODO implement/fix this?
        "metrics": {
            "precision": None,
            "recall": None,
            "macro_f1": None,
        },
        "confusion_matrix": {
            "labels": ["BACTERIA", "NORMAL", "VIRUS"],
            "matrix": None,
        },
        "status": "metrics",
    }