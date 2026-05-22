from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any

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
            "/metrics",
        ],
    }


# TODO get to retrieve model
# TODO post to request to train a model

# TODO post to classify an image: returns result with metrics
# TODO get model metrics (confusion matrix and model history)




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
            "resnet",
            "pretrained",
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
        "resnet": "ResNet-based deep learning model.",
        "pretrained": "Pretrained transfer learning model.",
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
    batch_size: Optional[int] = 32
    learning_rate: Optional[float] = 0.001


@app.post("/train")
async def train_model(request: TrainRequest):
    """
    Request model training.
    call the training code from src/training/train.py.
    """

    return {
        "message": "Training request received",
        "model_name": request.model_name,
        "epochs": request.epochs,
        "batch_size": request.batch_size,
        "learning_rate": request.learning_rate,
        "status": "",
    }



@app.post("/predict")
async def predict_image(
    model_name: str = "lightgbm",
    file: UploadFile = File(...),
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

    return {
        "message": "Image received successfully",
        "filename": filename,
        "model_name": model_name,
        "prediction": "not implemented yet",
        "class_probabilities": {
            "BACTERIA": None,
            "NORMAL": None,
            "VIRUS": None,
        },
    }


@app.get("/metrics")
async def get_metrics(model_name: str = "lightgbm"):
    """
    Retrieve model evaluation metrics.
    """

    return {
        "model_name": model_name,
        "metrics": {
            "accuracy": None,
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