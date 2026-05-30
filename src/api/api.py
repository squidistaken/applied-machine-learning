from src.features.preprocess_lightgbm import extract_features, get_feature_names
import numpy as np
from src.constants import MODELS_DIR, RESULTS_DIR, DEVICE
from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    BackgroundTasks,
    Form,
)
from typing import Any, cast
from src.api.schema import (
    ModelType,
    PredictionClass,
    ImageResults,
    TrainRequest,
    ModelObject,
    TrainingStatus,
    ModelMetrics,
)
import re
import src.training.train
from PIL import Image
import io
from src.data.dataset_pytorch import ChestXRayDatasetPyTorch
from src.data.dataset_lightgbm import ChestXRayDatasetLightGBM
from src.models.cnn import CNN
from src.models.resnet import ResNet
from src.models.lgbm import LightGBM
import torch
import pandas as pd
from typing import Optional


app = FastAPI(
    title="Chest X-ray Classification API",
    description="API for training models and classifying chest X-ray images.",
    version="1.0.0",
)


def parse_evaluation_report(
    model_name: ModelType,
) -> tuple[Optional[ModelMetrics], dict[str, Any]]:
    metrics_file = RESULTS_DIR / f"{model_name.value}_metrics.txt"

    if not metrics_file.exists():
        return None, {}

    with open(metrics_file, "r") as f:
        content = f.read()

    parsed_metrics = {}

    for key in ["loss", "macro_f1", "precision", "recall"]:
        match = re.search(rf"{key}\s*:\s*([0-9.]+)", content)

        if match:
            parsed_metrics[key] = float(match.group(1))

    if "loss" not in parsed_metrics and len(parsed_metrics) >= 3:
        parsed_metrics["loss"] = None

    metrics_data = None

    if len(parsed_metrics) == 4:
        metrics_data = ModelMetrics(**parsed_metrics)

    hyperparameters = {}

    hyperparams_section = re.search(
        r"HYPERPARAMETERS:\n-+\n(.*?)\n\n", content, re.DOTALL
    )

    if hyperparams_section:
        for line in hyperparams_section.group(1).strip().split("\n"):
            if ":" in line:
                k, v = line.split(":", 1)
                v_str = v.strip()
                try:
                    v_val = float(v_str) if "." in v_str else int(v_str)
                except ValueError:
                    v_val = v_str
                hyperparameters[k.strip()] = v_val

    return metrics_data, hyperparameters


@app.get("/models")
async def get_models():
    """
    Retrieve available model names.
    """
    return {"models": [model.value for model in ModelType]}


@app.get("/models/{model_name}", response_model=ModelObject)
async def get_model(model: ModelType) -> ModelObject:
    """
    Retrieve basic information about a specific model.

    Input:

        model_name: The model, chosen from available model names found in the '/models' endpoint

    Response details:

        model_type: The model

        status: The model's training status

        model_path: Local path to trained model, returns None if training is not complete

        hyperparameters: The hyperparameters corresponding to the trained model

        metrics: Model evaluation metrics
            loss:       Evaluation loss of the model
            macro_f1:   Macro-averaged F1 score
            precision:  Macro-averaged precision
            recall:     Macro-averaged recall

    """
    model_filenames = {
        ModelType.CNN: "CNN.pt",
        ModelType.RESNET: "ResNet.pt",
        ModelType.LGBM: "LightGBM.txt",
    }

    model_file = MODELS_DIR / model_filenames[model]

    status = TrainingStatus.PENDING
    if model_file.exists():
        status = TrainingStatus.COMPLETED

    metrics_data, hyperparameters = parse_evaluation_report(model)

    return ModelObject(
        model_type=model,
        status=status,
        model_path=str(model_file) if model_file.exists() else None,
        hyperparameters=hyperparameters,
        metrics=metrics_data,
    )


@app.post("/train")
async def train_model(
    request: TrainRequest, background_tasks: BackgroundTasks
) -> dict[str, Any]:
    """
    Request model training to run in the background.

    Input:

        model_name(string): The model to train, chosen from available model names found in the '/models' endpoint

        epochs(int | None): The number of epochs for training loop; If unspecified, default values per model:
            cnn: 20
            resnet: 10
            lgbm: 100

        batch_size(int): The batch size for the PyTorch models (cnn and resnet)

        learning_rate(float | None): The learning rate for training; If unspecified, default values per model:
            cnn: 0.0001
            resnet: 0.0001
            lgbm: 0.1

        patience(int): The patience for early stopping

        num_leaves(int): The number of leaves for the LightGBM tree

        max_depth(int):  The maximum depth for the LightGBM tree (a value of -1 indicates no max)

        weight_decay(float): The weight decay (L2 penalty)

    Response Details:

        model_name: The model

        epochs: The number of epochs for training loop

        batch_size: The batch size for the PyTorch models (cnn and resnet)

        learning_rate: The learning rate for training
        
        status: The training status


    """
    epochs = request.epochs

    if epochs is None:
        if request.model_name == ModelType.CNN:
            epochs = 20
        elif request.model_name == ModelType.RESNET:
            epochs = 10
        elif request.model_name == ModelType.LGBM:
            epochs = 100

    lr = request.learning_rate

    if lr is None:
        if request.model_name == ModelType.CNN:
            lr = 0.0001
        elif request.model_name == ModelType.RESNET:
            lr = 0.0001
        elif request.model_name == ModelType.LGBM:
            lr = 0.1

    background_tasks.add_task(
        func=src.training.train.train_model,
        model_name=request.model_name.value,
        epochs=cast(int, epochs),
        lr=cast(float, lr),
        batch_size=request.batch_size,
        patience=request.patience,
        num_leaves=request.num_leaves,
        max_depth=request.max_depth,
        weight_decay=request.weight_decay,
        device=DEVICE,
    )

    return {
        "message": f"Training request for {request.model_name.value} received and started in the background.",
        "model_name": request.model_name.value,
        "epochs": epochs,
        "batch_size": request.batch_size,
        "learning_rate": lr,
        "status": TrainingStatus.TRAINING,
    }


@app.post("/predict")
async def predict_image(
    model_name: ModelType = Form(ModelType.CNN),
    file: UploadFile = File(...),
) -> ImageResults:
    """
    Classify a chest X-ray image.

    Input: 
    
        model_name: The model, chosen from available model names found in the '/models' endpoint
    
        file: The xray image the model should clasify. 
            Accepted image types: png, jpg, jpeg, pmg 
            
    Response Details:

        filename: The uploaded file
        
        model_used: The model used for the prediction
        
        predicted class: The predicted class for the uploaded image
        
        probabilities: The model's predicted likelihood that the uploaded image belongs to a given class
    """
    if not file.filename:
        raise HTTPException(
            status_code=400, detail="File must have a filename."
        )

    allowed_extensions = ["png", "jpg", "jpeg", "pgm"]
    extension = file.filename.split(".")[-1].lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Please upload upload one of: {allowed_extensions}",
        )

    contents = await file.read()

    try:
        img = Image.open(io.BytesIO(contents)).convert("L")
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Invalid or corrupt image file: {e}"
        )

    if model_name in [ModelType.CNN, ModelType.RESNET]:
        try:
            dataset = ChestXRayDatasetPyTorch(split="test", augment=False)
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to load dataset classes: {e}"
            )

        model = (
            CNN(dataset=dataset)
            if model_name == ModelType.CNN
            else ResNet(dataset=dataset)
        )

        try:
            model.load()
        except FileNotFoundError:
            raise HTTPException(
                status_code=404,
                detail=f"Trained weights for {model_name.value} not found. Train the model first.",
            )

        model.to(DEVICE)
        model.eval()

        transform = ChestXRayDatasetPyTorch.compose_transforms(augment=False)
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    elif model_name == ModelType.LGBM:
        try:
            dataset = ChestXRayDatasetLightGBM(split="test", augmented=False)
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to load dataset classes: {e}"
            )

        model = LightGBM(dataset=dataset)
        try:
            model.load()
        except FileNotFoundError:
            raise HTTPException(
                status_code=404,
                detail=f"Trained weights for {model_name.value} not found. Train the model first.",
            )

        if model.model is None:
            raise HTTPException(
                status_code=500, detail="LightGBM model failed to initialise."
            )

        target_size = (64, 64)
        feats = extract_features(img, target_size=target_size)
        feats_df = pd.DataFrame([feats], columns=get_feature_names(target_size))
        probs = np.asarray(model.model.predict(feats_df)[0])

    pred_idx = int(np.argmax(probs))

    enum_classes = list(PredictionClass)
    predicted_class = enum_classes[pred_idx]

    prob_dict = {
        PredictionClass.BACTERIA: probs[0],
        PredictionClass.NORMAL: probs[1],
        PredictionClass.VIRUS: probs[2],
    }

    return ImageResults(
        filename=file.filename,
        model_used=model_name,
        predicted_class=predicted_class,
        probabilities=prob_dict,
    )


@app.get("/metrics")
async def get_metrics(model_name: ModelType = ModelType.CNN) -> ModelMetrics:
    """
    Retrieve model evaluation metrics.

        loss:       Evaluation loss of the model
        
        macro_f1:   Macro-averaged F1 score
        
        precision:  Macro-averaged precision
        
        recall:     Macro-averaged recall

    """
    metrics_data, _ = parse_evaluation_report(model_name)
    if metrics_data is None:
        raise HTTPException(
            status_code=404,
            detail=f"Metrics for {model_name.value} not found. Train the model first.",
        )

    return metrics_data
