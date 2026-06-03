import io
import numpy as np
import pandas as pd
from PIL import Image
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from src.constants import DEVICE
from src.features.preprocess_lightgbm import extract_features, get_feature_names
from src.data.dataset_pytorch import ChestXRayDatasetPyTorch
from src.data.dataset_lightgbm import ChestXRayDatasetLightGBM
from src.models.cnn import CNN
from src.models.resnet import ResNet
from src.models.lgbm import LightGBM
from src.api.schema import ModelType, PredictionClass, ImageResults
from src.utils.uq_utils import (
    get_mc_dropout_uncertainty,
    calculate_predictive_entropy,
)

router = APIRouter(prefix="/predict", tags=["Inference"])


@router.post(
    "",
    response_model=ImageResults,
    summary="Classify a Chest X-Ray",
    description="Upload a single raw chest X-ray image for classification.",
    response_description="An ImageResults object mapping the file to a predicted class and its probability distributions.",
    responses={
        200: {
            "description": "Image successfully processed, classified, and quantified for uncertainty."
        },
        400: {
            "description": "Bad Request. The file is not a supported image format."
        },
        404: {
            "description": "Not Found. The requested model has not been trained yet."
        },
        500: {
            "description": "Internal Server Error. Fails to process the image or initialize datasets."
        },
    },
)
async def predict_image(
    model_name: ModelType = Form(
        ModelType.CNN, description="Model architecture to use for predictions."
    ),
    file: UploadFile = File(
        ..., description="Chest X-Ray image file (png, jpg, jpeg, pgm)."
    ),
) -> ImageResults:
    if not file.filename:
        raise HTTPException(
            status_code=400, detail="File must have a valid filename."
        )

    allowed_extensions = ["png", "jpg", "jpeg", "pgm"]
    extension = file.filename.split(".")[-1].lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Please upload one of: {allowed_extensions}",
        )

    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("L")
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Invalid or corrupt image file: {e}"
        )

    uncertainty_score = 0.0
    epistemic_var = 0.0
    is_uncertain = False

    # Forward Pass via PyTorch Models.
    if model_name in [ModelType.CNN, ModelType.RESNET]:
        try:
            dataset = ChestXRayDatasetPyTorch(split="test", augment=False)
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to load dataset framework: {e}"
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
                detail=f"Trained weights for {model_name.value} not found. Please train the model first.",
            )

        model.to(DEVICE)

        transform = ChestXRayDatasetPyTorch.compose_transforms(augment=False)
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)

        mean_probs_tensor, variance_tensor, entropy_tensor = (
            get_mc_dropout_uncertainty(model=model, x=img_tensor, num_passes=15)
        )

        probs = mean_probs_tensor.cpu().numpy()[0]
        variance = variance_tensor.cpu().numpy()[0]

        epistemic_var = float(np.mean(variance))
        uncertainty_score = float(entropy_tensor.cpu().numpy()[0])

    elif model_name == ModelType.LGBM:
        try:
            dataset = ChestXRayDatasetLightGBM(split="test", augmented=False)
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to load dataset framework: {e}"
            )

        model = LightGBM(dataset=dataset)
        try:
            model.load()
        except FileNotFoundError:
            raise HTTPException(
                status_code=404,
                detail=f"Trained weights for {model_name.value} not found. Please train the model first.",
            )

        if model.model is None:
            raise HTTPException(
                status_code=500,
                detail="LightGBM model failed to initialise internally.",
            )

        target_size = (64, 64)
        feats = extract_features(img, target_size=target_size)
        feats_df = pd.DataFrame([feats], columns=get_feature_names(target_size))

        probs = np.asarray(model.model.predict(feats_df))[0]

        uncertainty_score = float(
            calculate_predictive_entropy(np.expand_dims(probs, axis=0))[0]
        )

        # Ensemble variance not applicable in standard LightGBM boosting
        # sequence.
        epistemic_var = 0.0

    # Threshold 0.75 maps to cases where model predictions approach flat
    # distributions.
    if uncertainty_score >= 0.75:
        is_uncertain = True

    pred_idx = int(np.argmax(probs))
    enum_classes = list(PredictionClass)
    predicted_class = enum_classes[pred_idx]

    prob_dict = {
        PredictionClass.BACTERIA: float(probs[0]),
        PredictionClass.NORMAL: float(probs[1]),
        PredictionClass.VIRUS: float(probs[2]),
    }

    return ImageResults(
        filename=file.filename,
        model_used=model_name,
        predicted_class=predicted_class,
        probabilities=prob_dict,
        uncertainty=uncertainty_score,
        epistemic_variance=epistemic_var,
        is_uncertain=is_uncertain,
    )
