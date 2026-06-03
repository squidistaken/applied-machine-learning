from typing import cast
from fastapi import APIRouter, BackgroundTasks
from src.constants import DEVICE
import src.training.train

from src.api.schema import (
    ModelType,
    TrainRequest,
    TrainingStatus,
    TrainResponse,
)

router = APIRouter(prefix="/train", tags=["Training"])


@router.post(
    "",
    response_model=TrainResponse,
    summary="Initiate Model Training",
    description="Dispatch a background task to train a machine learning model.",
    response_description="A TrainResponse confirming the task parameters and status.",
    responses={
        200: {"description": "Training task initiated successfully."},
        422: {
            "description": "Validation Error. Invalid parameters in the request payload."
        },
    },
)
async def train_model(
    request: TrainRequest, background_tasks: BackgroundTasks
) -> TrainResponse:
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
        enable_uq=request.enable_uq,
    )

    return TrainResponse(
        message=f"Training request for {request.model_name.value} received and started in the background (UQ metrics: {request.enable_uq}).",
        model_name=request.model_name,
        epochs=cast(int, epochs),
        batch_size=request.batch_size,
        learning_rate=cast(float, lr),
        status=TrainingStatus.TRAINING,
    )
