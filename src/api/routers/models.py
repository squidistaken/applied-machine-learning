from fastapi import APIRouter, Path as APIPath

from src.constants import MODELS_DIR
from src.api.schema import ModelType, ModelObject, TrainingStatus
from src.utils.api_utils import parse_evaluation_report

router = APIRouter(prefix="/models", tags=["Models"])


@router.get(
    "",
    response_model=dict[str, list[str]],
    summary="Get Available Models",
    description="Get a list of all model architectures currently supported by the API.",
    response_description="Dictionary containing available models.",
    responses={
        200: {
            "description": "List of available models successfully retrieved.",
            "content": {
                "application/json": {
                    "example": {"models": ["cnn", "resnet", "lgbm"]}
                }
            },
        }
    },
)
async def get_models() -> dict[str, list[str]]:
    return {"models": [model.value for model in ModelType]}


@router.get(
    "/{model_name}",
    response_model=ModelObject,
    summary="Get Model Status and Information",
    description="Get the current training status, hyperparameter configurations, and "
    "the latest evaluation metrics for a specific model architecture.",
    response_description="A ModelObject containing the configuration, status, and metrics.",
    responses={
        200: {"description": "Model information retrieved successfully."},
        422: {"description": "Validation Error. Invalid model name provided."},
    },
)
async def get_model(
    model: ModelType = APIPath(
        ..., description="The architecture name.", alias="model_name"
    ),
) -> ModelObject:
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
