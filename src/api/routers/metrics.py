from fastapi import APIRouter, Query, HTTPException

from src.api.schema import ModelType, ModelMetrics
from src.utils.api_utils import parse_evaluation_report

router = APIRouter(prefix="/metrics", tags=["Models"])


@router.get(
    "",
    response_model=ModelMetrics,
    summary="Get Model Evaluation Metrics",
    description="Get the performance metrics for a specified trained model.",
    response_description="A ModelMetrics object containing loss, precision, recall, and F1-score.",
    responses={
        200: {"description": "Metrics retrieved successfully."},
        404: {
            "description": "Model not found or not yet trained.",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Metrics for cnn not found. Please train the model first."
                    }
                }
            },
        },
    },
)
async def get_metrics(
    model_name: ModelType = Query(
        ModelType.CNN, description="The model architecture to query."
    ),
) -> ModelMetrics:
    metrics_data, _ = parse_evaluation_report(model_name)
    if metrics_data is None:
        raise HTTPException(
            status_code=404,
            detail=f"Metrics for {model_name.value} not found. Please train the model first.",
        )
    return metrics_data
