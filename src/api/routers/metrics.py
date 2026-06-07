from fastapi import APIRouter, Query, HTTPException, Path as APIPath
from fastapi.responses import FileResponse

from src.constants import RESULTS_DIR
from src.api.schema import ModelType, ModelMetrics, PlotListResponse
from src.utils.api_utils import parse_evaluation_report

router = APIRouter(prefix="/metrics", tags=["Models"])

_MODEL_FILE_PREFIX = {
    ModelType.CNN: "CNN",
    ModelType.RESNET: "ResNet",
    ModelType.LGBM: "LightGBM",
}

_PLOT_SUFFIXES = {
    "training_history": "training_history",
    "confusion_matrix": "confusion_matrix_test",
    "reliability_diagram": "reliability_diagram_test",
    "selective_prediction": "selective_prediction_test",
}


def _plot_path(model: ModelType, plot: str):
    """Resolve the on-disk path for a model's evaluation plot."""
    prefix = _MODEL_FILE_PREFIX[model]
    suffix = _PLOT_SUFFIXES[plot]
    return RESULTS_DIR / f"{prefix}_{suffix}.png"


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


@router.get(
    "/plots/{model_name}",
    response_model=PlotListResponse,
    summary="List Available Evaluation Plots",
    description="List the evaluation plots that have been generated for a model "
    "(training history, confusion matrix, reliability diagram, selective "
    "prediction). Only plots that exist on disk are returned.",
    response_description="A PlotListResponse listing available plot identifiers.",
)
async def list_plots(
    model: ModelType = APIPath(
        ..., description="The model architecture.", alias="model_name"
    ),
) -> PlotListResponse:
    available = [
        plot for plot in _PLOT_SUFFIXES if _plot_path(model, plot).exists()
    ]
    return PlotListResponse(model_name=model, plots=available)


@router.get(
    "/plots/{model_name}/{plot_name}",
    response_class=FileResponse,
    summary="Download an Evaluation Plot",
    description="Retrieve a single evaluation plot image for a trained model.",
    response_description="The requested plot as a PNG image.",
    responses={
        200: {"content": {"image/png": {}}, "description": "The plot image."},
        404: {"description": "The plot has not been generated for this model."},
    },
)
async def get_plot(
    model: ModelType = APIPath(
        ..., description="The model architecture.", alias="model_name"
    ),
    plot_name: str = APIPath(
        ..., description="The plot identifier (see list endpoint)."
    ),
) -> FileResponse:
    if plot_name not in _PLOT_SUFFIXES:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown plot '{plot_name}'. Valid options: {list(_PLOT_SUFFIXES)}.",
        )

    path = _plot_path(model, plot_name)
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Plot '{plot_name}' for {model.value} not found. Please train the model first.",
        )
    return FileResponse(path=path, media_type="image/png")
