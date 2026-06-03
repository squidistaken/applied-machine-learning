import json
import os
from typing import Any, Optional
from src.constants import RESULTS_DIR, DATA_DIR
from src.data.download_data import DataDownloader
from src.features.preprocess_pytorch import (
    preprocess_data as preprocess_pytorch_data,
)
from src.features.preprocess_lightgbm import (
    preprocess_data as preprocess_lightgbm_data,
)
from src.api.schema import (
    ModelType,
    ModelMetrics,
    DataType,
    SplitType,
    PredictionClass,
    DataPipelineType,
)


def parse_evaluation_report(
    model_name: ModelType,
) -> tuple[Optional[ModelMetrics], dict[str, Any]]:
    """
    Parse a saved model's evaluation report from the results directory.

    Args:
        model_name (ModelType): The architecture name of the model to parse.

    Returns:
        tuple[Optional[ModelMetrics], dict[str, Any]]: A tuple containing the parsed metrics
            (if available and complete) and a dictionary of the hyperparameters used.
    """
    json_metrics_file = RESULTS_DIR / f"{model_name.value}_metrics.json"

    if not json_metrics_file.exists():
        return None, {}

    try:
        with open(json_metrics_file, "r") as f:
            data = json.load(f)

        hyperparameters = data.get("hyperparameters", {})
        metrics_dict = data.get("test_metrics") or data.get(
            "validation_metrics"
        )

        metrics_data = None
        if metrics_dict:
            filtered_metrics: dict[str, Any] = {}
            for key in [
                "loss",
                "macro_f1",
                "precision",
                "recall",
                "ece",
                "predictive_entropy",
            ]:
                if key in metrics_dict:
                    filtered_metrics[key] = metrics_dict[key]

            if "loss" not in filtered_metrics and len(filtered_metrics) >= 3:
                filtered_metrics["loss"] = None

            if len(filtered_metrics) >= 3:
                metrics_data = ModelMetrics(**filtered_metrics)

        return metrics_data, hyperparameters
    except Exception:
        # Worst case: The data is corrupted.
        return None, {}


def get_data_files(data_type: DataType, split: SplitType) -> list[dict]:
    """
    Index all images for a specific data type and split.

    Args:
        data_type (DataType): The type of data to index (e.g., 'raw' or 'processed').
        split (SplitType): The dataset split to index (e.g., 'train' or 'test').

    Returns:
        list[dict]: A list of dictionaries containing metadata for each file.
    """
    base_dir = DATA_DIR / data_type.value / split.value
    if not base_dir.exists():
        return []

    classes = [
        PredictionClass.NORMAL.value,
        PredictionClass.BACTERIA.value,
        PredictionClass.VIRUS.value,
    ]
    files = []

    # Sorting ensures deterministic indexing.
    for cls in classes:
        cls_dir = base_dir / cls
        if cls_dir.is_dir():
            for file_path in sorted(cls_dir.iterdir()):
                if file_path.is_file() and not file_path.name.startswith("."):
                    files.append(
                        {
                            "filename": file_path.name,
                            "label": PredictionClass(cls),
                            "split": split,
                            "data_type": data_type,
                            "path": file_path,
                        }
                    )
    return files


def download_task(
    force_download: bool, username: Optional[str], key: Optional[str]
) -> None:
    """Download the dataset from Kaggle.

    Args:
        force_download (bool): Whether to force download.
        username (Optional[str]): The Kaggle username.
        key (Optional[str]): The Kaggle API key.
    """
    if username and key:
        os.environ["KAGGLE_USERNAME"] = username
        os.environ["KAGGLE_KEY"] = key
    downloader = DataDownloader(raw_data_path=DATA_DIR / "raw")
    downloader.run(force_download=force_download)


def preprocess_task(pipeline: DataPipelineType, lgb_size: int) -> None:
    """Preprocess the dataset.

    Args:
        pipeline (DataPipelineType): The preprocessing pipeline to use.
        lgb_size (int): The target size for LightGBM preprocessing.
    """
    if pipeline in [DataPipelineType.ALL, DataPipelineType.PYTORCH]:
        preprocess_pytorch_data()
    if pipeline in [DataPipelineType.ALL, DataPipelineType.LIGHTGBM]:
        preprocess_lightgbm_data(target_size=(lgb_size, lgb_size))
