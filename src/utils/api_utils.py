import re
import os
from typing import Any, Optional, cast
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
        data = cast(dict[str, Any], parsed_metrics)
        metrics_data = ModelMetrics(**data)

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
    """Background task to download the dataset from Kaggle.

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
    """Background task to preprocess the dataset.

    Args:
        pipeline (DataPipelineType): The preprocessing pipeline to use.
        lgb_size (int): The target size for LightGBM preprocessing.
    """
    if pipeline in [DataPipelineType.ALL, DataPipelineType.PYTORCH]:
        preprocess_pytorch_data()
    if pipeline in [DataPipelineType.ALL, DataPipelineType.LIGHTGBM]:
        preprocess_lightgbm_data(target_size=(lgb_size, lgb_size))
