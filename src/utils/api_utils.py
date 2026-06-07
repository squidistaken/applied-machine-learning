import json
import os
from typing import Any, Optional
from src.constants import RESULTS_DIR, DATA_DIR, LOGGER
from src.api import job_state
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
                "brier_score",
                "nll",
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
    job_state.start_job("download", "Contacting Kaggle...")
    # Kaggle's downloader gives no byte-level progress, so we report coarse,
    # indeterminate stages instead of a precise fraction.
    job_state.update_job(
        "download", progress=None, message="Downloading dataset from Kaggle..."
    )
    try:
        if username and key:
            os.environ["KAGGLE_USERNAME"] = username
            os.environ["KAGGLE_KEY"] = key
        downloader = DataDownloader(raw_data_path=DATA_DIR / "raw")
        downloader.run(force_download=force_download)
        job_state.complete_job("download", "Dataset downloaded and organised.")
    except Exception as e:
        LOGGER.error(f"Download task failed: {e}")
        job_state.fail_job("download", str(e))


def preprocess_task(pipeline: DataPipelineType, lgb_size: int) -> None:
    """Preprocess the dataset.

    Args:
        pipeline (DataPipelineType): The preprocessing pipeline to use.
        lgb_size (int): The target size for LightGBM preprocessing.
    """
    job_state.start_job("preprocess", "Starting preprocessing...")
    try:
        if pipeline in [DataPipelineType.ALL, DataPipelineType.PYTORCH]:

            def _pytorch_progress(done: int, total: int, message: str) -> None:
                # Reserve the second half of the bar for the LightGBM stage when
                # both pipelines run, otherwise use the full bar.
                scale = 0.5 if pipeline == DataPipelineType.ALL else 1.0
                job_state.update_job(
                    "preprocess",
                    progress=(done / total) * scale,
                    message=message,
                )

            preprocess_pytorch_data(progress_callback=_pytorch_progress)

        if pipeline in [DataPipelineType.ALL, DataPipelineType.LIGHTGBM]:
            base = 0.5 if pipeline == DataPipelineType.ALL else 0.0
            job_state.update_job(
                "preprocess",
                progress=base,
                message="Extracting LightGBM features (HOG + statistics)...",
            )
            preprocess_lightgbm_data(target_size=(lgb_size, lgb_size))

        job_state.complete_job("preprocess", "Preprocessing complete.")
    except Exception as e:
        LOGGER.error(f"Preprocess task failed: {e}")
        job_state.fail_job("preprocess", str(e))


def train_task(model_name: str, **kwargs: Any) -> None:
    """Run a training job while reporting live progress via ``job_state``.

    Args:
        model_name (str): The model architecture being trained (job id).
        **kwargs: Keyword arguments forwarded to ``train_model``.
    """
    # Imported lazily to avoid importing heavy training dependencies (and the
    # whole torch stack) unless a training job is actually dispatched.
    from src.training.train import train_model

    job_state.start_job(model_name, "Initialising training run...")

    def _on_epoch_end(epoch: int, total: int, metrics: dict) -> None:
        job_state.update_job(
            model_name,
            progress=epoch / total if total else None,
            message=f"Epoch {epoch}/{total}",
        )
        job_state.append_history(model_name, {"epoch": epoch, **metrics})

    try:
        train_model(model_name=model_name, on_epoch_end=_on_epoch_end, **kwargs)
        job_state.complete_job(
            model_name, "Training complete. Metrics and plots are ready."
        )
    except Exception as e:
        LOGGER.error(f"Training task for {model_name} failed: {e}")
        job_state.fail_job(model_name, str(e))
