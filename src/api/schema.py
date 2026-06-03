from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Optional, Any, Union, TypeVar, Type, List
from enum import Enum
from pathlib import Path
from abc import ABC

T = TypeVar("T", bound="BaseSchema")


class BaseSchema(BaseModel, ABC):
    """Base schema class with utility methods for JSON import/export."""

    def export_to_json(self, file_path: Union[Path, str]) -> None:
        """Export the schema instance to a JSON file.

        Args:
            file_path (Union[Path, str]): The path to the JSON file.
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=4))

    @classmethod
    def retrieve_from_json(cls: Type[T], file_path: Union[Path, str]) -> T:
        """Retrieve a schema instance from a JSON file.

        Args:
            file_path (Union[Path, str]): The path to the JSON file.

        Returns:
            T: A schema instance.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            return cls.model_validate_json(f.read())


# region Enums


class ModelType(str, Enum):
    """Enumeration class for available models."""

    CNN = "cnn"
    RESNET = "resnet"
    LGBM = "lgbm"


class PredictionClass(str, Enum):
    """Enumeration class for X-ray classification labels."""

    BACTERIA = "BACTERIA"
    NORMAL = "NORMAL"
    VIRUS = "VIRUS"


class TrainingStatus(str, Enum):
    """Enumeration class for model training statuses for async tracking."""

    PENDING = "pending"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"


class DataPipelineType(str, Enum):
    """Enumeration class for preprocessing pipeline types."""

    ALL = "all"
    PYTORCH = "pytorch"
    LIGHTGBM = "lightgbm"


class DataType(str, Enum):
    """Enumeration class for data directory types."""

    RAW = "raw"
    PROCESSED = "processed"


class SplitType(str, Enum):
    """Enumeration class for dataset splits."""

    TRAIN = "train"
    TEST = "test"


# endregion

# region Schemas


class DownloadRequest(BaseSchema):
    """Schema class for data download request parameters."""

    force_download: bool = Field(
        default=False,
        description="Force download the dataset from Kaggle even if it already exists locally.",
    )
    kaggle_username: Optional[str] = Field(
        default=None,
        description="Kaggle API username. If provided, overrides existing environment variables.",
    )
    kaggle_key: Optional[str] = Field(
        default=None,
        description="Kaggle API key. If provided, overrides existing environment variables.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "force_download": False,
                "kaggle_username": "johndoe",
                "kaggle_key": "your_api_key_here",
            }
        }
    )


class PreprocessRequest(BaseSchema):
    """Schema class for data preprocessing request parameters."""

    pipeline: DataPipelineType = Field(
        default=DataPipelineType.ALL,
        description="Which preprocessing pipeline to execute (PyTorch, LightGBM, or both).",
    )
    lgb_size: int = Field(
        default=64,
        description="Edge size for downsampling in LightGBM feature extraction.",
    )

    model_config = ConfigDict(
        json_schema_extra={"example": {"pipeline": "all", "lgb_size": 64}}
    )


class TrainRequest(BaseSchema):
    """Schema class for training request parameters."""

    model_name: ModelType = Field(
        ..., description="The architecture of the model to train."
    )
    epochs: Optional[int] = Field(
        None,
        description="Number of training epochs. Falls back to model-specific defaults if omitted.",
    )
    batch_size: int = Field(32, description="Batch size (PyTorch models only).")
    learning_rate: Optional[float] = Field(
        None,
        description="Learning rate. Falls back to model-specific defaults if omitted.",
    )
    patience: int = Field(
        3,
        description="Number of epochs to wait for validation loss improvement before early stopping.",
    )
    num_leaves: int = Field(
        31, description="Number of tree leaves (LightGBM only)."
    )
    max_depth: int = Field(
        -1, description="Maximum tree depth (-1 for no limit, LightGBM only)."
    )
    weight_decay: float = Field(
        0.0, description="Weight decay/L2 penalty (PyTorch models only)."
    )
    enable_uq: bool = Field(
        True,
        description="Whether to evaluate and compute Uncertainty Quantification (UQ) metrics during training/validation splits.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model_name": "cnn",
                "epochs": 20,
                "batch_size": 32,
                "learning_rate": 0.001,
                "patience": 3,
                "num_leaves": 31,
                "max_depth": -1,
                "weight_decay": 1e-4,
                "enable_uq": True,
            }
        }
    )


class ModelMetrics(BaseSchema):
    """Schema class for model evaluation metrics."""

    loss: Optional[float] = Field(
        None,
        description="Evaluation loss calculated on the validation/test set.",
    )
    macro_f1: float = Field(
        ..., ge=0.0, le=1.0, description="Macro-averaged F1 score."
    )
    precision: float = Field(
        ..., ge=0.0, le=1.0, description="Macro-averaged Precision."
    )
    recall: float = Field(
        ..., ge=0.0, le=1.0, description="Macro-averaged Recall."
    )
    ece: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Expected Calibration Error (ECE)."
    )
    predictive_entropy: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Average Predictive Entropy.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "loss": 0.342,
                "macro_f1": 0.89,
                "precision": 0.91,
                "recall": 0.88,
                "ece": 0.054,
                "predictive_entropy": 0.32,
            }
        }
    )


class ModelObject(BaseSchema):
    """Schema class for a model's configuration and performance status."""

    model_type: ModelType = Field(
        ..., description="The type of model architecture."
    )
    status: TrainingStatus = Field(
        default=TrainingStatus.COMPLETED,
        description="Current training status of the model.",
    )
    model_path: Optional[str] = Field(
        None,
        description="File path or URI where the trained model weights are saved.",
    )
    hyperparameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Hyperparameters used during the training phase.",
    )
    metrics: Optional[ModelMetrics] = Field(
        None,
        description="Performance metrics evaluated on the test dataset.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model_type": "cnn",
                "status": "completed",
                "model_path": "/app/models/CNN.pt",
                "hyperparameters": {
                    "learning_rate": 0.001,
                    "batch_size": 32,
                    "epochs": 20,
                    "enable_uq": True,
                },
                "metrics": {
                    "loss": 0.21,
                    "macro_f1": 0.92,
                    "precision": 0.93,
                    "recall": 0.91,
                    "ece": 0.048,
                    "predictive_entropy": 0.28,
                },
            }
        }
    )


class ImageResults(BaseSchema):
    """Schema class for the results returned after classifying an uploaded image."""

    filename: str = Field(
        ..., description="The original filename of the uploaded image."
    )
    model_used: ModelType = Field(
        ..., description="The model architecture used for the classification."
    )
    predicted_class: PredictionClass = Field(
        ...,
        description="Final predicted class with the highest probability.",
    )
    probabilities: Dict[PredictionClass, float] = Field(
        ...,
        description="Confidence scores (probabilities) for each possible class.",
    )
    uncertainty: Optional[float] = Field(
        None,
        description="The predictive entropy representing total/aleatoric uncertainty.",
    )
    epistemic_variance: Optional[float] = Field(
        None, description="The mean variance across MC Dropout passes."
    )
    is_uncertain: Optional[bool] = Field(
        None,
        description="Whether the predictive entropy exceeds a safe clinical threshold or not.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "filename": "scan_001.jpeg",
                "model_used": "resnet",
                "predicted_class": "PNEUMONIA",
                "probabilities": {
                    "NORMAL": 0.05,
                    "BACTERIA": 0.85,
                    "VIRUS": 0.10,
                },
                "uncertainty": 0.52,
                "epistemic_variance": 0.015,
                "is_uncertain": False,
            }
        }
    )


class DataMetadata(BaseSchema):
    """Schema class for a single data file's metadata."""

    index: int = Field(
        ...,
        description="The deterministic index of the image within its specific dataset split.",
    )
    filename: str = Field(..., description="The name of the image file.")
    label: PredictionClass = Field(
        ..., description="The verified class label of the image."
    )
    split: SplitType = Field(
        ..., description="The dataset split (e.g., train, test)."
    )
    data_type: DataType = Field(
        ..., description="The type of data (raw or processed)."
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "index": 42,
                "filename": "person101_virus_188.jpeg",
                "label": "VIRUS",
                "split": "train",
                "data_type": "raw",
            }
        }
    )


class PaginatedDataResponse(BaseSchema):
    """Schema class for paginated data retrieval."""

    total_items: int = Field(
        ...,
        description="Total number of items available in the requested split.",
    )
    page: int = Field(..., description="Current page number being viewed.")
    limit: int = Field(..., description="Number of items returned per page.")
    total_pages: int = Field(
        ..., description="Total number of pages available."
    )
    items: List[DataMetadata] = Field(
        ..., description="List of data item metadata for the current page."
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "total_items": 5216,
                "page": 1,
                "limit": 20,
                "total_pages": 261,
                "items": [
                    {
                        "index": 0,
                        "filename": "IM-0115-0001.jpeg",
                        "label": "NORMAL",
                        "split": "train",
                        "data_type": "raw",
                    }
                ],
            }
        }
    )


class BackgroundJobResponse(BaseSchema):
    """Schema for a generic background job acknowledgment."""

    message: str = Field(
        ...,
        description="Status message acknowledging the background job submission.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "message": "Dataset download initiated in the background."
            }
        }
    )


class TrainResponse(BaseSchema):
    """Schema class for a training job acknowledgment."""

    message: str = Field(
        ..., description="Status message indicating that training has started."
    )
    model_name: ModelType = Field(
        ..., description="The model architecture scheduled for training."
    )
    epochs: int = Field(
        ..., description="The assigned number of training epochs."
    )
    batch_size: int = Field(..., description="The assigned batch size.")
    learning_rate: float = Field(..., description="The assigned learning rate.")
    status: TrainingStatus = Field(
        ...,
        description="The current status of the training job (typically 'training').",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "message": "Training request for cnn received and started in the background.",
                "model_name": "cnn",
                "epochs": 20,
                "batch_size": 32,
                "learning_rate": 0.001,
                "status": "training",
            }
        }
    )


# endregion
