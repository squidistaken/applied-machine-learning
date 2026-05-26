from pydantic import BaseModel, Field
from typing import Dict, Optional, Any, Union, TypeVar, Type
from enum import Enum
from pathlib import Path
from abc import ABC

T = TypeVar("T", bound="BaseSchema")


class BaseSchema(BaseModel, ABC):
    """Base schema class."""

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


class TrainRequest(BaseSchema):
    """Schema class for training request parameters."""

    model_name: ModelType
    epochs: Optional[int] = Field(None, description="Number of training epochs")
    batch_size: int = Field(32, description="Batch size (PyTorch only)")
    learning_rate: Optional[float] = Field(None, description="Learning rate")
    patience: int = Field(3, description="Early stopping patience")
    num_leaves: int = Field(31, description="Number of leaves (LightGBM only)")
    max_depth: int = Field(-1, description="Max depth (LightGBM only)")
    weight_decay: float = Field(0.0, description="Weight decay (L2 penalty)")


class ModelMetrics(BaseSchema):
    """Schema class for model evaluation metrics."""

    loss: Optional[float] = Field(None, description="Evaluation loss")
    macro_f1: float = Field(
        ..., ge=0.0, le=1.0, description="Macro-averaged F1 score"
    )
    precision: float = Field(
        ..., ge=0.0, le=1.0, description="Macro-averaged Precision"
    )
    recall: float = Field(
        ..., ge=0.0, le=1.0, description="Macro-averaged Recall"
    )


class ModelObject(BaseSchema):
    """Schema class for a model's configuration and performance."""

    model_type: ModelType = Field(..., description="Model type")
    status: TrainingStatus = Field(
        default=TrainingStatus.COMPLETED,
        description="Current training status of the model",
    )
    model_path: Optional[str] = Field(
        None,
        description="File path or URI where the trained model weights are saved",
    )
    hyperparameters: Dict[str, Any] = Field(
        default_factory=dict, description="Hyperparameters used during training"
    )
    metrics: Optional[ModelMetrics] = Field(
        None,
        description="Performance metrics evaluated on the validation/test set",
    )


class ImageResults(BaseSchema):
    """Schema for the results returned after classifying an uploaded image."""

    filename: str = Field(..., description="Filename")
    model_used: ModelType = Field(
        ..., description="Model used for the classification"
    )
    predicted_class: PredictionClass = Field(
        ...,
        description="Final predicted class with the highest probability",
    )
    probabilities: Dict[PredictionClass, float] = Field(
        ..., description="Confidence scores for each class."
    )
