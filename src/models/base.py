from typing import Union, Any, Optional
from src.data.dataset_lightgbm import ChestXRayDatasetLightGBM
from src.data.dataset_pytorch import ChestXRayDatasetPyTorch
from abc import ABC, abstractmethod
from src.constants import MODELS_DIR, LOGGER
from pathlib import Path


class BaseModel(ABC):
    """Abstract Base Class for the machine learning models of the project.

    It is intended for three models:
    1. Three-layer CNN (Baseline).
    2. LightGBM Classifier.
    3. Pretrained and Finetuned CNN (ResNet-18).
    """

    def __init__(
        self, dataset: Union[ChestXRayDatasetLightGBM, ChestXRayDatasetPyTorch]
    ) -> None:
        self.dataset = dataset
        self.classes = (
            dataset.classes
            if dataset is ChestXRayDatasetPyTorch
            else ["BACTERIA", "NORMAL", "VIRUS"]
        )

    @abstractmethod
    def forward_pass(self, x: Any) -> Any:
        """
        Predict the class of the input.

        Args:
            x (Any): The input data.

        Returns:
            Any: The predicted class labels or probabilities.
        """
        ...

    @abstractmethod
    def backward_pass(self, x_train: Any, y_train: Any, **kwargs) -> Any:
        """
        Train the model.

        Args:
            x_train (Any): The training data.
            y_train (Any): The training labels.
            **kwargs: Additional hyperparamters or configuration.

        Returns:
            Any: The trained model.
        """
        ...

    def evaluate(self, x_test: Any, y_test: Any) -> Any:
        """
        Test the performance of the model

        Args:
            x_test (Any): The testing data features.
            y_test (Any): The testing data true labels.

        Returns:
            Any: Metric(s) indicating the performance.
        """
        ...

    def save_model(self, filename: Optional[str] = None) -> None:
        """
        Save a model and its weights.

        Args:
            filename (str): The path to save the model to.
        """
        if filename is None:
            default_ext = (
                ".pt"
                if isinstance(self.dataset, ChestXRayDatasetPyTorch)
                else ".model"
            )
            ext = getattr(self, "FILE_EXTENSION", default_ext)
            filename = f"{self.__class__.__name__}{ext}"

        path = MODELS_DIR / filename

        path.parent.mkdir(parents=True, exist_ok=True)

        LOGGER.info(f"Saving model to {path}...")
        self._save_weights(path)
        LOGGER.info("Model saved successfully.")

    @abstractmethod
    def _save_weights(self, path: Path) -> None:
        """
        Save model weights.

        Args:
            path (Path): The path to save the model weights to.
        """
        ...

    def load(self, filename: Optional[str] = None) -> None:
        """
        Load a trained model.

        Args:
            filename (str): The path to load the model from.
        """
        if filename is None:
            default_ext = (
                ".pt"
                if isinstance(self.dataset, ChestXRayDatasetPyTorch)
                else ".model"
            )
            ext = getattr(self, "FILE_EXTENSION", default_ext)
            filename = f"{self.__class__.__name__}{ext}"

        path = MODELS_DIR / filename

        if not path.exists():
            LOGGER.error(f"Model file not found: {path}")

            raise FileNotFoundError(
                f"Cannot load model. File does not exist: {path}"
            )

        LOGGER.info(f"Loading model from {path}...")
        self._load_weights(path)
        LOGGER.info("Model loaded successfully.")

    @abstractmethod
    def _load_weights(self, path: Path) -> None:
        """
        Load model weights.

        Args:
            path (Path): The path to load the model weights from.
        """
        ...
