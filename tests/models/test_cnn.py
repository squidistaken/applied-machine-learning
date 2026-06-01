import pytest
import torch
from pathlib import Path
from unittest.mock import MagicMock
from torch.utils.data import DataLoader, TensorDataset
from src.models.cnn import CNN
from src.data.dataset_pytorch import ChestXRayDatasetPyTorch
from typing import cast


@pytest.fixture
def mock_dataset() -> MagicMock:
    """Set up a mock PyTorch dataset for testing.

    Returns:
        MagicMock: A mocked ChestXRayDatasetPyTorch instance.
    """
    dataset = MagicMock(spec=ChestXRayDatasetPyTorch)
    dataset.classes = ["BACTERIA", "NORMAL", "VIRUS"]
    dataset.__len__.return_value = 16
    return dataset


@pytest.fixture
def model(mock_dataset: MagicMock) -> CNN:
    """Set up a CNN model instance for testing.

    Args:
        mock_dataset (MagicMock): The mocked PyTorch dataset.

    Returns:
        CNN: The initialized CNN model.
    """
    return CNN(mock_dataset)


def dataloader() -> DataLoader:
    """Create a dummy dataloader for testing the evaluation loop.

    Returns:
        DataLoader: A PyTorch dataloader yielding random images and labels.
    """
    images = torch.randn(16, 1, 128, 128)
    labels = torch.randint(0, 3, (16,))

    dataset = TensorDataset(images, labels)

    return DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
    )


def test_cnn_initialisation(model: CNN) -> None:
    """Test the initialisation of the CNN model.

    Args:
        model (CNN): The initialised CNN model.
    """
    assert model.num_classes == 3
    assert model.loss_function is not None
    assert model.optimizer is not None


def test_cnn_backward_pass(model: CNN) -> None:
    """Test that the backward pass executes successfully.

    Args:
        model (CNN): The initialised CNN model.
    """
    loader = cast(ChestXRayDatasetPyTorch, dataloader())
    model.backward_pass(x_train=loader, epochs=1)
    has_gradients = any(p.grad is not None for p in model.parameters())
    assert has_gradients is True


def test_cnn_forward_pass(model: CNN) -> None:
    """Test the forward pass of the CNN model.

    Args:
        model (CNN): The initialised CNN model.
    """
    x = torch.randn(8, 1, 128, 128)

    output = model(x)

    assert output.shape == (8, 3)

    x = torch.randn(4, 1, 128, 128)
    predictions = model.forward_pass(x)

    assert torch.all(predictions >= 0)
    assert torch.all(predictions < model.num_classes)


def test_cnn_eval(model: CNN) -> None:
    """Test the evaluation loop of the CNN model.

    Args:
        model (CNN): The initialised CNN model.
    """
    X = cast(ChestXRayDatasetPyTorch, dataloader())
    metrics = model.evaluate(X)

    assert isinstance(metrics, dict)

    assert "macro_f1" in metrics
    assert "precision" in metrics
    assert "recall" in metrics


def test_cnn_save_and_load_weights(model: CNN, tmp_path: Path) -> None:
    """Test saving and loading the model's state dictionary.

    Args:
        model (CNN): The initialised CNN model.
        tmp_path (Path): The Pytest fixture for temporary directories.
    """
    weights_file = tmp_path / "test_cnn_weights.pt"

    model._save_weights(weights_file)

    assert weights_file.exists()

    model._load_weights(weights_file)
