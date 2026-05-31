import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.cnn import CNN
from src.data.dataset_pytorch import ChestXRayDatasetPyTorch


@pytest.fixture
def model():
    return CNN(ChestXRayDatasetPyTorch(split="train"))


@pytest.fixture
def dataloader():
    images = torch.randn(16, 1, 128, 128)
    labels = torch.randint(0, 3, (16,))

    dataset = TensorDataset(images, labels)

    return DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
    )


def test_cnn_initalization(model):
    assert model.num_classes == 3
    assert model.loss_function is not None
    assert model.optimizer is not None


def test_cnn_forward_pass(model):
    x = torch.randn(8, 1, 128, 128)

    output = model(x)

    assert output.shape == (8, 3)

    x = torch.randn(4, 1, 128, 128)
    predictions = model.forward_pass(x)

    assert torch.all(predictions >= 0)
    assert torch.all(predictions < model.num_classes)


def test_cnn_eval(model):
    metrics = model.evaluate(dataloader)

    assert isinstance(metrics, dict)

    assert "macro_f1" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
