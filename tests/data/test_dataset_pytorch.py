import pytest
from unittest.mock import patch
from pathlib import Path
import tempfile
from PIL import Image
import torch
from torchvision import transforms
from typing import Iterator
from src.data.dataset_pytorch import ChestXRayDatasetPyTorch


@pytest.fixture
def dummy_pytorch_dataset() -> Iterator[Path]:
    """Set up a temporary directory and dummy data for PyTorch dataset tests.

    Yields:
        Iterator[Path]: An iterator containing a path to the dummy processed
                        directory.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        root_dir = Path(temp_dir)
        processed_dir = root_dir / "processed" / "train"

        (processed_dir / "NORMAL").mkdir(parents=True)
        (processed_dir / "BACTERIA").mkdir(parents=True)
        Image.new("L", (10, 10)).save(processed_dir / "NORMAL" / "img1.pgm")
        Image.new("L", (10, 10)).save(processed_dir / "BACTERIA" / "img2.pgm")
        Image.new("L", (10, 10)).save(processed_dir / "BACTERIA" / "img3.pgm")

        with patch("src.data.dataset_pytorch.DATA_DIR", root_dir):
            yield processed_dir


def test_init(dummy_pytorch_dataset: Path) -> None:
    """Test the initialisation and class mapping of the PyTorch dataset.

    Args:
        dummy_pytorch_dataset (Path): The dummy dummy_pytorch_dataset instance.
    """
    dataset = ChestXRayDatasetPyTorch(split="train")

    assert len(dataset) == 3
    assert dataset.classes == ["BACTERIA", "NORMAL"]
    assert dataset.class_to_idx == {"BACTERIA": 0, "NORMAL": 1}
    assert isinstance(dataset.transform, transforms.Compose)


def test_create_dataset(dummy_pytorch_dataset: Path) -> None:
    """Test the internal dataset creation logic.

    Args:
        dummy_pytorch_dataset (Path): The dummy dummy_pytorch_dataset instance.
    """
    processed_dir = dummy_pytorch_dataset
    dataset = ChestXRayDatasetPyTorch(split="train")
    samples = dataset._create_dataset()
    assert len(samples) == 3

    expected_samples = [
        (str(processed_dir / "BACTERIA" / "img2.pgm"), 0),
        (str(processed_dir / "BACTERIA" / "img3.pgm"), 0),
        (str(processed_dir / "NORMAL" / "img1.pgm"), 1),
    ]

    assert sorted(samples) == sorted(expected_samples)


def test_len(dummy_pytorch_dataset: Path) -> None:
    """Test the __len__ method of the dataset.

    Args:
        dummy_pytorch_dataset (Path): The dummy dummy_pytorch_dataset instance.
    """
    dataset = ChestXRayDatasetPyTorch(split="train")

    assert len(dataset) == 3


def test_getitem(dummy_pytorch_dataset: Path) -> None:
    """Test the __getitem__ method of the dataset.

    Args:
        dummy_pytorch_dataset (Path): The dummy dummy_pytorch_dataset instance.
    """
    dataset = ChestXRayDatasetPyTorch(split="train")

    image, label = dataset[0]
    assert isinstance(image, torch.Tensor)
    assert label == 0
    assert image.shape[0] == 1


def test_compose_transforms() -> None:
    """Test the transformation composition for both training and validation."""

    # Without augmentation.
    transform_no_aug = ChestXRayDatasetPyTorch.compose_transforms(augment=False)
    assert len(transform_no_aug.transforms) == 1
    assert isinstance(transform_no_aug.transforms[0], transforms.ToTensor)

    # With augmentation.
    transform_aug = ChestXRayDatasetPyTorch.compose_transforms(augment=True)
    assert len(transform_aug.transforms) > 1
    assert isinstance(transform_aug.transforms[-1], transforms.ToTensor)


def test_compute_sample_weights(dummy_pytorch_dataset: Path) -> None:
    """Test the computation of sample weights for class balancing.

    Args:
        dummy_pytorch_dataset (Path): The dummy dummy_pytorch_dataset instance.
    """
    dataset = ChestXRayDatasetPyTorch(split="train")
    weights = dataset.compute_sample_weights()

    # total=3, counts={'BACTERIA': 2, 'NORMAL': 1} -> idxs {0:2, 1:1}
    # weight_0 = 3/2 = 1.5, weight_1 = 3/1 = 3.0

    class_counts = {"BACTERIA": 2, "NORMAL": 1}
    total_samples = 3
    class_weights = {
        dataset.class_to_idx[cls]: total_samples / count
        for cls, count in class_counts.items()
    }

    # BACTERIA/img2, BACTERIA/img3, NORMAL/img1 -> Targets: 0, 0, 1
    expected_weights = [class_weights[0], class_weights[0], class_weights[1]]

    assert len(weights) == 3
    assert weights == expected_weights
