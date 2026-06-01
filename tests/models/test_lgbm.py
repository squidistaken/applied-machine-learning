import numpy as np
import pytest
from pathlib import Path
from typing import Tuple
from unittest.mock import MagicMock
from src.models.lgbm import LightGBM
from src.data.dataset_lightgbm import ChestXRayDatasetLightGBM


@pytest.fixture
def mock_dataset() -> MagicMock:
    """Set up a mock LightGBM dataset for testing.

    Returns:
        MagicMock: A mocked ChestXRayDatasetLightGBM instance.
    """
    dataset = MagicMock(spec=ChestXRayDatasetLightGBM)
    dataset.classes = ["BACTERIA", "NORMAL", "VIRUS"]
    return dataset


@pytest.fixture
def model(mock_dataset: MagicMock) -> LightGBM:
    """Set up a LightGBM model instance for testing.

    Args:
        mock_dataset (MagicMock): The mock LightGBM dataset.

    Returns:
        LightGBM: An initialised LightGBM model.
    """
    return LightGBM(mock_dataset)


def sample_data() -> Tuple[np.ndarray, np.ndarray]:
    """Create sample dummy data for training and evaluating LightGBM.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing random features
                                       and integer labels.
    """
    rng = np.random.default_rng(42)

    X = rng.random((100, 20))
    y = rng.integers(0, 3, 100)

    return X, y


def test_lgmb_initialization(model: LightGBM) -> None:
    """Test the initialisation of the LightGBM model.

    Args:
        model (LightGBM): The initialised LightGBM model.
    """
    assert len(model.params) > 0
    assert model.params["num_class"] == 3


def test_lgbm_untrained_forward_pass_raises_error(model: LightGBM) -> None:
    """Test that predicting with an untrained model raises a ValueError.

    Args:
        model (LightGBM): The initialised untrained LightGBM model.
    """
    X, _ = sample_data()

    with pytest.raises(ValueError, match="Model is not trained yet"):
        model.forward_pass(X)


def test_lgbm_backward_pass(model: LightGBM) -> None:
    """Test the backward pass of the LightGBM model.

    Args:
        model (LightGBM): The initialised LightGBM model.
    """
    X, y = sample_data()

    model.backward_pass(x_train=X, y_train=y, num_boost_round=5)

    assert model.model is not None


def test_lgbm_backward_pass_with_validation(model: LightGBM) -> None:
    """Test training with a validation dataset and metric tracking.

    Args:
        model (LightGBM): The initialised LightGBM model.
    """
    X, y = sample_data()
    X_train, X_val = X[:80], X[80:]
    y_train, y_val = y[:80], y[80:]

    evals_result = {}

    model.backward_pass(
        x_train=X_train,
        y_train=y_train,
        x_val=X_val,
        y_val=y_val,
        num_boost_round=5,
        evals_result=evals_result,
    )

    assert model.model is not None
    assert "val" in evals_result
    assert "macro_f1" in evals_result["val"]
    assert len(evals_result["val"]["macro_f1"]) > 0


def test_lgbm_forward_pass(model: LightGBM) -> None:
    """Test the forward pass of the LightGBM model.

    Args:
        model (LightGBM): The initialised LightGBM model.
    """
    X, y = sample_data()

    model.backward_pass(x_train=X, y_train=y, num_boost_round=5)

    predictions = model.forward_pass(X)

    assert predictions.shape == (len(X),)
    assert np.all(predictions >= 0)
    assert np.all(predictions < 3)


def test_lgbm_eval(model: LightGBM) -> None:
    """Test the evaluation loop of the LightGBM model.

    Args:
        model (LightGBM): The initialised LightGBM model.
    """
    X, y = sample_data()

    model.backward_pass(
        x_train=X,
        y_train=y,
        num_boost_round=5,
    )

    metrics = model.evaluate(x_test=X, y_test=y)

    assert isinstance(metrics, dict)
    assert "macro_f1" in metrics
    assert "precision" in metrics
    assert "recall" in metrics


def test_lgbm_save_and_load_weights(model: LightGBM, tmp_path: Path) -> None:
    """Test saving and loading the LightGBM booster object.

    Args:
        model (LightGBM): The initialised LightGBM model.
        tmp_path (Path): The Pytest fixture for temporary directories.
    """
    X, y = sample_data()
    model.backward_pass(x_train=X, y_train=y, num_boost_round=2)

    weights_file = tmp_path / "test_lgbm_weights.txt"

    model._save_weights(weights_file)
    assert weights_file.exists()

    new_mock = MagicMock(spec=ChestXRayDatasetLightGBM)
    new_mock.classes = ["BACTERIA", "NORMAL", "VIRUS"]

    new_model = LightGBM(new_mock)
    new_model._load_weights(weights_file)

    assert new_model.model is not None
