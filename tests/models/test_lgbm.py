import numpy as np
import pytest

from src.models.lgbm import LightGBM
from src.data.dataset_lightgbm import ChestXRayDatasetLightGBM


@pytest.fixture
def model():
    return LightGBM(ChestXRayDatasetLightGBM(split="train"))


def sample_data():
    rng = np.random.default_rng(42)

    X = rng.random((100, 20))
    y = rng.integers(0, 2, 100)

    return X, y

def test_lgmb_initialization(model):
    assert len(model.params) > 0
    assert model.params["num_class"] == 3


def test_lgbm_backward_pass(model):
    X, y = sample_data()

    model.backward_pass(x_train=X, y_train=y, num_boost_round=5)

    assert model.model is not None


def test_lgbm_forward_pass(model):
    X, y = sample_data()

    model.backward_pass(x_train=X, y_train=y, num_boost_round=5)

    predictions = model.forward_pass(X)

    assert predictions.shape == (len(X),)
    assert np.all(predictions >= 0)
    assert np.all(predictions < 3)


def test_lgbm_eval(model):
    X, y = sample_data()

    model.backward_pass(
        x_train=X,
        y_train=y,
        num_boost_round=5,
    )

    metrics = model.evaluate(X, y)

    assert "macro_f1" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
