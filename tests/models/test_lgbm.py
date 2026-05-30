import numpy as np
import pandas as pd
import pytest

from src.models.lgbm import LightGBM
import lightgbm as lgb
from torch.utils.data import Dataset

class mock_dataset(Dataset):
    def __init__(self):
        self.classes = ["normal", "bacterial", "viral"]


@pytest.fixture
def sample_data():
    rng = np.random.default_rng(42)

    X = rng.random((100, 20))
    y = rng.integers(0, 2, 100)

    return X, y

@pytest.fixture
def model():
    dataset = mock_dataset()
    return LightGBM(dataset)

def test_lgmb_initialization(model):
    assert len(model.params) > 0
    assert model.params["num_classes"] == 3

def test_lgbm_backward_pass(model):
    X, y = sample_data()

    model.backward_pass(x_train=X, y_train=y, num_boost_round=5)

    assert model.model is not None

def test_lgbm_forward_pass(model, sample_data):
    X, y = sample_data()

    model.backward_pass(x_train=X, y_train=y, num_boost_round=5)

    predictions = model.forward_pass(X)

    assert predictions.shape == (len(X),)
    assert np.all(predictions >= 0)
    assert np.all(predictions < 3)


def test_lgbm_eval(model, sample_data):
    X, y = sample_data

    model.backward_pass(
        x_train=X,
        y_train=y,
        num_boost_round=5,
    )

    metrics = model.evaluate(X, y)

    assert "macro_f1" in metrics
    assert "precision" in metrics
    assert "recall" in metrics


