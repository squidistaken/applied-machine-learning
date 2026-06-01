import pytest

from src.api.schema import ModelMetrics
from fastapi.testclient import TestClient
from src.api.router import app


client = TestClient(app)


def test_and_retrieve_json(tmp_path):
    metrics = ModelMetrics(
        loss=0.2,
        macro_f1=0.9,
        precision=0.91,
        recall=0.89,
    )

    file_path = tmp_path / "metrics.json"

    metrics.export_to_json(file_path)

    loaded = ModelMetrics.retrieve_from_json(file_path)

    assert loaded == metrics


def test_data():
    response = client.get("/data")

    assert response.status_code == 200

def test_metrics():
    response = client.get("/metrics")

    assert response.status_code == 200


def test_models():
    response = client.get("/models")

    assert response.status_code == 200

def test_predict():
    response = client.get("/predict")

    assert response.status_code == 200