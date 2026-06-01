import pytest

from src.api.schema import ModelMetrics
from fastapi.testclient import TestClient
from src.api.router import app
from src.api.schema import TrainRequest, ModelType


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


def test_data_download():
    response = client.get("/data", headers={"data_type": "raw", "split": "train", "page": "0", "limit": "20"})

    assert response.status_code == 200



def test_models():
    response = client.get("/models", headers={"model_name": ModelType.CNN})

    assert response.status_code == 200