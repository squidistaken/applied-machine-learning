from pathlib import Path
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from src.api.schema import ModelMetrics, ModelType
from src.api.router import app

client = TestClient(app)


def test_export_and_retrieve_json(tmp_path: Path) -> None:
    """Test exporting and retrieving ModelMetrics to and from JSON.

    Args:
        tmp_path (Path): The temporary directory path provided by pytest.
    """
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


def test_data_download() -> None:
    """Test the data list endpoint for returning correct status code."""
    response = client.get(
        "/data",
        params={"data_type": "raw", "split": "train", "page": 1, "limit": 20},
    )

    assert response.status_code == 200
    assert "items" in response.json()


def test_models_list() -> None:
    """Test the models list endpoint for returning correct status code."""
    response = client.get("/models")

    assert response.status_code == 200
    assert "models" in response.json()


def test_models_get_specific() -> None:
    """Test retrieving specific model information for returning correct status code."""
    response = client.get(f"/models/{ModelType.CNN.value}")

    assert response.status_code == 200
    assert response.json()["model_type"] == ModelType.CNN.value


def test_predict_invalid_file() -> None:
    """Test the predict endpoint with an invalid file extension for returning correct status code."""
    # Create a dummy text file instead of an image
    files = {"file": ("document.txt", b"dummy text content", "text/plain")}
    data = {"model_name": ModelType.CNN.value}

    response = client.post("/predict", files=files, data=data)

    assert response.status_code == 400
    assert "Invalid file type" in response.json()["detail"]


@patch("src.api.routers.train.src.training.train.train_model")
def test_train_model_dispatch(mock_train_model: MagicMock) -> None:
    """Test the training endpoint successfully dispatches a background task.

    Args:
        mock_train_model (MagicMock): The mocked method train_model.
    """
    payload = {"model_name": ModelType.CNN.value, "epochs": 2, "batch_size": 16}

    response = client.post("/train", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == ModelType.CNN.value
    assert data["status"] == "training"
    assert "background" in data["message"]

    mock_train_model.assert_called_once()


@patch("src.api.routers.metrics.parse_evaluation_report")
def test_get_metrics_not_found(mock_parse: MagicMock) -> None:
    """Test the metrics endpoint when a model has not been trained yet.

    Args:
        mock_parse (MagicMock): The mocked method parse_evaluation_report.
    """
    mock_parse.return_value = (None, {})

    response = client.get(f"/metrics?model_name={ModelType.RESNET.value}")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


@patch("src.api.routers.data.download_task")
def test_data_download_dispatch(mock_download_task: MagicMock) -> None:
    """Test the data download endpoint dispatches the Kaggle download task.

    Args:
        mock_download_task (MagicMock): The mocked method download_task.
    """
    payload = {
        "force_download": True,
        "kaggle_username": "test_user",
        "kaggle_key": "test_key",
    }

    response = client.post("/data/download", json=payload)

    assert response.status_code == 200
    assert "background" in response.json()["message"]
    mock_download_task.assert_called_once()
