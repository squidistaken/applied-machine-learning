import requests
from src.constants import API_URL


def is_api_running():
    """Check if the FastAPI server is reachable."""
    try:
        requests.get(f"{API_URL}/models", timeout=2)
        return True
    except requests.exceptions.ConnectionError:
        return False


def download_data(force: bool):
    return requests.post(
        f"{API_URL}/data/download", json={"force_download": force}
    )


def preprocess_data(pipeline: str = "all", lgb_size: int = 64):
    return requests.post(
        f"{API_URL}/data/preprocess",
        json={"pipeline": pipeline, "lgb_size": lgb_size},
    )


def get_data_metadata(data_type="processed", split="train", limit=50):
    return requests.get(
        f"{API_URL}/data",
        params={"data_type": data_type, "split": split, "limit": limit},
    )


def get_image(data_type: str, split: str, index: int):
    return requests.get(f"{API_URL}/data/{data_type}/{split}/{index}")


def train_model(payload: dict):
    return requests.post(f"{API_URL}/train", json=payload)


def get_model_status(model_name: str):
    return requests.get(f"{API_URL}/models/{model_name}")


def predict_image(model_name: str, file_bytes: bytes, filename: str):
    files = {"file": (filename, file_bytes, "image/jpeg")}
    data = {"model_name": model_name}
    return requests.post(f"{API_URL}/predict", data=data, files=files)
