import requests
from src.constants import API_URL


def is_api_running() -> bool:
    """Check if the FastAPI server is reachable.

    Args:
        bool: Whether the API is reachable.
    """
    try:
        requests.get(f"{API_URL}/models", timeout=2)
        return True
    except requests.exceptions.ConnectionError:
        return False


def download_data(force: bool) -> requests.Response:
    """Download the dataset from Kaggle.

    Args:
        force (bool): Whether to force download.

    Returns:
        requests.Response: A response object from the API.
    """
    return requests.post(
        f"{API_URL}/data/download", json={"force_download": force}
    )


def preprocess_data(
    pipeline: str = "all", lgb_size: int = 64
) -> requests.Response:
    """Preprocess the dataset.

    Args:
        pipeline (str, optional): The preprocessing pipeline to use. Defaults to "all".
        lgb_size (int, optional): The target size for LightGBM preprocessing. Defaults to 64.

    Returns:
        requests.Response: A response object from the API.
    """
    return requests.post(
        f"{API_URL}/data/preprocess",
        json={"pipeline": pipeline, "lgb_size": lgb_size},
    )


def get_data_status(job: str) -> requests.Response:
    """Poll the status of a data job.

    Args:
        job (str): The job ID.

    Returns:
        requests.Response: A response object from the API.
    """
    return requests.get(f"{API_URL}/data/status/{job}", timeout=5)


def get_data_metadata(
    data_type="processed", split="train", page=1, limit=50
) -> requests.Response:
    """Get data metadata.

    Args:
        data_type (str, optional): The type of data to get. Defaults to "processed".
        split (str, optional): The split to get. Defaults to "train".
        page (int, optional): The page number. Defaults to 1.
        limit (int, optional): The number of items per page. Defaults to 50.

    Returns:
        requests.Response: A response object from the API.
    """
    return requests.get(
        f"{API_URL}/data",
        params={
            "data_type": data_type,
            "split": split,
            "page": page,
            "limit": limit,
        },
    )


def get_image(data_type: str, split: str, index: int) -> requests.Response:
    """Get an image from the dataset.


    Args:
        data_type (str): The data type.
        split (str): The split.
        index (int): The index.

    Returns:
        requests.Response: A response object from the API.
    """
    return requests.get(f"{API_URL}/data/{data_type}/{split}/{index}")


def preview_preprocessing(
    file_bytes: bytes, filename: str
) -> requests.Response:
    """Upload a raw image and get back the preprocessed PNG.

    Args:
        file_bytes (bytes): The image bytes.
        filename (str): The image filename.

    Returns:
        requests.Response: A response object from the API.
    """
    files = {"file": (filename, file_bytes, "application/octet-stream")}
    return requests.post(f"{API_URL}/data/preview", files=files)


def get_models() -> requests.Response:
    """Get a list of available models.

    Returns:
        requests.Response: A response object from the API.
    """
    return requests.get(f"{API_URL}/models", timeout=5)


def train_model(payload: dict) -> requests.Response:
    """Train a model.

    Args:
        payload (dict): The training payload.

    Returns:
        requests.Response: A response object from the API.
    """
    return requests.post(f"{API_URL}/train", json=payload)


def get_training_status(model_name: str) -> requests.Response:
    """Get the status of a training job.

    Args:
        model_name (str): The model name.

    Returns:
        requests.Response: A response object from the API.
    """
    return requests.get(f"{API_URL}/train/status/{model_name}", timeout=5)


def get_model_status(model_name: str) -> requests.Response:
    """Get the status of a model.

    Args:
        model_name (str): The model name.

    Returns:
        requests.Response: A response object from the API.
    """
    return requests.get(f"{API_URL}/models/{model_name}")


def get_metrics(model_name: str) -> requests.Response:
    """Get metrics.

    Args:
        model_name (str): The model name.

    Returns:
        requests.Response: A response object from the API.
    """
    return requests.get(f"{API_URL}/metrics", params={"model_name": model_name})


def list_plots(model_name: str) -> requests.Response:
    """List plots.

    Args:
        model_name (str): The model name.

    Returns:
        requests.Response: A response object from the API.
    """
    return requests.get(f"{API_URL}/metrics/plots/{model_name}", timeout=5)


def get_plot(model_name: str, plot_name: str) -> requests.Response:
    """Get a plot.

    Args:
        model_name (str): The model name.
        plot_name (str): The plot name.

    Returns:
        requests.Response: A response object from the API.
    """
    return requests.get(f"{API_URL}/metrics/plots/{model_name}/{plot_name}")


def predict_image(
    model_name: str, file_bytes: bytes, filename: str
) -> requests.Response:
    """Predict an image.

    Args:
        model_name (str): The model name.
        file_bytes (bytes): The image bytes.
        filename (str): The image filename.

    Returns:
        requests.Response: A response object from the API.
    """
    files = {"file": (filename, file_bytes, "image/jpeg")}
    data = {"model_name": model_name}
    return requests.post(f"{API_URL}/predict", data=data, files=files)
