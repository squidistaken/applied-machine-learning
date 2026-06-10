# AML: Pneumonia Classification via Chest X-Rays
Repository for the Applied Machine Learning course (WBAI065-05) at the University of Groningen.

This project classifies chest X-rays as `NORMAL`, `BACTERIA`, or `VIRUS` to support pneumonia diagnosis. It trains and compares several models and ships them behind a FastAPI backend, an interactive Streamlit dashboard, and a CLI, all sharing the same data, preprocessing, training, and inference pipelines. Predictions come with an uncertainty estimate to flag unreliable results.

## Team

 * Anneke Catherine Naseef (S6490662)
 * Marcus Harald Olof Persson (S5343798)
 * Ignacio Jacob Uroz Rodríguez	(S5118913)
 * Julian Wilbert Sprietsma	(S5096219)

## Running via Docker

1. Build the image.

```bash
docker compose build
```

2. Run the image.

```bash
docker compose up
```

This exposes the following ports:
 * `8000`: FastAPI application.
 * `8501`: Streamlit dashboard.
 * `6006`: TensorBoard.

Once the stack is up, open the dashboard at [http://localhost:8501](http://localhost:8501).

## Development

We use [uv](https://docs.astral.sh/uv/) for project management.

1. Clone the project.
2. Synchronise the project.

```bash
uv sync
```

3. Create a copy of [`example.config.yaml`](example.config.yaml) and rename it to `config.yaml`. Update the configuration, if desired.

## Dashboard

An interactive [Streamlit](https://streamlit.io/) dashboard wraps the FastAPI backend with four pages:

 * **Introduction**: Project overview.
 * **Data & Preprocessing**: Download the dataset, run the preprocessing pipeline, and compare the raw and preprocessed X-rays with a draggable before/after slider.
 * **Model Training**: Configure a model, launch a run, watch validation metrics live, and review the saved metrics and evaluation plots.
 * **Showcase**: Upload an X-ray and classify it. Results update live as you switch models, complete with an uncertainty (reliability) verdict.

When running via Docker (`docker compose up`), the dashboard is served automatically at [http://localhost:8501](http://localhost:8501).

To run it locally for development:

1. Start the FastAPI backend — the dashboard talks to it exclusively:

```bash
uv run uvicorn src.api.router:app --port 8000
```

2. In a separate terminal, launch the dashboard:

```bash
uv run streamlit run main.py
```

The dashboard opens at [http://localhost:8501](http://localhost:8501) and expects the API at `API_URL` (default `http://127.0.0.1:8000`, configurable in `config.yaml`).

## API

A [FastAPI](https://fastapi.tiangolo.com/) backend exposes the data, training, and inference pipelines over HTTP. It is a Level 2 REST API: distinct resource URIs, proper use of HTTP verbs, and meaningful status codes. The dashboard talks to it exclusively, but it can also be used directly. It is grouped into the following routers:

 * `/data`: Download and preprocess the dataset.
 * `/models`: List available and trained models.
 * `/metrics`: Retrieve saved metrics and evaluation plots.
 * `/train`: Launch and monitor training runs.
 * `/predict`: Classify an uploaded X-ray, with an uncertainty estimate.

Interactive, auto-generated documentation is available at [http://localhost:8000/docs](http://localhost:8000/docs) once the backend is running.

## Command Line Interface (CLI)

The project can be run via a CLI, for convenient usage and testing.

### Downloading Data

#### Option 1: Download Script using Kaggle API

```bash
uv run -m src.data.download [--force]
```
 * `--force`: Forces a redownload of the data, in the event of missing or corrupted raw data. Defaults to `False`.

This requires a Kaggle API token to be set up on your device: https://www.kaggle.com/settings/api

#### Option 2: Manual Download

Dataset: https://www.kaggle.com/datasets/tolgadincer/labeled-chest-xray-images

1. Extract the archive and place it in a directory named "`DATA_DIR`/raw/" (ex. `data/raw/<the extracted folder>`).
2. Run the download script for automated reorganisation.

### Preprocessing and Feature Extraction

```bash
uv run -m src.features.preprocess_data [--pipeline] [--lgb-size]
```
 * `--pipeline`: Chooses which pipeline to run: `pytorch`, `lightgbm`, `all`. Running the `pytorch` pipeline is required in order to run the `lightgbm` pipeline. Defaults to `all`.
 * `--lgb-size`: Determines the edge size for downsampling in LightGBM feature extraction. Defaults to 64.

### Training a Model

```bash
uv run -m src.training.train --model <model_name> [options]
```
 * `--model`: The model architecture to train: `cnn`, `resnet`, `lgbm`.
 * `--epochs`: Number of training epochs. Defaults dynamically.
 * `--batch-size`: Batch size for PyTorch models. Defaults to 32.
 * `--lr`: Learning rate. Defaults dynamically.
 * `--patience`: Epochs to wait for improvement before early stopping. Defaults to 3.
 * `--num-leaves`: Number of leaves for LightGBM. Defaults to 31.
 * `--max-depth`: Maximum tree depth for LightGBM. Defaults to -1.
 * ``--weight-decay`: Weight decay for PyTorch models. Defaults to 0.0.
 * `--device`: Device for PyTorch models (`cuda`, `mps`, `cpu`). Defaults to auto-detection.

### Cross-Validation

```bash
uv run -m src.training.cv --model <model_name> [options]
```
 * `--model`: The model to cross-validate: `cnn`, `resnet`, `lgbm`.
 * `--splits`: Number of folds (k). Defaults to 5.
 * `--epochs`: Number of training epochs. Defaults dynamically.
 * `-batch-size`: Batch size for PyTorch models. Defaults to 32.
 * `--lr`: Learning rate. Defaults dynamically.
 * `--weight-decay`: Weight decay for PyTorch models. Defaults to 0.0.
 * `--device`: Device for PyTorch models (`cuda`, `mps`, `cpu`). Defaults to auto-detection.
 * `--grid-search`: Enable hyperparameter grid search cross-validation.

## Running Tests

```bash
uv run pytest tests
```
