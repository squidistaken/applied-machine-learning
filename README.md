# AML: Pneumonia Classification via Chest X-Rays
Repository for the Applied Machine Learning course (WBAI065-05) at the University of Groningen.

## Development

We use [uv](https://docs.astral.sh/uv/) for project management.

1. Clone the project.
2. Synchronise the project.

```bash
uv sync
```

3. Create a copy of [`example.config.yaml`](example.config.yaml) and rename it to `config.yaml`. Update the configuration, if desired.

## Command Line Interface (CLI)

The project can be run via a CLI, for convenient usage and testing.

### Downloading Data

```bash
uv run -m src.data.download [--force]
```
 * `--force`: Forces a redownload of the data, in the event of missing or corrupted raw data. Defaults to `False`.
 
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
 * `--model`: The model architecture to train: `cnn`, `resnet`, `lgbm`. (Required)
 * `--epochs`: Number of training epochs. Defaults dynamically.
 * `--batch-size`: Batch size for PyTorch models. Defaults to 32.
 * `--lr`: Learning rate. Defaults dynamically.
 * `--patience`: Epochs to wait for improvement before early stopping. Defaults to 3.
 * `--num-leaves`: Number of leaves for LightGBM. Defaults to 31.
 * `--max-depth`: Maximum tree depth for LightGBM. Defaults to -1.
 * `--device`: Device for PyTorch models (`cuda`, `mps`, `cpu`). Defaults to auto-detection.

### Cross-Validation

```bash
uv run -m src.training.cv --model <model_name> [options]
```
 * `--model`: The model to cross-validate: `cnn`, `resnet`, `lgbm`. (Required)
 * `--splits`: Number of folds (k). Defaults to 5.
 * `--epochs`: Number of training epochs. Defaults dynamically.
 * `--lr`: Learning rate. Defaults dynamically.
 * `--device`: Device for PyTorch models (`cuda`, `mps`, `cpu`). Defaults to auto-detection.
 * `--grid-search`: Enable hyperparameter grid search cross-validation.

### Running Tests

```bash
uv run pytest tests
```
