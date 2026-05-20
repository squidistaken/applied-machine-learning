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

### Running Tests

```bash
uv run pytest tests
```
