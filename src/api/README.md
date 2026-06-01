# Chest X-ray Classification API

This folder contains the FastAPI backend for the chest X-ray classification project.

The API provides endpoints for managing the data pipeline, retrieving model information, training models, classifying chest X-ray images, and retrieving evaluation metrics. It is designed to connect the machine learning models with the rest of the project, such as a dashboard or Streamlit app.

## Main API file

The main FastAPI application is defined in:

```text
src/api/router.py
```

The application includes separate routers for:

```text
src/api/routers/data.py
src/api/routers/models.py
src/api/routers/metrics.py
src/api/routers/train.py
src/api/routers/predict.py
```

The API schemas are defined in:

```text
src/api/schema.py
```

## Running the API

From the root directory of the project, run:

```bash
uvicorn src.api.router:app --reload
```

After starting the server, the API should be available at:

```text
http://127.0.0.1:8000
```

The automatic FastAPI documentation can be opened at:

```text
http://127.0.0.1:8000/docs
```

This documentation page can be used to test the available endpoints directly in the browser.

## API purpose

The goal of the API is to provide a backend interface for the chest X-ray classification system. It allows users or frontend applications to:

- download the dataset,
- preprocess the data,
- list available models,
- retrieve model status and metrics,
- start model training,
- upload an image for prediction,
- receive predicted classes and probability scores.

The API currently supports both PyTorch-based models and a LightGBM-based model.

## Supported models

The API currently supports the following model types:

```text
cnn
resnet
lgbm
```

These correspond to the model architectures used in the project:

- `cnn`: custom convolutional neural network,
- `resnet`: ResNet-based model,
- `lgbm`: LightGBM model using extracted image features.

## Prediction classes

The chest X-ray classification labels are:

```text
BACTERIA
NORMAL
VIRUS
```

These labels represent bacterial pneumonia, normal chest X-rays, and viral pneumonia.

# Available endpoints

## Data management endpoints

### `POST /data/download`

Starts a background task to download the chest X-ray dataset from Kaggle.

This endpoint can optionally receive Kaggle credentials in the request body. If Kaggle credentials are already available through environment variables, they do not need to be provided in the request.

### Request body example

```json
{
  "force_download": false,
  "kaggle_username": "johndoe",
  "kaggle_key": "your_api_key_here"
}
```

### Response example

```json
{
  "message": "Dataset download initiated in the background."
}
```

### Notes

- `force_download` controls whether the dataset should be downloaded again even if it already exists locally.
- `kaggle_username` and `kaggle_key` are optional.
- The download is started as a background task, so the API responds immediately after dispatching the job.

---

### `POST /data/preprocess`

Starts a background task to preprocess the downloaded raw dataset.

The preprocessing pipeline can be selected depending on whether the data should be prepared for PyTorch models, LightGBM, or both.

### Request body example

```json
{
  "pipeline": "all",
  "lgb_size": 64
}
```

### Available pipeline options

```text
all
pytorch
lightgbm
```

### Response example

```json
{
  "message": "Preprocessing pipeline 'all' initiated in the background."
}
```

### Notes

- `pipeline` specifies which preprocessing pipeline should be executed.
- `lgb_size` controls the image size used for LightGBM feature extraction.
- The preprocessing job runs as a background task.

---

### `GET /data`

Returns metadata for the available data files.

This endpoint supports pagination and filtering by data type and dataset split.

### Query parameters

```text
data_type: raw or processed
split: train or test
page: page number, starting from 1
limit: number of items per page
```

### Example request

```text
/data?data_type=raw&split=train&page=1&limit=20
```

### Response example

```json
{
  "total_items": 5216,
  "page": 1,
  "limit": 20,
  "total_pages": 261,
  "items": [
    {
      "index": 0,
      "filename": "IM-0115-0001.jpeg",
      "label": "NORMAL",
      "split": "train",
      "data_type": "raw"
    }
  ]
}
```

### Notes

- This endpoint does not return the image itself.
- It returns metadata such as filename, label, split, data type, and index.
- The index can be used to retrieve the actual image through the file endpoint.

---

### `GET /data/{data_type}/{split}/{index}`

Retrieves an actual image file by its index.

### Example request

```text
/data/raw/train/0
```

### Path parameters

```text
data_type: raw or processed
split: train or test
index: index of the image
```

### Notes

- This endpoint returns the image file itself.
- If the index is out of bounds, the API returns a `404` error.
- The endpoint can be used to inspect or download individual dataset images.

---

## Model information endpoints

### `GET /models`

Returns the list of model architectures supported by the API.

### Response example

```json
{
  "models": ["cnn", "resnet", "lgbm"]
}
```

### Notes

This endpoint is useful for checking which models can be used for training, prediction, and metrics retrieval.

---

### `GET /models/{model_name}`

Returns information about a specific model, including its training status, saved model path, hyperparameters, and available evaluation metrics.

### Example request

```text
/models/cnn
```

### Path parameter

```text
model_name: cnn, resnet, or lgbm
```

### Response example

```json
{
  "model_type": "cnn",
  "status": "completed",
  "model_path": "/app/models/CNN.pt",
  "hyperparameters": {
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 20
  },
  "metrics": {
    "loss": 0.21,
    "macro_f1": 0.92,
    "precision": 0.93,
    "recall": 0.91
  }
}
```

### Possible model statuses

```text
pending
training
completed
failed
```

### Notes

- If the model file exists locally, the model status is returned as `completed`.
- If the model file does not exist, the model status may be returned as `pending`.
- Metrics and hyperparameters are retrieved from the available evaluation report if it exists.

---

## Training endpoint

### `POST /train`

Starts model training as a background task.

The endpoint receives training parameters and dispatches the training function in the background. If some parameters are not provided, model-specific default values are used.

### Request body example

```json
{
  "model_name": "cnn",
  "epochs": 20,
  "batch_size": 32,
  "learning_rate": 0.001,
  "patience": 3,
  "num_leaves": 31,
  "max_depth": -1,
  "weight_decay": 0.0001
}
```

### Request fields

```text
model_name: model architecture to train, required
epochs: number of training epochs, optional
batch_size: batch size, default is 32
learning_rate: learning rate, optional
patience: early stopping patience, default is 3
num_leaves: number of LightGBM tree leaves, LightGBM only
max_depth: maximum LightGBM tree depth, LightGBM only
weight_decay: L2 penalty for PyTorch models
```

### Response example

```json
{
  "message": "Training request for cnn received and started in the background.",
  "model_name": "cnn",
  "epochs": 20,
  "batch_size": 32,
  "learning_rate": 0.001,
  "status": "training"
}
```

### Default training values

If `epochs` is not provided, the API assigns default values depending on the model:

```text
cnn: 20 epochs
resnet: 10 epochs
lgbm: 100 epochs
```

If `learning_rate` is not provided, the API assigns default values depending on the model:

```text
cnn: 0.0001
resnet: 0.0001
lgbm: 0.1
```

### Notes

- Training is started as a background task.
- The API response confirms that training has started.
- The endpoint does not wait until training is finished before responding.
- The actual training function is called from the project training module.

---

## Prediction endpoint

### `POST /predict`

Uploads a single chest X-ray image and returns the predicted class and class probabilities.

### Input

This endpoint expects form-data input:

```text
model_name: cnn, resnet, or lgbm
file: uploaded chest X-ray image
```

### Accepted file formats

```text
png
jpg
jpeg
pgm
```

### Example response

```json
{
  "filename": "scan_001.jpeg",
  "model_used": "resnet",
  "predicted_class": "BACTERIA",
  "probabilities": {
    "BACTERIA": 0.85,
    "NORMAL": 0.05,
    "VIRUS": 0.10
  }
}
```

### How prediction works

For `cnn` and `resnet`:

- the API loads the corresponding trained PyTorch model,
- applies the PyTorch image preprocessing pipeline,
- performs a forward pass,
- applies softmax to obtain class probabilities,
- returns the predicted class with the highest probability.

For `lgbm`:

- the API loads the trained LightGBM model,
- extracts image features from the uploaded image,
- converts the features into the expected dataframe format,
- uses the LightGBM model to predict probabilities,
- returns the predicted class with the highest probability.

### Error cases

The endpoint may return an error if:

- the uploaded file has no filename,
- the file type is not supported,
- the image file is invalid or corrupted,
- the selected model has not been trained yet,
- the model weights cannot be found,
- the dataset framework cannot be loaded.

---

## Metrics endpoint

### `GET /metrics`

Returns evaluation metrics for a selected model.

### Query parameter

```text
model_name: cnn, resnet, or lgbm
```

### Example request

```text
/metrics?model_name=cnn
```

### Response example

```json
{
  "loss": 0.342,
  "macro_f1": 0.89,
  "precision": 0.91,
  "recall": 0.88
}
```

### Returned metrics

```text
loss: evaluation loss on the validation or test set
macro_f1: macro-averaged F1 score
precision: macro-averaged precision
recall: macro-averaged recall
```

### Notes

- Metrics are retrieved from the available evaluation report.
- If no metrics are available for the selected model, the API returns a `404` error.
- The error message asks the user to train the model first.

---

# Schemas

The API uses Pydantic schemas to define the structure of request and response data.

Important schemas include:

```text
DownloadRequest
PreprocessRequest
TrainRequest
TrainResponse
ModelObject
ModelMetrics
ImageResults
DataMetadata
PaginatedDataResponse
BackgroundJobResponse
```

These schemas validate input data and make the automatic FastAPI documentation clearer.

## Important schema descriptions

### `DownloadRequest`

Used by:

```text
POST /data/download
```

Defines the optional parameters for downloading the dataset from Kaggle.

Fields:

```text
force_download
kaggle_username
kaggle_key
```

---

### `PreprocessRequest`

Used by:

```text
POST /data/preprocess
```

Defines which preprocessing pipeline should be executed.

Fields:

```text
pipeline
lgb_size
```

---

### `TrainRequest`

Used by:

```text
POST /train
```

Defines the training settings sent to the API.

Fields:

```text
model_name
epochs
batch_size
learning_rate
patience
num_leaves
max_depth
weight_decay
```

---

### `TrainResponse`

Returned by:

```text
POST /train
```

Confirms that a training job has been started.

Fields:

```text
message
model_name
epochs
batch_size
learning_rate
status
```

---

### `ModelObject`

Returned by:

```text
GET /models/{model_name}
```

Contains model status and information.

Fields:

```text
model_type
status
model_path
hyperparameters
metrics
```

---

### `ModelMetrics`

Returned by:

```text
GET /metrics
```

Contains evaluation metrics for a trained model.

Fields:

```text
loss
macro_f1
precision
recall
```

---

### `ImageResults`

Returned by:

```text
POST /predict
```

Contains prediction results for an uploaded chest X-ray image.

Fields:

```text
filename
model_used
predicted_class
probabilities
```

---

### `DataMetadata`

Used inside:

```text
GET /data
```

Contains metadata for one data file.

Fields:

```text
index
filename
label
split
data_type
```

---

### `PaginatedDataResponse`

Returned by:

```text
GET /data
```

Contains a paginated list of data file metadata.

Fields:

```text
total_items
page
limit
total_pages
items
```

---

### `BackgroundJobResponse`

Returned by:

```text
POST /data/download
POST /data/preprocess
```

Confirms that a background task has been started.

Fields:

```text
message
```

---

# Example workflow

A typical workflow could be:

## 1. Start the API

```bash
uvicorn src.api.router:app --reload
```

## 2. Download the dataset

```text
POST /data/download
```

## 3. Preprocess the dataset

```text
POST /data/preprocess
```

## 4. Check available models

```text
GET /models
```

## 5. Train a model

```text
POST /train
```

## 6. Check model information

```text
GET /models/{model_name}
```

## 7. Retrieve model metrics

```text
GET /metrics?model_name=cnn
```

## 8. Upload an image for prediction

```text
POST /predict
```


