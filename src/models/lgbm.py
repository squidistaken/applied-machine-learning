import lightgbm as lgb
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from pathlib import Path
from typing import Optional, Dict, Union
import pandas as pd
from src.models.base import BaseModel
from src.data.dataset_lightgbm import ChestXRayDatasetLightGBM
from src.constants import LOGGER


class LightGBMModel(BaseModel):
    """
    LightGBM Class for Chest X-Ray Image Classification.
    """

    def __init__(
        self, dataset: ChestXRayDatasetLightGBM, params: Optional[Dict] = None
    ) -> None:
        super().__init__(dataset=dataset, file_extension=".txt")

        # NOTE: Default hyperparameters for LightGBM multiclass classification.
        # We will need to modify pending the CV results.
        self.params = params or {
            "objective": "multiclass",
            "num_class": len(self.classes),
            "metric": "multi_logloss",
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1,
            "verbose": -1,
            "random_state": 42,
        }
        self.model = None

    def forward_pass(self, x: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict the class of the input.

        Args:
            x (Union[np.ndarray, pd.DataFrame]): The input data.

        Returns:
            np.ndarray: A list of predicted class labels or probabilities.
        """
        if self.model is None:
            raise ValueError(
                "Model is not trained yet. Call backward_pass first."
            )

        y_pred_prob = np.asarray(self.model.predict(x))
        predictions = np.argmax(y_pred_prob, axis=1)

        return predictions

    def backward_pass(
        self,
        x_train: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_train: Optional[Union[np.ndarray, pd.Series]] = None,
        val_dataset: Optional[ChestXRayDatasetLightGBM] = None,
        num_boost_round: int = 100,
        **kwargs,
    ) -> None:
        """
        Train the model.

        Args:
            x_train (Optional[Union[np.ndarray, pd.DataFrame]]): The training
                                                                 data.
            y_train (Optional[Union[np.ndarray, pd.Series]]): The training
                                                              labels.
            val_dataset (Optional[ChestXRayDatasetLightGBM]): The validation
                                                              dataset.
            num_boost_round (int): The number of boosting rounds.
            **kwargs: The additional hyperparamters or configuration.
        """
        if x_train is not None and y_train is not None:
            lgb_train = lgb.Dataset(x_train, label=y_train)
        elif isinstance(self.dataset, ChestXRayDatasetLightGBM):
            lgb_train = self.dataset.get_lgb_dataset(free_raw_data=False)
        else:
            raise TypeError(
                "Dataset must be an instance of ChestXRayDatasetLightGBM."
            )

        valid_sets = [lgb_train]
        valid_names = ["train"]
        callbacks = []

        if val_dataset is not None:
            lgb_val = val_dataset.get_lgb_dataset(free_raw_data=False)

            valid_sets.append(lgb_val)
            valid_names.append("val")

            # TODO: Temporary early stopping metric, we need to standardise it
            # for all models.
            callbacks.append(lgb.early_stopping(stopping_rounds=10))
            callbacks.append(lgb.log_evaluation(period=10))

        LOGGER.info("Training LightGBM model...")

        self.model = lgb.train(
            self.params,
            train_set=lgb_train,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )

        LOGGER.info("LightGBM training completed.")

    def evaluate(
        self,
        x_test: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_test: Optional[Union[np.ndarray, pd.Series]] = None,
        test_dataset: Optional[ChestXRayDatasetLightGBM] = None,
    ) -> Dict[str, float]:
        """
        Test the performance of the model.

        Args:
            x_test (Optional[Union[np.ndarray, pd.DataFrame]]): The testing
                                                                data features.
            y_test (Optional[Union[np.ndarray, pd.Series]]): The testing data
                                                             true labels.
            test_dataset (Optional[ChestXRayDatasetLightGBM]): The testing
                                                               dataset.

        Returns:
            Dict[str, float]: Metric(s) indicating the performance.
        """
        if test_dataset is not None:
            x_test, y_test = test_dataset.get_data()
        elif x_test is None or y_test is None:
            raise ValueError(
                "Must provide either (x_test, y_test) or test_dataset."
            )

        predictions = self.forward_pass(x_test)

        # TODO: Ensure correct evaluation metrics.
        macro_f1 = f1_score(y_test, predictions, average="macro")
        precision = precision_score(
            y_test, predictions, average="macro", zero_division=0
        )
        recall = recall_score(
            y_test, predictions, average="macro", zero_division=0
        )
        metrics = {
            "macro_f1": float(macro_f1),
            "precision": float(precision),
            "recall": float(recall),
        }

        LOGGER.info(f"Evaluation metrics: {metrics}")

        return metrics

    def _save_weights(self, path: Path) -> None:
        """
        Save model weights.

        Args:
            path (Path): The path to save the model weights to.
        """
        if self.model is None:
            raise ValueError("Cannot save an untrained model.")

        self.model.save_model(str(path))

    def _load_weights(self, path: Path) -> None:
        """
        Load model weights.

        Args:
            path (Path): The path to load the model weights from.
        """
        self.model = lgb.Booster(model_file=str(path))
