from src.utils.uq_utils import (
    calculate_ece,
    calculate_brier_score,
    calculate_nll,
)
import lightgbm as lgb
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from pathlib import Path
from typing import Callable, Optional, Union, cast
import pandas as pd
from src.models.base import BaseModel
from src.data.dataset_lightgbm import ChestXRayDatasetLightGBM
from src.constants import LOGGER
from src.utils.uq_utils import calculate_predictive_entropy


class LightGBM(BaseModel):
    """
    LightGBM Class for Chest X-Ray Image Classification.
    """

    def __init__(
        self,
        dataset: ChestXRayDatasetLightGBM,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        max_depth: int = -1,
        params: Optional[dict] = None,
    ) -> None:
        """Initialise the class.

        Args:
            dataset (ChestXRayDatasetLightGBM): The dataset to train on.
            learning_rate (float, optional): The learning rate for training.
                                             Defaults to 0.05.
            num_leaves (int, optional): The number of leaves in the tree.
                                        Defaults to 31.
            max_depth (int, optional): The maximum depth of the tree.
                                       Defaults to -1.
            params (Optional[dict], optional): The additional hyperparameters.
                                               Defaults to None.
        """
        super().__init__(dataset=dataset, file_extension=".txt")

        self.params = {
            "objective": "multiclass",
            "num_class": len(self.classes),
            "metric": "multi_logloss",
            "boosting_type": "gbdt",
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "max_depth": max_depth,
            "verbose": -1,
            "random_state": 42,
            "class_weight": "balanced",
            "lambda_l1": 0.1,
            "lambda_l2": 0.2,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "min_data_in_leaf": 20,
        }

        # Safely update with any additional custom parameters passed in.
        if params:
            self.params.update(params)

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
        x_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_val: Optional[Union[np.ndarray, pd.Series]] = None,
        val_dataset: Optional[ChestXRayDatasetLightGBM] = None,
        num_boost_round: int = 100,
        patience: int = 3,
        evals_result: Optional[dict] = None,
        enable_uq: bool = True,
        epoch_callback: Optional[Callable[[int, int, dict], None]] = None,
        **kwargs,
    ) -> None:
        """
        Train the model.

        Args:
            x_train (Optional[Union[np.ndarray, pd.DataFrame]]): The training
                                                                 data.
            y_train (Optional[Union[np.ndarray, pd.Series]]): The training
                                                              labels.
            x_val (Optional[Union[np.ndarray, pd.DataFrame]]): The validation features (for CV splits).
            y_val (Optional[Union[np.ndarray, pd.Series]]): The validation labels (for CV splits).
            val_dataset (Optional[ChestXRayDatasetLightGBM]): The validation dataset wrapper (for Trainer runs).
            num_boost_round (int): The number of boosting rounds.
            patience (int): Early stopping patience threshold.
            evals_result (Optional[dict]): Dictionary to store evaluation history.
            enable_uq (bool): Whether to enable uncertainty quantification. Defaults to True.
            epoch_callback (Optional[Callable[[int, int, dict], None]]): The epoch callback.
            **kwargs: The additional hyperparameters or configuration.
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

        if evals_result is not None:
            callbacks.append(lgb.record_evaluation(evals_result))

        val_X, val_y = None, None
        lgb_val = None

        if val_dataset is not None:
            val_X, val_y = val_dataset.get_data()
            lgb_val = val_dataset.get_lgb_dataset(free_raw_data=False)
        elif x_val is not None and y_val is not None:
            val_X, val_y = x_val, y_val
            lgb_val = lgb.Dataset(x_val, label=y_val)

        if lgb_val is not None:
            valid_sets.append(lgb_val)
            valid_names.append("val")

            if patience > 0:
                callbacks.append(lgb.early_stopping(stopping_rounds=patience))

            callbacks.append(lgb.log_evaluation(period=10))

            if val_X is not None and val_y is not None:

                def record_custom_metrics(env):
                    preds = env.model.predict(
                        val_X, num_iteration=env.iteration + 1
                    )
                    pred_labels = np.argmax(preds, axis=1)
                    f1 = float(
                        f1_score(
                            val_y, pred_labels, average="macro", zero_division=0
                        )
                    )
                    prec = float(
                        precision_score(
                            val_y, pred_labels, average="macro", zero_division=0
                        )
                    )
                    rec = float(
                        recall_score(
                            val_y, pred_labels, average="macro", zero_division=0
                        )
                    )

                    if enable_uq:
                        val_y_np = np.asarray(val_y)
                        ece = float(calculate_ece(val_y_np, preds))
                        entropies = calculate_predictive_entropy(preds)
                        mean_entropy = float(np.mean(entropies))
                        brier = float(calculate_brier_score(val_y_np, preds))
                    else:
                        ece = 0.0
                        mean_entropy = 0.0
                        brier = 0.0

                    if evals_result is not None:
                        if "val" not in evals_result:
                            evals_result["val"] = {}
                        if "macro_f1" not in evals_result["val"]:
                            evals_result["val"]["macro_f1"] = []
                        if "precision" not in evals_result["val"]:
                            evals_result["val"]["precision"] = []
                        if "recall" not in evals_result["val"]:
                            evals_result["val"]["recall"] = []
                        if "ece" not in evals_result["val"]:
                            evals_result["val"]["ece"] = []
                        if "brier" not in evals_result["val"]:
                            evals_result["val"]["brier"] = []
                        if "predictive_entropy" not in evals_result["val"]:
                            evals_result["val"]["predictive_entropy"] = []

                        evals_result["val"]["macro_f1"].append(f1)
                        evals_result["val"]["precision"].append(prec)
                        evals_result["val"]["recall"].append(rec)
                        evals_result["val"]["ece"].append(ece)
                        evals_result["val"]["brier"].append(brier)
                        evals_result["val"]["predictive_entropy"].append(
                            mean_entropy
                        )

                callbacks.append(record_custom_metrics)

        if epoch_callback is not None:

            def report_progress(env):
                metrics: dict = {}
                val_res = (
                    evals_result.get("val", {})
                    if evals_result is not None
                    else {}
                )
                for key in [
                    "multi_logloss",
                    "macro_f1",
                    "precision",
                    "recall",
                    "ece",
                ]:
                    series = val_res.get(key)
                    if series:
                        # Rename loss for a consistent live-chart schema.
                        out_key = "eval_loss" if key == "multi_logloss" else key
                        metrics[out_key] = float(series[-1])
                epoch_callback(env.iteration + 1, num_boost_round, metrics)

            callbacks.append(report_progress)

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
        enable_uq: bool = True,
        test_dataset: Optional[ChestXRayDatasetLightGBM] = None,
        **kwargs,
    ) -> dict[str, float]:
        """
        Test the performance of the model.

        Args:
            x_test (Optional[Union[np.ndarray, pd.DataFrame]]): The testing
                                                                data features.
            y_test (Optional[Union[np.ndarray, pd.Series]]): The testing data
                                                             true labels.
            test_dataset (Optional[ChestXRayDatasetLightGBM]): The testing
                                                               dataset.
            enable_uq (bool): Whether to enable uncertainty quantification. Defaults to True.

        Returns:
            dict[str, float]: Metric(s) indicating the performance.
        """
        if test_dataset is not None:
            x_test, y_test = test_dataset.get_data()
        elif x_test is None or y_test is None:
            raise ValueError(
                "Must provide either (x_test, y_test) or test_dataset."
            )

        model = self.model
        if model is None:
            raise ValueError(
                "Model is not trained yet. Call backward_pass first."
            )

        y_pred_prob = np.asarray(cast(lgb.Booster, self.model).predict(x_test))
        predictions = self.forward_pass(x_test)

        macro_f1 = f1_score(y_test, predictions, average="macro")
        precision = precision_score(
            y_test, predictions, average="macro", zero_division=0
        )
        recall = recall_score(
            y_test, predictions, average="macro", zero_division=0
        )

        if enable_uq:
            y_test_np = np.asarray(y_test)
            ece = calculate_ece(y_test_np, y_pred_prob)
            entropies = calculate_predictive_entropy(y_pred_prob)
            mean_entropy = float(np.mean(entropies))
            brier = calculate_brier_score(y_test_np, y_pred_prob)
            nll = calculate_nll(y_test_np, y_pred_prob)
        else:
            ece = 0.0
            mean_entropy = 0.0
            brier = 0.0
            nll = 0.0

        metrics = {
            "macro_f1": float(macro_f1),
            "precision": float(precision),
            "recall": float(recall),
            "ece": float(ece),
            "predictive_entropy": float(mean_entropy),
            "brier_score": float(brier),
            "nll": float(nll),
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
