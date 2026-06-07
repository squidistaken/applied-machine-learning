from src.models.cnn import CNN
import copy
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Union, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from src.constants import DEVICE, LOGGER, LOGS_DIR, RESULTS_DIR
from src.data.dataset_lightgbm import ChestXRayDatasetLightGBM
from src.data.dataset_pytorch import ChestXRayDatasetPyTorch
from src.models.base import BaseModel
from src.utils.uq_utils import (
    calculate_ece,
    calculate_predictive_entropy,
    calculate_brier_score,
    calculate_nll,
    compute_reliability_curve,
    selective_prediction_curve,
)
from src.utils.calibration import TemperatureScaler, probs_to_logits

plt.style.use("seaborn-v0_8-dark-palette")
plt.rcParams.update(
    {
        "figure.figsize": (12, 6),
        "axes.labelsize": 16,
        "axes.grid": True,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "axes.titlesize": 18,
        "legend.fontsize": 16,
        "lines.linewidth": 2,
        "text.usetex": False,
        "font.family": "serif",
        "image.cmap": "magma",
    }
)


class Trainer:
    """Trainer class."""

    def __init__(
        self,
        model: BaseModel,
        train_data: Union[
            ChestXRayDatasetLightGBM, ChestXRayDatasetPyTorch, Subset
        ],
        eval_data: Union[
            ChestXRayDatasetLightGBM, ChestXRayDatasetPyTorch, Subset
        ],
        test_data: Optional[
            Union[ChestXRayDatasetLightGBM, ChestXRayDatasetPyTorch, Subset]
        ] = None,
        batch_size: int = 32,
        device: str = DEVICE,
        enable_uq: bool = True,
    ):
        """Initialise the class.

        Args:
            model (BaseModel): The model instance to train and evaluate.
            train_data (Union[ChestXRayDatasetLightGBM, ChestXRayDatasetPyTorch, Subset]): The dataset to train on.
            eval_data (Union[ChestXRayDatasetLightGBM, ChestXRayDatasetPyTorch, Subset]): The dataset to validate on.
            test_data (Optional[Union[ChestXRayDatasetLightGBM, ChestXRayDatasetPyTorch, Subset]]): The dataset to test on.
            batch_size (int): Mini-batch processing volume (only used in PyTorch). Defaults to 32.
            device (str): THe targeted processing device. Defaults to DEVICE.
            enable_uq (bool): Whether to compute validation calibration errors. Defaults to True.
        """
        self.model = model
        self.batch_size = batch_size
        self.enable_uq = enable_uq
        self.is_pytorch = isinstance(self.model, nn.Module)

        self.history: dict[str, list[float]] = {}
        self.reset_history()

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_name = self.model.__class__.__name__
        self.tb_dir = (
            Path(LOGS_DIR) / "tensorboard" / f"{model_name}_{timestamp}"
        )
        self.writer = SummaryWriter(log_dir=str(self.tb_dir))

        LOGGER.info(f"TensorBoard initialized. Logs saved to: {self.tb_dir}")

        if self.is_pytorch:
            self.device = torch.device(device)
            pytorch_model = cast(nn.Module, self.model)
            pytorch_model.to(self.device)
            self._setup_pytorch_loaders(
                cast(ChestXRayDatasetPyTorch, train_data),
                cast(ChestXRayDatasetPyTorch, eval_data),
                cast(ChestXRayDatasetPyTorch, test_data),
            )
        else:
            self.train_data = train_data
            self.eval_data = eval_data
            self.test_data = test_data

    def _setup_pytorch_loaders(
        self,
        train_data: Union[ChestXRayDatasetPyTorch, Subset],
        eval_data: Union[ChestXRayDatasetPyTorch, Subset],
        test_data: Optional[Union[ChestXRayDatasetPyTorch, Subset]],
    ) -> None:
        """Set up PyTorch DataLoaders.

        Args:
            train_data (Union[ChestXRayDatasetPyTorch, Subset]): The training dataset.
            eval_data (Union[ChestXRayDatasetPyTorch, Subset]): The validation dataset.
            test_data (Optional[Union[ChestXRayDatasetPyTorch, Subset]]): The testing dataset.
        """
        sampler = None
        shuffle_train = True

        # We extract underlying dataset metadata to resolve class imbalance
        # issues.
        if isinstance(train_data, ChestXRayDatasetPyTorch):
            weights = train_data.compute_sample_weights()
            sampler = WeightedRandomSampler(
                weights, num_samples=len(weights), replacement=True
            )
            shuffle_train = False

        elif isinstance(train_data, Subset) and isinstance(
            train_data.dataset, ChestXRayDatasetPyTorch
        ):
            all_weights = train_data.dataset.compute_sample_weights()
            subset_weights = [all_weights[i] for i in train_data.indices]
            sampler = WeightedRandomSampler(
                subset_weights,
                num_samples=len(subset_weights),
                replacement=True,
            )
            shuffle_train = False

        self.train_loader = DataLoader(
            cast(Dataset, train_data),
            batch_size=self.batch_size,
            shuffle=shuffle_train,
            sampler=sampler,
        )
        self.eval_loader = DataLoader(
            cast(Dataset, eval_data), batch_size=self.batch_size, shuffle=False
        )
        self.test_loader = (
            DataLoader(
                cast(Dataset, test_data),
                batch_size=self.batch_size,
                shuffle=False,
            )
            if test_data
            else None
        )

    def reset_history(self) -> None:
        """Reset the internal training and validation performance metrics
        histories.
        """
        self.history = {
            "train_loss": [],
            "eval_loss": [],
            "eval_macro_f1": [],
            "eval_precision": [],
            "eval_recall": [],
            "eval_ece": [],
            "eval_brier": [],
            "eval_predictive_entropy": [],
        }

    def train(
        self,
        num_epochs: int = 10,
        learning_rate: float = 1e-3,
        patience: int = 3,
        epoch_callback: Optional[Callable[[int, int, dict], None]] = None,
    ) -> None:
        """Train the model.

        Args:
            num_epochs (int): The number of training epochs or boosting rounds. Defaults to 10.
            learning_rate (float): The initial learning step increment for PyTorch optimizers. Defaults to 1e-3.
            patience (int): The patience level before early stopping. Defaults to 3.
            epoch_callback (Optional[Callable[[int, int, dict], None]]): Called as
                ``(current_epoch, total_epochs, metrics)`` at the end of each
                epoch/boosting round, to support live progress reporting.
        """
        self.reset_history()

        if self.is_pytorch:
            self._train_pytorch(
                num_epochs, learning_rate, patience, epoch_callback
            )
        else:
            self._train_lightgbm(num_epochs, patience, epoch_callback)

    def _train_lightgbm(
        self,
        num_epochs: int,
        patience: int,
        epoch_callback: Optional[Callable[[int, int, dict], None]] = None,
    ) -> None:
        """Train the LightGBM model.

        Args:
            num_epochs (int): The number of boosting iterations to perform.
            patience (int): The iterative patience before halting the boosting sequence.
        """
        LOGGER.info(
            "Training LightGBM model utilizing its native backward_pass..."
        )
        evals_result: dict = {}

        self.model.backward_pass(
            None,
            None,
            val_dataset=self.eval_data,
            num_boost_round=num_epochs,
            patience=patience,
            evals_result=evals_result,
            enable_uq=self.enable_uq,
            epoch_callback=epoch_callback,
        )

        if not evals_result:
            LOGGER.warning(
                "No evaluation results returned from LightGBM training process."
            )
            self.writer.close()
            return

        val_res = evals_result.get("val", {})
        train_res = evals_result.get("train", {})

        self.history["train_loss"] = train_res.get("multi_logloss", [])
        self.history["eval_loss"] = val_res.get("multi_logloss", [])
        self.history["eval_macro_f1"] = val_res.get("macro_f1", [])
        self.history["eval_precision"] = val_res.get("precision", [])
        self.history["eval_recall"] = val_res.get("recall", [])

        if self.enable_uq:
            self.history["eval_ece"] = val_res.get("ece", [])
            self.history["eval_brier"] = val_res.get("brier", [])
            self.history["eval_predictive_entropy"] = val_res.get(
                "predictive_entropy", []
            )

        epochs_run = len(self.history["train_loss"])
        for i in range(epochs_run):
            self.writer.add_scalar(
                "Loss/train", self.history["train_loss"][i], i
            )
            if i < len(self.history["eval_loss"]):
                self.writer.add_scalar(
                    "Loss/eval", self.history["eval_loss"][i], i
                )
            if i < len(self.history["eval_macro_f1"]):
                self.writer.add_scalar(
                    "Metrics/macro_f1", self.history["eval_macro_f1"][i], i
                )
            if i < len(self.history["eval_precision"]):
                self.writer.add_scalar(
                    "Metrics/precision", self.history["eval_precision"][i], i
                )
            if i < len(self.history["eval_recall"]):
                self.writer.add_scalar(
                    "Metrics/recall", self.history["eval_recall"][i], i
                )

            if self.enable_uq:
                if i < len(self.history["eval_ece"]):
                    self.writer.add_scalar(
                        "Calibration/ece", self.history["eval_ece"][i], i
                    )
                if i < len(self.history["eval_brier"]):
                    self.writer.add_scalar(
                        "Calibration/brier_score",
                        self.history["eval_brier"][i],
                        i,
                    )
                if i < len(self.history["eval_predictive_entropy"]):
                    self.writer.add_scalar(
                        "Uncertainty/predictive_entropy",
                        self.history["eval_predictive_entropy"][i],
                        i,
                    )

        self.writer.close()

    def _train_pytorch(
        self,
        num_epochs: int,
        learning_rate: float,
        patience: int,
        epoch_callback: Optional[Callable[[int, int, dict], None]] = None,
    ) -> None:
        """Train the PyTorch model(s).

        Args:
            num_epochs (int): The total training cycles to run.
            learning_rate (float): The learning rate parameter fed to the Adam Optimiser.
            patience (int): The patience level before early stopping.
        """
        pytorch_model = cast(CNN, self.model)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            pytorch_model.parameters(), lr=learning_rate
        )

        best_eval_loss = float("inf")
        best_model_weights = None
        patience_counter = 0

        try:
            dataiter = iter(self.train_loader)
            images, _ = next(dataiter)
            img_grid = torchvision.utils.make_grid(images[:16], normalize=True)
            self.writer.add_image(
                "Chest_X-Ray_Training_Sample", img_grid, global_step=0
            )
        except Exception as e:
            LOGGER.warning(f"Could not log images to TensorBoard: {e}")

        for epoch in range(num_epochs):
            pytorch_model.train()
            total_loss = 0.0

            progress_bar = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch + 1}/{num_epochs} [Training]",
                leave=False,
            )

            for inputs, labels in progress_bar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = pytorch_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_train_loss = total_loss / len(self.train_loader)
            self.history["train_loss"].append(avg_train_loss)

            eval_metrics = self.evaluate(use_test=False)

            self.history["eval_loss"].append(eval_metrics["loss"])
            self.history["eval_macro_f1"].append(eval_metrics["macro_f1"])
            self.history["eval_precision"].append(eval_metrics["precision"])
            self.history["eval_recall"].append(eval_metrics["recall"])
            self.history["eval_ece"].append(eval_metrics["ece"])
            self.history["eval_brier"].append(eval_metrics["brier_score"])
            self.history["eval_predictive_entropy"].append(
                eval_metrics["predictive_entropy"]
            )

            self.writer.add_scalar("Loss/train", avg_train_loss, epoch)
            self.writer.add_scalar("Loss/eval", eval_metrics["loss"], epoch)
            self.writer.add_scalar(
                "Metrics/macro_f1", eval_metrics["macro_f1"], epoch
            )
            self.writer.add_scalar(
                "Metrics/precision", eval_metrics["precision"], epoch
            )
            self.writer.add_scalar(
                "Metrics/recall", eval_metrics["recall"], epoch
            )

            if self.enable_uq:
                self.writer.add_scalar(
                    "Calibration/ece", eval_metrics["ece"], epoch
                )
                self.writer.add_scalar(
                    "Calibration/brier_score",
                    eval_metrics["brier_score"],
                    epoch,
                )
                self.writer.add_scalar(
                    "Uncertainty/predictive_entropy",
                    eval_metrics["predictive_entropy"],
                    epoch,
                )

            LOGGER.info(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Eval Loss: {eval_metrics['loss']:.4f} | "
                f"Macro-F1: {eval_metrics['macro_f1']:.4f} | "
                f"ECE: {eval_metrics['ece']:.4f} | "
                f"Brier: {eval_metrics['brier_score']:.4f}"
            )

            if epoch_callback is not None:
                epoch_callback(
                    epoch + 1,
                    num_epochs,
                    {"train_loss": avg_train_loss, **eval_metrics},
                )

            if eval_metrics["loss"] < best_eval_loss:
                best_eval_loss = eval_metrics["loss"]
                patience_counter = 0
                best_model_weights = copy.deepcopy(pytorch_model.state_dict())
                LOGGER.info(
                    f"--> New best Validation Loss: {best_eval_loss:.4f}. Checkpoint stored in memory!"
                )
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    LOGGER.info(
                        f"Early stopping triggered after {epoch + 1} epochs."
                    )
                    break

        if best_model_weights is not None:
            LOGGER.info(
                f"Restoring best model weights (Validation Loss: {best_eval_loss:.4f})"
            )
            pytorch_model.load_state_dict(best_model_weights)

        self.writer.close()

    def get_predictions(
        self, use_test: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Collect ground-truth labels and model prediction indices.

        Args:
            use_test (bool): Whether to use the test set for predicting. Defaults to False.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing the ground-truth
                                           label index mappings and the
                                           generated model prediction index
                                           classes..

        Raises:
            TypeError: If evaluating LightGBM and the input is not of type
                       ChestXRayDatasetLightGBM.
        """
        if not self.is_pytorch:
            dataset = self.test_data if use_test else self.eval_data
            if not isinstance(dataset, ChestXRayDatasetLightGBM):
                raise TypeError(
                    f"Expected ChestXRayDatasetLightGBM for LightGBM predictions, but got {type(dataset)}"
                )
            X, y = dataset.get_data()
            preds = self.model.forward_pass(X)
            y_array = y.to_numpy() if hasattr(y, "to_numpy") else np.array(y)
            return y_array, preds

        pytorch_model = cast(nn.Module, self.model)
        loader = (
            self.test_loader
            if (use_test and self.test_loader is not None)
            else self.eval_loader
        )

        pytorch_model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in tqdm(
                loader, desc="Getting predictions...", leave=False
            ):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = pytorch_model(inputs)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        return np.array(all_labels), np.array(all_preds)

    def evaluate(self, use_test: bool = False) -> dict[str, float]:
        """Evaluate the model.

        Args:
            use_test (bool): Whether to use the test set for evaluation. Defaults to False.

        Returns:
            dict[str, float]: A metric(s).

        Raises:
            TypeError: If running LightGBM evaluation and input is not of type
                       ChestXRayDatasetLightGBM.
        """
        if not self.is_pytorch:
            dataset = self.test_data if use_test else self.eval_data
            if not isinstance(dataset, ChestXRayDatasetLightGBM):
                raise TypeError(
                    f"Expected ChestXRayDatasetLightGBM for LightGBM evaluation, but got {type(dataset)}"
                )
            X, y = dataset.get_data()

            if hasattr(self.model, "evaluate"):
                return cast(
                    dict[str, float],
                    self.model.evaluate(X, y, enable_uq=self.enable_uq),
                )
            return cast(dict[str, float], self.model.evaluate(X, y))

        pytorch_model = cast(nn.Module, self.model)
        criterion = nn.CrossEntropyLoss()

        if use_test and self.test_loader is None:
            LOGGER.warning(
                "Test loader is unavailable. Evaluating on validation dataset instead."
            )
            loader = self.eval_loader
        elif use_test and self.test_loader is not None:
            loader = self.test_loader
        else:
            loader = self.eval_loader

        pytorch_model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for inputs, labels in tqdm(loader, desc="Evaluating", leave=False):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = pytorch_model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                if self.enable_uq:
                    probs = torch.softmax(outputs, dim=1)
                    all_probs.extend(probs.cpu().numpy())

                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(loader)
        all_labels_np = np.array(all_labels)
        all_preds_np = np.array(all_preds)

        macro_f1 = f1_score(all_labels_np, all_preds_np, average="macro")
        precision = precision_score(
            all_labels_np, all_preds_np, average="macro", zero_division=0
        )
        recall = recall_score(
            all_labels_np, all_preds_np, average="macro", zero_division=0
        )

        if self.enable_uq and all_probs:
            all_probs_np = np.array(all_probs)
            ece = calculate_ece(all_labels_np, all_probs_np)
            entropies = calculate_predictive_entropy(all_probs_np)
            mean_entropy = float(np.mean(entropies))
            brier = calculate_brier_score(all_labels_np, all_probs_np)
            nll = calculate_nll(all_labels_np, all_probs_np)
        else:
            ece = 0.0
            mean_entropy = 0.0
            brier = 0.0
            nll = 0.0

        return {
            "loss": float(avg_loss),
            "macro_f1": float(macro_f1),
            "precision": float(precision),
            "recall": float(recall),
            "ece": float(ece),
            "predictive_entropy": float(mean_entropy),
            "brier_score": float(brier),
            "nll": float(nll),
        }

    def _collect_outputs(
        self, use_test: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Collect labels, probabilities and logits for a dataset split.

        Args:
            use_test (bool): Whether to use the test split. Defaults to False.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing the
                                                       true labels, the
                                                       predicted probabilities,
                                                       and the logits.
        """
        if not self.is_pytorch:
            dataset = self.test_data if use_test else self.eval_data
            if not isinstance(dataset, ChestXRayDatasetLightGBM):
                raise TypeError(
                    f"Expected ChestXRayDatasetLightGBM, but got {type(dataset)}"
                )
            X, y = dataset.get_data()
            booster = getattr(self.model, "model", None)
            if booster is None:
                raise ValueError("LightGBM model is not trained yet.")
            probs = np.asarray(booster.predict(X))
            labels = y.to_numpy() if hasattr(y, "to_numpy") else np.array(y)
            return labels, probs, probs_to_logits(probs)

        pytorch_model = cast(nn.Module, self.model)
        loader = (
            self.test_loader
            if (use_test and self.test_loader is not None)
            else self.eval_loader
        )

        pytorch_model.eval()
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in tqdm(
                loader, desc="Collecting outputs...", leave=False
            ):
                inputs = inputs.to(self.device)
                logits = pytorch_model(inputs)
                all_logits.extend(logits.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        logits_np = np.array(all_logits)
        labels_np = np.array(all_labels)

        # Stable softmax to recover probabilities from the raw logits.
        shifted = logits_np - logits_np.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        probs_np = exp / exp.sum(axis=1, keepdims=True)

        return labels_np, probs_np, logits_np

    def fit_temperature_scaler(self) -> TemperatureScaler:
        """Fit a temperature scaler on the validation split.

        Returns:
            TemperatureScaler: A fitted scaler.
        """
        labels, _, logits = self._collect_outputs(use_test=False)
        scaler = TemperatureScaler().fit(logits, labels)
        LOGGER.info(f"Fitted temperature scaling: T = {scaler.temperature:.4f}")
        return scaler

    def uq_report(
        self,
        use_test: bool = True,
        calibrator: Optional[TemperatureScaler] = None,
    ) -> dict[str, float]:
        """Compute an uncertainty/calibration report for a split.

        Args:
            use_test (bool): Whether to use the test split. Defaults to True.
            calibrator (Optional[TemperatureScaler]): The calibrated metrics.

        Returns:
            dict[str, float]: A metric(s).
        """
        labels, probs, logits = self._collect_outputs(use_test=use_test)

        report = {
            "ece": calculate_ece(labels, probs),
            "brier_score": calculate_brier_score(labels, probs),
            "nll": calculate_nll(labels, probs),
            "aurc": float(selective_prediction_curve(labels, probs)["aurc"]),
        }

        if calibrator is not None:
            cal_probs = calibrator.predict_proba(logits)
            report["temperature"] = float(calibrator.temperature)
            report["ece_calibrated"] = calculate_ece(labels, cal_probs)
            report["brier_score_calibrated"] = calculate_brier_score(
                labels, cal_probs
            )
            report["nll_calibrated"] = calculate_nll(labels, cal_probs)

        return report

    def plot_reliability_diagram(
        self,
        use_test: bool = True,
        show: bool = False,
        calibrator: Optional[TemperatureScaler] = None,
        num_bins: int = 10,
    ) -> None:
        """Plot a reliability diagram (confidence vs. accuracy).

        Args:
            use_test (bool): Whether to use the test split. Defaults to True.
            show (bool): Whether to display the figure. Defaults to False.
            calibrator (Optional[TemperatureScaler]): The calibrator to overlay.
            num_bins (int): The number of confidence bins. Defaults to 10.
        """
        if not self.enable_uq:
            LOGGER.warning("enable_uq is False; skipping reliability diagram.")
            return

        labels, probs, logits = self._collect_outputs(use_test=use_test)
        curve = compute_reliability_curve(labels, probs, num_bins)
        ece = calculate_ece(labels, probs, num_bins)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            color="gray",
            label="Perfect calibration",
        )
        ax.plot(
            curve["bin_confidence"],
            curve["bin_accuracy"],
            marker="o",
            color="crimson",
            label=f"Uncalibrated (ECE={ece:.3f})",
        )

        if calibrator is not None:
            cal_probs = calibrator.predict_proba(logits)
            cal_curve = compute_reliability_curve(labels, cal_probs, num_bins)
            cal_ece = calculate_ece(labels, cal_probs, num_bins)
            ax.plot(
                cal_curve["bin_confidence"],
                cal_curve["bin_accuracy"],
                marker="s",
                color="forestgreen",
                label=(
                    f"Calibrated, T={calibrator.temperature:.2f} "
                    f"(ECE={cal_ece:.3f})"
                ),
            )

        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title("Reliability Diagram")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc="upper left")
        fig.tight_layout()

        if hasattr(self, "writer") and self.writer is not None:
            self.writer.add_figure(
                "Evaluation/Reliability_Diagram", fig, global_step=0
            )

        model_name = self.model.__class__.__name__
        suffix = "test" if use_test else "val"
        save_path = (
            Path(RESULTS_DIR) / f"{model_name}_reliability_diagram_{suffix}.png"
        )
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        LOGGER.info(f"Reliability diagram saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

    def plot_selective_prediction(
        self,
        use_test: bool = True,
        show: bool = False,
    ) -> None:
        """Plot the selective-prediction (accuracy-rejection) curve.

        Args:
            use_test (bool): Whether to use the test split. Defaults to True.
            show (bool): Whether to display the figure. Defaults to False.
        """
        if not self.enable_uq:
            LOGGER.warning(
                "enable_uq is False; skipping selective-prediction plot."
            )
            return

        labels, probs, _ = self._collect_outputs(use_test=use_test)
        # Rank by predictive entropy so abstention uses the same signal the API
        # reports as its uncertainty score.
        entropy = calculate_predictive_entropy(probs)
        curve = selective_prediction_curve(labels, probs, uncertainty=entropy)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            curve["coverage"],
            curve["accuracy"],
            marker="o",
            color="crimson",
            label=f"Accuracy (AURC={curve['aurc']:.3f})",
        )
        ax.set_xlabel("Coverage (fraction of most confident predictions kept)")
        ax.set_ylabel("Accuracy on kept predictions")
        ax.set_title("Selective Prediction: Accuracy vs. Coverage")
        ax.set_xlim(0, 1)
        ax.legend(loc="lower left")
        fig.tight_layout()

        if hasattr(self, "writer") and self.writer is not None:
            self.writer.add_figure(
                "Evaluation/Selective_Prediction", fig, global_step=0
            )

        model_name = self.model.__class__.__name__
        suffix = "test" if use_test else "val"
        save_path = (
            Path(RESULTS_DIR)
            / f"{model_name}_selective_prediction_{suffix}.png"
        )
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        LOGGER.info(f"Selective-prediction plot saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

    def plot_history(
        self,
        show: bool = False,
    ) -> None:
        """Plot the history.

        Args:
            show (bool): Whether to immediately show the plots. Defaults to
                         False.
        """
        if not self.history.get("train_loss"):
            LOGGER.warning("No training history data found to plot.")
            return

        epochs_range = range(1, len(self.history["train_loss"]) + 1)
        has_metrics = bool(self.history.get("eval_macro_f1"))

        fig, axes = plt.subplots(
            1, 2 if has_metrics else 1, figsize=(16 if has_metrics else 10, 6)
        )
        ax1 = axes[0] if has_metrics else axes
        ax2 = axes[1] if has_metrics else None

        # Axis panel 1: Loss curves
        ax1.plot(
            epochs_range,
            self.history["train_loss"],
            label="Train Loss",
            marker="o",
        )
        if self.history.get("eval_loss"):
            eval_epochs_range = range(1, len(self.history["eval_loss"]) + 1)
            ax1.plot(
                eval_epochs_range,
                self.history["eval_loss"],
                label="Eval Loss",
                marker="s",
            )

        ax1.set_xlabel("Epochs / Boosting Rounds")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training and Evaluation Loss")
        ax1.legend()

        # Axis panel 2: Classification Accuracy and Uncertainty
        if ax2 is not None:
            eval_epochs_range = range(1, len(self.history["eval_macro_f1"]) + 1)
            ax2.plot(
                eval_epochs_range,
                self.history["eval_macro_f1"],
                label="Validation Macro-F1",
                marker="^",
                color="crimson",
            )
            if self.history.get("eval_precision"):
                ax2.plot(
                    eval_epochs_range,
                    self.history["eval_precision"],
                    label="Validation Precision",
                    marker="d",
                    color="forestgreen",
                )
            if self.history.get("eval_recall"):
                ax2.plot(
                    eval_epochs_range,
                    self.history["eval_recall"],
                    label="Validation Recall",
                    marker="v",
                    color="darkorange",
                )
            if self.enable_uq:
                if self.history.get("eval_ece"):
                    ax2.plot(
                        eval_epochs_range,
                        self.history["eval_ece"],
                        label="Expected Calibration Error (ECE)",
                        marker="x",
                        color="purple",
                    )
                if self.history.get("eval_brier"):
                    ax2.plot(
                        eval_epochs_range,
                        self.history["eval_brier"],
                        label="Brier Score",
                        marker="P",
                        color="teal",
                    )
                if self.history.get("eval_predictive_entropy"):
                    ax2.plot(
                        eval_epochs_range,
                        self.history["eval_predictive_entropy"],
                        label="Predictive Entropy",
                        marker="o",
                        color="pink",
                    )

            ax2.set_xlabel("Epochs / Boosting Rounds")
            ax2.set_ylabel("Metric Score")
            ax2.set_title("Validation Metrics & Calibration")
            ax2.legend()
            ax2.set_ylim(-0.05, 1.05)

        plt.tight_layout()

        model_name = self.model.__class__.__name__
        save_path = Path(RESULTS_DIR) / f"{model_name}_training_history.png"

        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        LOGGER.info(f"Training history plot saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

    def plot_confusion_matrix(
        self,
        show: bool = False,
        use_test: bool = False,
    ) -> None:
        """Plot a confusion matrix.

        Args:
            show (bool): Whether to immediately show the plots. Defaults to False.
            use_test (bool): Whether to use the test set for evaluation. Defaults to False.
        """
        LOGGER.info("Generating confusion matrix...")
        y_true, y_pred = self.get_predictions(use_test=use_test)

        display_labels = getattr(
            self.model, "classes", ["BACTERIA", "NORMAL", "VIRUS"]
        )
        labels = list(range(len(display_labels)))
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=display_labels
        )
        fig, ax = plt.subplots(figsize=(8, 6))
        disp.plot(cmap="Blues", ax=ax, xticks_rotation="horizontal")

        ax.grid(False)
        ax.set_title("Confusion Matrix")
        fig.tight_layout()

        if hasattr(self, "writer") and self.writer is not None:
            self.writer.add_figure(
                "Evaluation/Confusion_Matrix", fig, global_step=0
            )

        model_name = self.model.__class__.__name__
        suffix = "test" if use_test else "val"
        save_path = (
            Path(RESULTS_DIR) / f"{model_name}_confusion_matrix_{suffix}.png"
        )

        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        LOGGER.info(f"Confusion matrix plot saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

    def save_model(self, path: Union[str, Path]) -> None:
        """
        Save a model and its weights.

        Args:
            filename (str): The path to save the model to.
        """
        self.model.save_model(str(path))

    def load_model(self, path: Union[str, Path]) -> None:
        """
        Load a trained model.

        Args:
            filename (str): The path to load the model from.
        """
        self.model.load(str(path))
