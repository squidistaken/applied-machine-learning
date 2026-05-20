import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, Dataset, WeightedRandomSampler
from typing import Optional, Union, cast
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from src.models.base import BaseModel
from src.data.dataset_pytorch import ChestXRayDatasetPyTorch
from src.data.dataset_lightgbm import ChestXRayDatasetLightGBM
from src.constants import LOGGER, DEVICE

# Define consistent plotting style for "aesthetic."
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
    ):
        """
        Initialise the class.

        Args:
            model (BaseModel): The model to train and evaluate.
            train_data (Union[ChestXRayDatasetLightGBM, ChestXRayDatasetPyTorch, Subset]): The training data.
            eval_data (Union[ChestXRayDatasetLightGBM, ChestXRayDatasetPyTorch, Subset]): The evaluation data.
            test_data (Optional[Union[ChestXRayDatasetLightGBM, ChestXRayDatasetPyTorch, Subset]]): The test data. Defaults to None.
            batch_size (int): The batch size (PyTorch only). Defaults to 32.
            device (str): The device to run models on. Defaults to DEVICE.
        """
        self.model = model
        self.batch_size = batch_size
        self.history: dict[str, list] = {}
        self.reset_history()
        self.is_pytorch = isinstance(self.model, nn.Module)

        if self.is_pytorch:
            self.device = torch.device(device)
            pytorch_model = cast(nn.Module, self.model)
            pytorch_model.to(self.device)

            # We handle class imbalance with the WeightedRandomSampler.
            sampler = None
            shuffle_train = True

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
                batch_size=batch_size,
                shuffle=shuffle_train,
                sampler=sampler,
            )
            self.eval_loader = DataLoader(
                cast(Dataset, eval_data), batch_size=batch_size, shuffle=False
            )
            self.test_loader = (
                DataLoader(
                    cast(Dataset, test_data),
                    batch_size=batch_size,
                    shuffle=False,
                )
                if test_data
                else None
            )
        else:
            self.train_data = train_data
            self.eval_data = eval_data
            self.test_data = test_data

    def reset_history(self) -> None:
        """Reset the training history."""
        self.history = {
            "train_loss": [],
            "eval_loss": [],
            "eval_macro_f1": [],
            "eval_precision": [],
            "eval_recall": [],
        }

    def train(
        self,
        num_epochs: int = 10,
        learning_rate: float = 1e-3,
        patience: int = 3,
    ) -> None:
        """
        Train the model.

        Args:
            num_epochs (int, optional): The number of epochs. Defaults to 10.
            learning_rate (float, optional): The learning rate. Defaults to
                                             1e-3.
            patience (int, optional): The patience for early stopping. Defaults
                                      to 3.
        """
        self.reset_history()

        if not self.is_pytorch:
            LOGGER.info(
                "Training LightGBM model utilizing its native backward_pass..."
            )
            evals_result = {}
            self.model.backward_pass(
                None,
                None,
                val_dataset=self.eval_data,
                num_boost_round=num_epochs,
                patience=patience,
                evals_result=evals_result,
            )

            if evals_result:
                if (
                    "train" in evals_result
                    and "multi_logloss" in evals_result["train"]
                ):
                    self.history["train_loss"] = evals_result["train"][
                        "multi_logloss"
                    ]
                if (
                    "val" in evals_result
                    and "multi_logloss" in evals_result["val"]
                ):
                    self.history["eval_loss"] = evals_result["val"][
                        "multi_logloss"
                    ]

                if "val" in evals_result:
                    if "macro_f1" in evals_result["val"]:
                        self.history["eval_macro_f1"] = evals_result["val"][
                            "macro_f1"
                        ]
                    if "precision" in evals_result["val"]:
                        self.history["eval_precision"] = evals_result["val"][
                            "precision"
                        ]
                    if "recall" in evals_result["val"]:
                        self.history["eval_recall"] = evals_result["val"][
                            "recall"
                        ]
            return

        pytorch_model = cast(nn.Module, self.model)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            pytorch_model.parameters(), lr=learning_rate
        )

        best_eval_loss = float("inf")
        patience_counter = 0

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

            LOGGER.info(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Eval Loss: {eval_metrics['loss']:.4f} | "
                f"Macro-F1: {eval_metrics['macro_f1']:.4f}"
            )

            if eval_metrics["loss"] < best_eval_loss:
                best_eval_loss = eval_metrics["loss"]
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    LOGGER.info(
                        f"Early stopping triggered after {epoch + 1} epochs."
                    )
                    break

    def get_predictions(
        self, use_test: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Get true labels and predicted labels for the evaluated dataset.

        Args:
            use_test (bool): Whether to use the test set. Defaults to False.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing the true labels
                                           and predicted labels.
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

        if use_test and self.test_loader is not None:
            loader = self.test_loader
        else:
            loader = self.eval_loader

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
        """
        Evaluate the model on the evaluation or test set.

        Args:
            use_test (bool): Whether to use the test set. Defaults to False.

        Returns:
            dict[str, float]: A dictionary containing the evaluation metrics.
        """
        if not self.is_pytorch:
            dataset = self.test_data if use_test else self.eval_data
            if not isinstance(dataset, ChestXRayDatasetLightGBM):
                raise TypeError(
                    f"Expected ChestXRayDatasetLightGBM for LightGBM evaluation, but got {type(dataset)}"
                )
            X, y = dataset.get_data()

            return cast(dict[str, float], self.model.evaluate(X, y))

        pytorch_model = cast(nn.Module, self.model)
        criterion = nn.CrossEntropyLoss()

        if use_test and self.test_loader is None:
            LOGGER.warning(
                "Test loader is not available. Evaluating on evaluation set instead."
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

        with torch.no_grad():
            for inputs, labels in tqdm(loader, desc="Evaluating", leave=False):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = pytorch_model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(loader)

        macro_f1 = f1_score(all_labels, all_preds, average="macro")
        precision = precision_score(
            all_labels, all_preds, average="macro", zero_division=0
        )
        recall = recall_score(
            all_labels, all_preds, average="macro", zero_division=0
        )

        return {
            "loss": float(avg_loss),
            "macro_f1": float(macro_f1),
            "precision": float(precision),
            "recall": float(recall),
        }

    def plot_history(
        self, show: bool = False, save_path: Optional[str] = None
    ) -> None:
        """
        Plot history of metrics.

        Args:
            show (bool, optional): Whether to show the plot immediately.
                                   Defaults to False.
            save_path (Optional[str], optional): The path to save the plot to. Defaults to None.
        """
        if not self.history.get("train_loss"):
            LOGGER.warning("No training history to plot.")
            return

        epochs_range = range(1, len(self.history["train_loss"]) + 1)

        has_metrics = bool(
            self.history.get("eval_macro_f1")
            and len(self.history["eval_macro_f1"]) > 0
        )

        if has_metrics:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
            ax2 = None

        # The left panel are loss metrics.
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

        # The right panel are validation metrics.
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

            ax2.set_xlabel("Epochs / Boosting Rounds")
            ax2.set_ylabel("Metric Score")
            ax2.set_title("Validation Classification Metrics")
            ax2.legend()
            ax2.set_ylim(-0.05, 1.05)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            LOGGER.info(f"Training history plot saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

    def plot_confusion_matrix(
        self,
        show: bool = False,
        save_path: Optional[str] = None,
        use_test: bool = False,
    ) -> None:
        """
        Plot the confusion matrix.

        Args:
            show (bool, optional): Whether to display the plot. Defaults to
                                   False.
            save_path (Optional[str], optional): The path to save the plot.
                                                 Defaults to None.
            use_test (bool): Whether to evaluate on the test set instead of
                             the validation set. Defaults to False.
        """
        LOGGER.info("Generating confusion matrix...")
        y_true, y_pred = self.get_predictions(use_test=use_test)
        cm = confusion_matrix(y_true, y_pred)

        if hasattr(self.model, "classes"):
            display_labels = self.model.classes
        else:
            display_labels = ["BACTERIA", "NORMAL", "VIRUS"]

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=display_labels
        )

        fig, ax = plt.subplots(figsize=(8, 6))
        disp.plot(cmap="Blues", ax=ax, xticks_rotation="horizontal")
        plt.title("Confusion Matrix")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            LOGGER.info(f"Confusion matrix plot saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

    def save_model(self, path: str) -> None:
        """
        Save the model.

        Args:
            path (Union[str, Path]): The path to save the model to.
        """
        self.model.save_model(path)

    def load_model(self, path: str) -> None:
        """
        Load the model.

        Args:
            path (Union[str, Path]): The path to load the model from.
        """
        self.model.load(path)
