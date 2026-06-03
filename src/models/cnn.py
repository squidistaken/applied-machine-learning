import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from typing import Optional, cast
from pathlib import Path
from src.models.base import BaseModel
from src.data.dataset_pytorch import ChestXRayDatasetPyTorch
from src.constants import LOGGER
from src.utils.uq_utils import calculate_ece, calculate_predictive_entropy


class CNN(BaseModel, nn.Module):
    """
    Baseline CNN Class for Chest X-Ray Image Classification.
    """

    def __init__(
        self,
        dataset: ChestXRayDatasetPyTorch,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
    ) -> None:
        """Initialise the class.

        Args:
            dataset (ChestXRayDatasetPyTorch): The dataset to train on.
            learning_rate (float): The learning rate for training. Defaults to
                                   0.001.
            weight_decay (float): The weight decay (L2 penalty). Defaults to
                                  1e-4.
        """
        BaseModel.__init__(self, dataset)
        nn.Module.__init__(self)

        self.num_classes = len(self.classes)

        # To prevent overfitting, we add batch normalisation and dropout.
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=16, kernel_size=3, padding=1
        )
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm2d(32)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(32, self.num_classes)

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass.

        Args:
            x (torch.Tensor): The input data tensor.

        Returns:
            torch.Tensor: An output tensor.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.global_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        x = self.fc(x)

        return x

    def forward_pass(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict the class of the input.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: A list of predicted class labels or probabilities.
        """
        self.eval()

        with torch.no_grad():
            outputs = self.forward(x)
            predictions = torch.argmax(outputs, dim=1)

        return predictions

    def backward_pass(
        self,
        x_train: ChestXRayDatasetPyTorch,
        y_train: Optional[ChestXRayDatasetPyTorch] = None,
        epochs: int = 10,
        **kwargs,
    ) -> None:
        """
        Train the model.

        Args:
            x_train (ChestXRayDatasetPyTorch): The training DataLoader.
            y_train (Optional[ChestXRayDatasetPyTorch]): The training labels
                                                         (ignored as the images
                                                         provide the labels.).
            epochs (int): The number of epochs.
            **kwargs: The additional hyperparameters or configurations.
        """
        device = next(self.parameters()).device
        self.train()

        for epoch in range(epochs):
            total_loss = 0.0
            for images, labels in x_train:
                images = cast(torch.Tensor, images).to(device)
                labels = cast(torch.Tensor, labels).to(device)

                self.optimizer.zero_grad()
                outputs = self.forward(images)
                loss = self.loss_function(outputs, labels)

                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(x_train)

            LOGGER.info(
                f"CNN | Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}"
            )

    def evaluate(
        self,
        x_test: ChestXRayDatasetPyTorch,
        y_test: Optional[ChestXRayDatasetPyTorch] = None,
        enable_uq: bool = True,
        **kwargs,
    ) -> dict[str, float]:
        """
        Test the performance of the model.

        Args:
            x_test (ChestXRayDatasetPyTorch): The testing data features.
            y_test (Optional[ChestXRayDatasetPyTorch]): The testing data true
                                                        labels (Ignored as the
                                                        images provide the
                                                        labels.).
            enable_uq (bool): Whether to enable uncertainty quantification.
                              Defaults to True.

        Returns:
            dict[str, float]: Metric(s) indicating the performance.
        """
        device = next(self.parameters()).device
        self.eval()

        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for images, labels in x_test:
                images = cast(torch.Tensor, images).to(device)
                labels = cast(torch.Tensor, labels).to(device)
                outputs = self.forward(images)
                predictions = torch.argmax(outputs, dim=1)

                if enable_uq:
                    probs = torch.softmax(outputs, dim=1)
                    all_probs.extend(probs.cpu().numpy())

                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        all_labels_np = np.array(all_labels)
        macro_f1 = f1_score(all_labels_np, all_preds, average="macro")
        precision = precision_score(
            all_labels_np, all_preds, average="macro", zero_division=0
        )
        recall = recall_score(
            all_labels_np, all_preds, average="macro", zero_division=0
        )

        if enable_uq and len(all_probs) > 0:
            all_probs_np = np.array(all_probs)
            ece = calculate_ece(all_labels_np, all_probs_np)
            entropies = calculate_predictive_entropy(all_probs_np)
            mean_entropy = float(np.mean(entropies))
        else:
            ece = 0.0
            mean_entropy = 0.0

        metrics = {
            "macro_f1": float(macro_f1),
            "precision": float(precision),
            "recall": float(recall),
            "ece": float(ece),
            "predictive_entropy": float(mean_entropy),
        }

        return metrics

    def _save_weights(self, path: Path) -> None:
        """
        Save model weights.

        Args:
            path (Path): The path to save the model weights to.
        """
        torch.save(self.state_dict(), path)

    def _load_weights(self, path: Path) -> None:
        """
        Load model weights.

        Args:
            path (Path): The path to load the model weights from.
        """
        current_device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.load_state_dict(torch.load(path, map_location=current_device))
        self.eval()
