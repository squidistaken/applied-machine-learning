import json
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn as nn


class TemperatureScaler:
    """
    Class for temperature scaling for multiclass classifiers.
    """

    def __init__(self, temperature: float = 1.0) -> None:
        """Initialise the class.

        Args:
            temperature (float): The initial/loaded temperature. Defaults to
                                 1.0.
        """
        self.temperature = float(temperature)

    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        lr: float = 0.01,
        max_iter: int = 100,
    ) -> "TemperatureScaler":
        """Fit the temperature by minimising validation NLL.

        Args:
            logits (np.ndarray): The uncalibrated logits, shape (N, num_classes).
            labels (np.ndarray): The true class indices, shape (N,).
            lr (float): The LBFGS learning rate. Defaults to 0.01.
            max_iter (int): The maximum LBFGS iterations. Defaults to 100.

        Returns:
            TemperatureScaler: A fitted instance for chaining.
        """
        logits_t = torch.as_tensor(np.asarray(logits), dtype=torch.float32)
        labels_t = torch.as_tensor(np.asarray(labels), dtype=torch.long)

        # Optimise log(T) so that T = exp(log_t) stays strictly positive and we
        # never risk a division by zero.
        log_t = torch.zeros(1, requires_grad=True)
        optimizer = torch.optim.LBFGS([log_t], lr=lr, max_iter=max_iter)
        criterion = nn.CrossEntropyLoss()

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss = criterion(logits_t / torch.exp(log_t), labels_t)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.temperature = float(torch.exp(log_t.detach()).item())

        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature scaling to logits.

        Args:
            logits (np.ndarray): The uncalibrated logits.

        Returns:
            np.ndarray: An array of temperature-scaled logits.
        """
        return np.asarray(logits) / self.temperature

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        """Return calibrated probabilities from logits.

        Args:
            logits (np.ndarray): The uncalibrated logits.

        Returns:
            np.ndarray: An array of calibrated probabilities.
        """
        scaled = self.transform(logits)
        scaled = scaled - np.max(scaled, axis=1, keepdims=True)
        exp = np.exp(scaled)

        return exp / np.sum(exp, axis=1, keepdims=True)

    def save(self, path: Union[str, Path]) -> None:
        """Save the fitted temperature to a JSON file.

        Args:
            path (Union[str, Path]): The destination path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump({"temperature": self.temperature}, f, indent=4)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TemperatureScaler":
        """Load a fitted temperature from a JSON file.

        Args:
            path (Union[str, Path]): The source path.

        Returns:
            TemperatureScaler: A loaded scaler.
        """
        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)

        return cls(temperature=float(data["temperature"]))


def probs_to_logits(probs: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    """
    Convert a probability vector into pseudo-logits via the (clipped) log.

    Args:
        probs (np.ndarray): The predicted probabilities.
        epsilon (float): The clip value to prevent log(0).

    Returns:
        np.ndarray: An array of log-probabilities to use as logits.
    """
    return np.log(np.clip(probs, epsilon, 1.0))
