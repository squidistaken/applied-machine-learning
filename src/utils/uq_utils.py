import numpy as np
import torch
import torch.nn as nn


def enable_dropout(model: nn.Module) -> None:
    """
    Enable dropout layers during test/inference time for Monte Carlo (MC)
    Dropout in PyTorch models.

    Args:
        model (nn.Module): The PyTorch model to modify.
    """
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()


def calculate_predictive_entropy(
    probs: np.ndarray, epsilon: float = 1e-9
) -> np.ndarray:
    """
    Calculate Shannon Entropy of predicted probabilities to measure
    total/aleatoric uncertainty.

    Args:
        probs (np.ndarray): The class probabilities.
        epsilon (float): The value to prevent log(0) numerical instability.

    Returns:
        np.ndarray: An array of shape containing the predictive entropy for
                    each sample.
    """
    return -np.sum(probs * np.log(probs + epsilon), axis=1)


def calculate_ece(
    y_true: np.ndarray, y_prob: np.ndarray, num_bins: int = 10
) -> float:
    """
    Calculate Expected Calibration Error (ECE) to measure the difference
    between predicted confidence and actual accuracy.

    Args:
        y_true (np.ndarray): The true class indices.
        y_prob (np.ndarray): The predicted class probabilities.
        num_bins (int): The number of bins to divide probabilities into.

    Returns:
        float: An ECE score.
    """
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = predictions == y_true

    ece = 0.0
    bin_boundaries = np.linspace(0, 1, num_bins + 1)

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += prop_in_bin * np.abs(avg_confidence_in_bin - accuracy_in_bin)

    return float(ece)


def get_mc_dropout_uncertainty(
    model: nn.Module, x: torch.Tensor, num_passes: int = 15
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Get Monte Carlo Dropout forward passes to estimate epistemic and aleatoric
    uncertainty.

    Args:
        model (nn.Module): The PyTorch model.
        x (torch.Tensor): The input data tensor.
        num_passes (int): The number of stochastic forward passes.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the
                                                         mean probabilities,
                                                         the variance of
                                                         probabilities, and the
                                                         Shannon entropy of the
                                                         mean probabilities.
    """
    model.eval()
    enable_dropout(model)

    mc_probs = []
    with torch.no_grad():
        for _ in range(num_passes):
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            mc_probs.append(probs)

    # We stack a new dimension: (num_passes, N, C).
    mc_probs = torch.stack(mc_probs, dim=0)
    mean_probs = torch.mean(mc_probs, dim=0)
    variance_probs = torch.var(mc_probs, dim=0)
    mean_probs_np = mean_probs.cpu().numpy()
    entropy_np = calculate_predictive_entropy(mean_probs_np)
    entropy_tensor = torch.tensor(entropy_np, device=x.device)

    return mean_probs, variance_probs, entropy_tensor
