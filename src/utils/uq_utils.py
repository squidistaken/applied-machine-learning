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


def compute_reliability_curve(
    y_true: np.ndarray, y_prob: np.ndarray, num_bins: int = 10
) -> dict[str, np.ndarray]:
    """
    Bin predictions by confidence and measure the accuracy within each bin.

    Args:
        y_true (np.ndarray): The true class indices.
        y_prob (np.ndarray): The predicted class probabilities.
        num_bins (int): The number of equal-width confidence bins.

    Returns:
        dict[str, np.ndarray]: A dictionary with the bin edges and, per bin, the
                               mean confidence, mean accuracy, and sample count.
    """
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = predictions == np.asarray(y_true)

    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_confidence = np.zeros(num_bins)
    bin_accuracy = np.zeros(num_bins)
    bin_counts = np.zeros(num_bins)

    for i in range(num_bins):
        in_bin = (confidences > bin_boundaries[i]) & (
            confidences <= bin_boundaries[i + 1]
        )
        count = int(np.sum(in_bin))
        bin_counts[i] = count

        if count > 0:
            bin_confidence[i] = float(np.mean(confidences[in_bin]))
            bin_accuracy[i] = float(np.mean(accuracies[in_bin]))

    return {
        "bin_edges": bin_boundaries,
        "bin_confidence": bin_confidence,
        "bin_accuracy": bin_accuracy,
        "bin_counts": bin_counts,
    }


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
    curve = compute_reliability_curve(y_true, y_prob, num_bins)
    total = curve["bin_counts"].sum()

    if total == 0:
        return 0.0

    weights = curve["bin_counts"] / total
    gaps = np.abs(curve["bin_confidence"] - curve["bin_accuracy"])

    return float(np.sum(weights * gaps))


def calculate_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Calculate the multiclass Brier score, the mean squared error between the
    predicted probability vector and the one-hot true label.

    Args:
        y_true (np.ndarray): The true class indices.
        y_prob (np.ndarray): The predicted class probabilities.

    Returns:
        float: A mean Brier score across samples.
    """
    y_true = np.asarray(y_true).astype(int)
    one_hot = np.zeros_like(y_prob, dtype=float)
    one_hot[np.arange(len(y_true)), y_true] = 1.0

    return float(np.mean(np.sum((y_prob - one_hot) ** 2, axis=1)))


def calculate_nll(
    y_true: np.ndarray, y_prob: np.ndarray, epsilon: float = 1e-12
) -> float:
    """
    Calculate the mean negative log-likelihood (cross-entropy) of the true
    labels under the predicted probabilities.

    Args:
        y_true (np.ndarray): The true class indices.
        y_prob (np.ndarray): The predicted class probabilities.
        epsilon (float): The clip value to prevent log(0) numerical instability.

    Returns:
        float: A mean negative log-likelihood across samples.
    """
    y_true = np.asarray(y_true).astype(int)
    probs = np.clip(y_prob, epsilon, 1.0)
    true_class_probs = probs[np.arange(len(y_true)), y_true]

    return float(-np.mean(np.log(true_class_probs)))


def selective_prediction_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    uncertainty: np.ndarray | None = None,
    num_points: int = 20,
) -> dict[str, np.ndarray | float]:
    """
    Compute a selective-prediction (accuracy-rejection) curve.

    Args:
        y_true (np.ndarray): The true class indices.
        y_prob (np.ndarray): The predicted class probabilities.
        uncertainty (np.ndarray | None): Per-sample uncertainty used to rank
                                         predictions. Defaults to ``1 - max
                                         probability`` (least-confident first).
        num_points (int): The number of coverage levels to evaluate.

    Returns:
        dict[str, np.ndarray | float]: A dictionary containing the coverage,
                                       accuracy and risk (1 - accuracy) arrays,
                                       plus the scalar area under the
                                       risk-coverage curve.
    """
    y_true = np.asarray(y_true).astype(int)
    predictions = np.argmax(y_prob, axis=1)
    correct = (predictions == y_true).astype(float)

    if uncertainty is None:
        uncertainty = 1.0 - np.max(y_prob, axis=1)
    uncertainty = np.asarray(uncertainty)

    # Rank most-confident (lowest uncertainty) first so that shrinking the
    # coverage keeps only the predictions the model is surest about.
    order = np.argsort(uncertainty)
    correct_sorted = correct[order]
    n = len(y_true)

    coverages = np.linspace(1.0 / num_points, 1.0, num_points)
    cov_arr = np.zeros(num_points)
    acc_arr = np.zeros(num_points)
    risk_arr = np.zeros(num_points)

    for i, cov in enumerate(coverages):
        k = max(1, int(round(cov * n)))
        cov_arr[i] = k / n
        acc_arr[i] = float(np.mean(correct_sorted[:k]))
        risk_arr[i] = 1.0 - acc_arr[i]

    # Trapezoidal area under the risk-coverage curve.
    aurc = float(
        np.sum(np.diff(cov_arr) * (risk_arr[:-1] + risk_arr[1:]) / 2.0)
    )

    return {
        "coverage": cov_arr,
        "accuracy": acc_arr,
        "risk": risk_arr,
        "aurc": aurc,
    }


def decompose_mc_uncertainty(
    mc_probs: torch.Tensor, epsilon: float = 1e-9
) -> dict[str, torch.Tensor]:
    """
    Split Monte Carlo Dropout predictive uncertainty into its aleatoric and
    epistemic components.

    Args:
        mc_probs (torch.Tensor): The stacked per-pass probabilities.
        epsilon (float): The value to prevent log(0) numerical instability.

    Returns:
        dict[str, torch.Tensor]: A dictionary containing per-sample the total,
                                 aleatoric, and epistemic uncertainties.
    """
    mean_probs = torch.mean(mc_probs, dim=0)

    total = -torch.sum(mean_probs * torch.log(mean_probs + epsilon), dim=1)

    per_pass_entropy = -torch.sum(
        mc_probs * torch.log(mc_probs + epsilon), dim=2
    )
    aleatoric = torch.mean(per_pass_entropy, dim=0)

    epistemic = torch.clamp(total - aleatoric, min=0.0)

    return {"total": total, "aleatoric": aleatoric, "epistemic": epistemic}


def get_mc_dropout_uncertainty(
    model: nn.Module,
    x: torch.Tensor,
    num_passes: int = 15,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """
    Get Monte Carlo Dropout forward passes to estimate epistemic and aleatoric
    uncertainty.

    Args:
        model (nn.Module): The PyTorch model.
        x (torch.Tensor): The input data tensor.
        num_passes (int): The number of stochastic forward passes.
        temperature (float): The temperature-scaling factor applied to the
                             logits before the softmax of each pass. Defaults
                             to 1.0.

    Returns:
        tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]: A tuple
            containing the mean probabilities, the variance of probabilities
            across passes, and the decomposed predictive uncertainty.
    """
    model.eval()
    enable_dropout(model)

    mc_probs_list = []
    with torch.no_grad():
        for _ in range(num_passes):
            logits = model(x)
            probs = torch.softmax(logits / temperature, dim=1)
            mc_probs_list.append(probs)

    mc_probs = torch.stack(mc_probs_list, dim=0)
    mean_probs = torch.mean(mc_probs, dim=0)
    variance_probs = torch.var(mc_probs, dim=0)
    uncertainty = decompose_mc_uncertainty(mc_probs)

    return mean_probs, variance_probs, uncertainty
