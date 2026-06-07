import numpy as np
import torch
import torch.nn as nn

from src.utils.uq_utils import (
    enable_dropout,
    calculate_predictive_entropy,
    compute_reliability_curve,
    calculate_ece,
    calculate_brier_score,
    calculate_nll,
    selective_prediction_curve,
    decompose_mc_uncertainty,
    get_mc_dropout_uncertainty,
)


class _DropoutModel(nn.Module):
    """Classifier class with a dropout layer for uncertainty testing."""

    def __init__(self) -> None:
        """Initialise the model with a linear layer and dropout."""
        super().__init__()
        self.fc = nn.Linear(4, 2)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: A logit.
        """
        return self.fc(self.dropout(x))


def test_enable_dropout() -> None:
    """Test that dropout layers are switched to train mode while others stay
    in eval mode."""
    model = _DropoutModel()
    model.eval()

    enable_dropout(model)

    assert model.dropout.training is True
    assert model.fc.training is False


def test_calculate_predictive_entropy_uniform() -> None:
    """Test that a uniform distribution yields the maximum entropy."""
    probs = np.array([[0.5, 0.5]])

    entropy = calculate_predictive_entropy(probs)

    assert np.isclose(entropy[0], np.log(2), atol=1e-6)


def test_calculate_predictive_entropy_confident() -> None:
    """Test that a confident prediction yields near-zero entropy."""
    probs = np.array([[1.0, 0.0]])

    entropy = calculate_predictive_entropy(probs)

    assert np.isclose(entropy[0], 0.0, atol=1e-6)


def test_compute_reliability_curve_structure() -> None:
    """Test that the reliability curve has consistent shapes and bin counts."""
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.3, 0.7]])
    num_bins = 5

    curve = compute_reliability_curve(y_true, y_prob, num_bins=num_bins)

    assert curve["bin_edges"].shape == (num_bins + 1,)
    assert curve["bin_confidence"].shape == (num_bins,)
    assert curve["bin_accuracy"].shape == (num_bins,)
    assert curve["bin_counts"].sum() == len(y_true)


def test_calculate_ece_perfectly_calibrated() -> None:
    """Test that perfectly confident and correct predictions give zero ECE."""
    y_true = np.array([0, 0])
    y_prob = np.array([[1.0, 0.0], [1.0, 0.0]])

    assert calculate_ece(y_true, y_prob) == 0.0


def test_calculate_ece_miscalibrated() -> None:
    """Test that a confident but wrong prediction yields the confidence gap."""
    y_true = np.array([1])
    y_prob = np.array([[0.9, 0.1]])

    assert np.isclose(calculate_ece(y_true, y_prob), 0.9, atol=1e-6)


def test_calculate_brier_score_bounds() -> None:
    """Test the Brier score for a perfect and a worst-case prediction."""
    perfect = calculate_brier_score(np.array([0]), np.array([[1.0, 0.0]]))
    worst = calculate_brier_score(np.array([0]), np.array([[0.0, 1.0]]))

    assert np.isclose(perfect, 0.0, atol=1e-6)
    assert np.isclose(worst, 2.0, atol=1e-6)


def test_calculate_nll_perfect_and_uniform() -> None:
    """Test the NLL for a perfect prediction and a uniform prediction."""
    perfect = calculate_nll(np.array([0]), np.array([[1.0, 0.0]]))
    uniform = calculate_nll(np.array([0]), np.array([[0.5, 0.5]]))

    assert np.isclose(perfect, 0.0, atol=1e-6)
    assert np.isclose(uniform, np.log(2), atol=1e-6)


def test_selective_prediction_curve_perfect() -> None:
    """Test that a perfect classifier yields zero risk and zero AURC."""
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]])
    num_points = 4

    curve = selective_prediction_curve(y_true, y_prob, num_points=num_points)

    assert np.asarray(curve["coverage"]).shape == (num_points,)
    assert np.allclose(curve["risk"], 1.0 - np.asarray(curve["accuracy"]))
    assert np.isclose(curve["aurc"], 0.0, atol=1e-6)


def test_selective_prediction_curve_ranks_confident_first() -> None:
    """Test that low coverage retains only the most confident prediction."""
    y_true = np.array([0, 1])
    y_prob = np.array([[0.99, 0.01], [0.6, 0.4]])

    curve = selective_prediction_curve(y_true, y_prob, num_points=2)

    assert np.asarray(curve["accuracy"])[0] == 1.0


def test_decompose_mc_uncertainty_deterministic() -> None:
    """Test that identical MC passes give zero epistemic uncertainty."""
    probs = torch.tensor([[0.7, 0.3]])
    mc_probs = torch.stack([probs, probs, probs], dim=0)

    uncertainty = decompose_mc_uncertainty(mc_probs)

    assert torch.isclose(
        uncertainty["epistemic"], torch.zeros(1), atol=1e-6
    ).all()
    assert torch.allclose(
        uncertainty["total"], uncertainty["aleatoric"], atol=1e-6
    )


def test_decompose_mc_uncertainty_non_negative_epistemic() -> None:
    """Test that disagreeing MC passes give non-negative epistemic
    uncertainty."""
    mc_probs = torch.tensor([[[0.9, 0.1]], [[0.1, 0.9]]])

    uncertainty = decompose_mc_uncertainty(mc_probs)

    assert (uncertainty["epistemic"] >= 0.0).all()
    assert uncertainty["epistemic"].item() > 0.0


def test_get_mc_dropout_uncertainty_shapes() -> None:
    """Test the output shapes of the MC Dropout uncertainty estimator."""
    model = _DropoutModel()
    x = torch.randn(3, 4)

    mean_probs, variance_probs, uncertainty = get_mc_dropout_uncertainty(
        model, x, num_passes=5
    )

    assert mean_probs.shape == (3, 2)
    assert variance_probs.shape == (3, 2)
    assert torch.allclose(mean_probs.sum(dim=1), torch.ones(3), atol=1e-6)
    assert uncertainty["total"].shape == (3,)
