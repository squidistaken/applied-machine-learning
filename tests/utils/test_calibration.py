from pathlib import Path

import numpy as np

from src.utils.calibration import TemperatureScaler, probs_to_logits


def test_transform_scales_logits() -> None:
    """Test that transform divides the logits by the temperature."""
    scaler = TemperatureScaler(temperature=2.0)
    logits = np.array([[2.0, 4.0]])

    scaled = scaler.transform(logits)

    assert np.allclose(scaled, np.array([[1.0, 2.0]]))


def test_predict_proba_sums_to_one() -> None:
    """Test that calibrated probabilities form a valid distribution."""
    scaler = TemperatureScaler(temperature=1.5)
    logits = np.array([[2.0, 1.0, 0.0], [0.0, 0.0, 5.0]])

    probs = scaler.predict_proba(logits)

    assert np.allclose(probs.sum(axis=1), np.ones(2))
    assert (probs >= 0.0).all()


def test_predict_proba_matches_softmax_at_unit_temperature() -> None:
    """Test that a unit temperature reduces to a plain softmax."""
    scaler = TemperatureScaler(temperature=1.0)
    logits = np.array([[1.0, 2.0, 3.0]])

    probs = scaler.predict_proba(logits)
    expected = np.exp(logits) / np.sum(np.exp(logits))

    assert np.allclose(probs, expected)


def test_fit_reduces_nll() -> None:
    """Test that fitting yields a positive temperature on overconfident
    logits."""
    rng = np.random.default_rng(0)
    logits = rng.normal(scale=5.0, size=(50, 2))
    labels = (logits[:, 1] > logits[:, 0]).astype(int)

    scaler = TemperatureScaler()
    returned = scaler.fit(logits, labels)

    assert returned is scaler
    assert scaler.temperature > 0.0


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    """Test that a fitted temperature survives a save and load cycle.

    Args:
        tmp_path (Path): The temporary directory path provided by pytest.
    """
    scaler = TemperatureScaler(temperature=2.5)
    file_path = tmp_path / "nested" / "temperature.json"

    scaler.save(file_path)
    loaded = TemperatureScaler.load(file_path)

    assert loaded.temperature == 2.5


def test_probs_to_logits_inverts_softmax() -> None:
    """Test that pseudo-logits recover the original probabilities under a
    softmax."""
    probs = np.array([[0.2, 0.3, 0.5]])

    logits = probs_to_logits(probs)
    recovered = np.exp(logits) / np.sum(np.exp(logits))

    assert np.allclose(recovered, probs, atol=1e-6)


def test_probs_to_logits_clips_zero() -> None:
    """Test that a zero probability is clipped to avoid a negative infinity."""
    logits = probs_to_logits(np.array([[1.0, 0.0]]))

    assert np.isfinite(logits).all()
