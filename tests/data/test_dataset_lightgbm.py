import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
from pathlib import Path
import tempfile
from typing import Iterator, Tuple

from src.data.dataset_lightgbm import ChestXRayDatasetLightGBM


@pytest.fixture
def dummy_lgbm_data() -> Iterator[Tuple[Path, pd.DataFrame]]:
    """Set up a temporary directory and dummy data for LightGBM dataset tests.

    Yields:
        Iterator[Tuple[Path, pd.DataFrame]]: An iterator containing a tuple
                                             containing the temporary directory
                                             and dummy data.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        root_path = Path(temp_dir)
        features_dir = root_path / "features"
        features_dir.mkdir()

        dummy_df = pd.DataFrame(
            {"feature1": [1, 2, 3], "feature2": [4, 5, 6], "label": [0, 1, 0]}
        )

        with patch("src.data.dataset_lightgbm.DATA_DIR", root_path):
            yield features_dir, dummy_df


@patch("src.data.dataset_lightgbm.pd.read_csv")
def test_init_and_create_dataset(
    mock_read_csv: MagicMock, dummy_lgbm_data: Tuple[Path, pd.DataFrame]
) -> None:
    """Test the initialisation and creation of the dataset.

    Args:
        mock_read_csv (MagicMock): The mocked read_csv function.
        dummy_lgbm_data (Tuple[Path, pd.DataFrame]): The dummy_lgbm_data
                                                     instance.
    """
    features_dir, dummy_df = dummy_lgbm_data
    mock_read_csv.return_value = dummy_df
    csv_path = features_dir / "train.csv"
    csv_path.touch()
    dataset = ChestXRayDatasetLightGBM(split="train", augmented=False)

    mock_read_csv.assert_called_with(csv_path)

    pd.testing.assert_frame_equal(dataset.X, dummy_df.drop(columns=["label"]))
    pd.testing.assert_series_equal(dataset.y, dummy_df["label"])

    assert len(dataset) == 3
    assert dataset.split == "train"
    assert dataset.augmented is False


@patch("src.data.dataset_lightgbm.pd.read_csv")
def test_getitem(
    mock_read_csv: MagicMock, dummy_lgbm_data: tuple[Path, pd.DataFrame]
) -> None:
    """Test the __getitem__ method of the dataset.

    Args:
        mock_read_csv (MagicMock): The mocked read_csv function.
        dummy_lgbm_data (Tuple[Path, pd.DataFrame]): The dummy_lgbm_data
                                                     instance.
    """
    features_dir, dummy_df = dummy_lgbm_data
    mock_read_csv.return_value = dummy_df

    (features_dir / "train.csv").touch()

    dataset = ChestXRayDatasetLightGBM(split="train", augmented=False)
    features, label = dataset[1]

    assert label == 1
    assert features["feature1"] == 2
    assert features["feature2"] == 5


@patch("src.data.dataset_lightgbm.pd.read_csv")
def test_get_data(
    mock_read_csv: MagicMock, dummy_lgbm_data: tuple[Path, pd.DataFrame]
) -> None:
    """Test the get_data method of the dataset.

    Args:
        mock_read_csv (MagicMock): The mocked read_csv function.
        dummy_lgbm_data (Tuple[Path, pd.DataFrame]): The dummy_lgbm_data
                                                     instance.
    """
    features_dir, dummy_df = dummy_lgbm_data
    mock_read_csv.return_value = dummy_df

    (features_dir / "train.csv").touch()

    dataset = ChestXRayDatasetLightGBM(split="train", augmented=False)
    X, y = dataset.get_data()

    pd.testing.assert_frame_equal(X, dummy_df.drop(columns=["label"]))
    pd.testing.assert_series_equal(y, dummy_df["label"])


@patch("src.data.dataset_lightgbm.lgb.Dataset")
@patch("src.data.dataset_lightgbm.pd.read_csv")
def test_get_lgb_dataset(
    mock_read_csv: MagicMock,
    mock_lgb_dataset: MagicMock,
    dummy_lgbm_data: tuple[Path, pd.DataFrame],
) -> None:
    """Test the get_lgb_dataset method of the dataset.

    Args:
        mock_read_csv (MagicMock): The mocked read_csv function.
        mock_lgb_dataset (MagicMock): The mocked lgb.Dataset function.
        dummy_lgbm_data (Tuple[Path, pd.DataFrame]): The dummy_lgbm_data
                                                     instance.
    """
    features_dir, dummy_df = dummy_lgbm_data
    mock_read_csv.return_value = dummy_df

    (features_dir / "train.csv").touch()

    dataset = ChestXRayDatasetLightGBM(split="train", augmented=False)

    dataset.get_lgb_dataset(free_raw_data=True)
    mock_lgb_dataset.assert_called_once()

    args, kwargs = mock_lgb_dataset.call_args

    pd.testing.assert_frame_equal(args[0], dataset.X)
    pd.testing.assert_series_equal(kwargs["label"], dataset.y)

    assert kwargs["free_raw_data"] is True
