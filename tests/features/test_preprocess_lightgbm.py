import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from PIL import Image
from pathlib import Path
import tempfile
from typing import Iterator, Tuple
from src.features.preprocess_lightgbm import (
    extract_features,
    get_feature_names,
    preprocess_split,
    preprocess_data,
)


@pytest.fixture
def lgbm_setup() -> Iterator[Tuple[Path, Path]]:
    """Set up a temporary directory and dummy data for LightGBM testing.

    Yields:
        Iterator[Tuple[Path, Path]]: An iterator containing a tuple containing
                                     the root path and the features directory.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        root_path = Path(temp_dir)
        processed_dir = root_path / "processed" / "train"
        features_dir = root_path / "features"
        (processed_dir / "NORMAL").mkdir(parents=True)
        (processed_dir / "BACTERIA").mkdir(parents=True)

        Image.new("L", (64, 64)).save(processed_dir / "NORMAL" / "img1.pgm")
        Image.new("L", (64, 64)).save(processed_dir / "BACTERIA" / "img2.pgm")
        Image.new("L", (64, 64)).save(processed_dir / "BACTERIA" / "img3.pgm")

        with patch("src.features.preprocess_lightgbm.DATA_DIR", root_path):
            yield root_path, features_dir


@patch("src.features.preprocess_lightgbm.hog")
def test_extract_features(mock_hog: MagicMock) -> None:
    """Test the extraction of statistical and HOG features.

    Args:
        mock_hog (MagicMock): The mocked HOG feature generator from skimage.
    """
    # 64x64 with 8x8 cells and 2x2 blocks yields 1568 features
    mock_hog.return_value = np.ones(1568)
    img = Image.new("L", (128, 128))
    features = extract_features(img, target_size=(64, 64))

    # 5 stats + 1568 HOG features
    assert len(features) == 1573
    mock_hog.assert_called_once()


def test_get_feature_names() -> None:
    """Test the generation of tabular column names."""
    names = get_feature_names(target_size=(64, 64))

    assert names[0] == "pixel_mean"


@patch("src.features.preprocess_lightgbm.pd.DataFrame.to_csv")
@patch("src.features.preprocess_lightgbm.extract_features")
def test_preprocess_split_train_with_augment(
    mock_extract_features: MagicMock,
    mock_to_csv: MagicMock,
    lgbm_setup: Tuple[Path, Path],
) -> None:
    """Test preprocessing a split while balancing classes via offline augmentation.

    Args:
        mock_extract_features (MagicMock): The mocked feature extraction
                                           method.
        mock_to_csv (MagicMock): The mocked pandas CSV exporter.
        lgbm_setup (Tuple[Path, Path]): The lgbm_setup instance.
    """
    root_path, features_dir = lgbm_setup
    mock_extract_features.return_value = np.random.rand(1573)

    preprocess_split("train", ["BACTERIA", "NORMAL"], (64, 64), augment=True)

    # BACTERIA (2) is majority, NORMAL (1) is minority.
    # 1 augmentation expected for NORMAL.
    # Total calls: 2 (BACTERIA) + 1 (NORMAL base) + 1 (NORMAL aug) = 4
    assert mock_extract_features.call_count == 4
    assert mock_to_csv.call_count == 2

    cv_path = features_dir / "train.csv"
    final_path = root_path / "processed" / "train.csv"

    mock_to_csv.assert_any_call(cv_path, index=False)
    mock_to_csv.assert_any_call(final_path, index=False)


@patch("src.features.preprocess_lightgbm.preprocess_split")
def test_preprocess_data(
    mock_preprocess_split: MagicMock, lgbm_setup: Tuple[Path, Path]
) -> None:
    """Test the main logic of LightGBM preprocessing.

    Args:
        mock_preprocess_split (MagicMock): The mocked split processing logic.
        lgbm_setup (Tuple[Path, Path]): The lgbm_setup instance.
    """
    root_path, _ = lgbm_setup
    (root_path / "processed" / "train" / "DUMMY").mkdir(parents=True)

    preprocess_data(target_size=(32, 32))

    assert mock_preprocess_split.call_count == 2
    args, kwargs = mock_preprocess_split.call_args_list[0]
    assert args[0] == "train"
    assert kwargs["augment"] is True
