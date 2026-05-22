import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile
from typing import Iterator, Tuple
from src.features.preprocess_pytorch import (
    preprocess_split,
    preprocess_data,
)


@pytest.fixture
def pytorch_setup() -> Iterator[Tuple[Path, Path]]:
    """Set up a temporary directory for PyTorch preprocessing tests.

    Yields:
        Iterator[Tuple[Path, Path]]: An iterator containing a tuple containing
                                     the raw data path  and the processed data
                                     path.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        root_path = Path(temp_dir)
        raw_dir = root_path / "raw"
        processed_dir = root_path / "processed"
        (raw_dir / "train" / "NORMAL").mkdir(parents=True)
        (raw_dir / "train" / "NORMAL" / "img1.jpeg").touch()

        with patch("src.features.preprocess_pytorch.DATA_DIR", root_path):
            yield raw_dir, processed_dir


@patch("src.features.preprocess_pytorch.ImagePreprocessor")
def test_preprocess_split(
    mock_preprocessor_cls: MagicMock, pytorch_setup: Tuple[Path, Path]
) -> None:
    """Test that a single split is correctly processed and saved as PGM.

    Args:
        mock_logger (MagicMock): The mocked application logger.
        mock_preprocessor_cls (MagicMock): The mocked ImagePreprocessor class.
        pytorch_setup (Tuple[Path, Path]): The pytorch_setup instance.
    """
    raw_dir, processed_dir = pytorch_setup
    mock_preprocessor_instance = MagicMock()
    mock_preprocessor_cls.return_value = mock_preprocessor_instance

    preprocess_split("train", ["NORMAL"], mock_preprocessor_instance)

    input_path = raw_dir / "train" / "NORMAL" / "img1.jpeg"
    output_path = processed_dir / "train" / "NORMAL" / "img1.pgm"

    mock_preprocessor_instance.run.assert_called_once_with(str(input_path))
    mock_preprocessor_instance.save_image.assert_called_once()

    args, kwargs = mock_preprocessor_instance.save_image.call_args
    assert args[1] == str(output_path)
    assert kwargs["format"] == "PPM"


@patch("src.features.preprocess_pytorch.preprocess_split")
def test_preprocess_data(
    mock_preprocess_split: MagicMock, pytorch_setup: Tuple[Path, Path]
) -> None:
    """Test the PyTorch online preprocessing.

    Args:
        mock_logger (MagicMock): The mocked application logger.
        mock_preprocess_split (MagicMock): The mocked method for split
                                           processing.
        pytorch_setup (Tuple[Path, Path]): The pytorch_setup instance.
    """
    preprocess_data()

    # It should trigger for 'train' and 'test'
    assert mock_preprocess_split.call_count == 2

    # We capture the preprocessor instance passed to the first call.
    preprocessor_instance = mock_preprocess_split.call_args_list[0][0][2]

    mock_preprocess_split.assert_any_call(
        "train", ["NORMAL"], preprocessor_instance
    )
    mock_preprocess_split.assert_any_call(
        "test", ["NORMAL"], preprocessor_instance
    )
