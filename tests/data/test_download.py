import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile
from typing import Iterator, Tuple
from src.data.download import DataDownloader


@pytest.fixture
def downloader_setup() -> Iterator[Tuple[DataDownloader, Path]]:
    """Set up a temporary directory and a DataDownloader instance.

    Yields:
        Iterator[Tuple[DataDownloader, Path]]: An iterator containing a tuple
                                               containing the downloader
                                               instance and the temporary raw
                                               data path.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        raw_data_path = Path(temp_dir)
        downloader = DataDownloader(raw_data_path)
        yield downloader, raw_data_path


@patch("src.data.download.kagglehub")
def test_download_from_kaggle(
    mock_kagglehub: MagicMock, downloader_setup: Tuple[DataDownloader, Path]
) -> None:
    """Test the Kaggle download method of the DataDownloader.

    Args:
        mock_kagglehub (MagicMock): The mocked kagglehub library.
        downloader_setup (Tuple[DataDownloader, Path]): The downloader instance
                                                        and path.
    """
    downloader, raw_data_path = downloader_setup
    downloader._download_from_kaggle()

    mock_kagglehub.dataset_download.assert_called_once_with(
        handle="tolgadincer/labeled-chest-xray-images",
        output_dir=str(raw_data_path),
        force_download=True,
    )


@patch("src.data.download.shutil")
def test_flatten_directory(
    mock_shutil: MagicMock, downloader_setup: Tuple[DataDownloader, Path]
) -> None:
    """Test the flattening of the nested Kaggle directory structure.

    Args:
        mock_shutil (MagicMock): The mocked shutil for moving files.
        downloader_setup (Tuple[DataDownloader, Path]): The downloader instance
                                                        and path.
    """
    downloader, raw_data_path = downloader_setup
    nested_folder = raw_data_path / "chest_xray"
    nested_folder.mkdir()
    (nested_folder / "file1.txt").touch()
    (nested_folder / "subdir").mkdir()

    downloader._flatten_directory()

    mock_shutil.move.assert_any_call(
        str(nested_folder / "file1.txt"), str(raw_data_path / "file1.txt")
    )
    mock_shutil.move.assert_any_call(
        str(nested_folder / "subdir"), str(raw_data_path / "subdir")
    )

    mock_shutil.rmtree.assert_called_with(
        raw_data_path / ".complete", ignore_errors=True
    )


@patch("src.data.download.shutil")
def test_flatten_directory_no_nested_folder(
    mock_shutil: MagicMock, downloader_setup: Tuple[DataDownloader, Path]
) -> None:
    """Test flatten behavior when the expected nested directory is missing.

    Args:
        mock_shutil (MagicMock): The mocked shutil.
        downloader_setup (Tuple[DataDownloader, Path]): The downloader instance
                                                        and path.
    """
    downloader, _ = downloader_setup
    downloader._flatten_directory()

    mock_shutil.move.assert_not_called()


@patch("src.data.download.shutil")
def test_organise_classes(
    mock_shutil: MagicMock, downloader_setup: Tuple[DataDownloader, Path]
) -> None:
    """Test the reorganization of pneumonia images into bacteria and virus
    classes.

    Args:
        mock_shutil (MagicMock): The mocked shutil.
        downloader_setup (Tuple[DataDownloader, Path]): The downloader instance
                                                        and path.
    """
    downloader, raw_data_path = downloader_setup
    train_pneumonia_path = raw_data_path / "train" / "PNEUMONIA"
    train_pneumonia_path.mkdir(parents=True)
    (train_pneumonia_path / "bacteria_1.jpeg").touch()
    (train_pneumonia_path / "virus_1.jpeg").touch()
    (train_pneumonia_path / "unknown_1.jpeg").touch()

    downloader._organise_classes()

    train_bacteria_path = raw_data_path / "train" / "BACTERIA"
    train_virus_path = raw_data_path / "train" / "VIRUS"

    mock_shutil.move.assert_any_call(
        str(train_pneumonia_path / "bacteria_1.jpeg"),
        str(train_bacteria_path / "bacteria_1.jpeg"),
    )
    mock_shutil.move.assert_any_call(
        str(train_pneumonia_path / "virus_1.jpeg"),
        str(train_virus_path / "virus_1.jpeg"),
    )
    assert mock_shutil.move.call_count == 2


@patch.object(DataDownloader, "_download_from_kaggle")
@patch.object(DataDownloader, "_flatten_directory")
@patch.object(DataDownloader, "_organise_classes")
def test_run_force_download(
    mock_organise: MagicMock,
    mock_flatten: MagicMock,
    mock_download: MagicMock,
    downloader_setup: Tuple[DataDownloader, Path],
) -> None:
    """Test the full run pipeline when force_download is enabled.

    Args:
        mock_organise (MagicMock): The mocked _organise_classes method.
        mock_flatten (MagicMock): The mocked _flatten_directory method.
        mock_download (MagicMock): The mocked _download_from_kaggle method.
        downloader_setup (Tuple[DataDownloader, Path]): The downloader instance
                                                        and path.
    """
    downloader, _ = downloader_setup
    downloader.run(force_download=True)

    mock_download.assert_called_once()
    mock_flatten.assert_called_once()
    mock_organise.assert_called_once()
