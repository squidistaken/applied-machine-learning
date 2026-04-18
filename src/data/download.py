import kagglehub
import shutil
from pathlib import Path
import argparse


def download_data(force_download: bool = False) -> None:
    """Download data from Kaggle.

    Args:
        force_download (bool): Whether to force download the data even if it
                               already exists. Defaults to False.
    """
    path = Path("data/raw")

    if not force_download and (
        (path / "train").exists() and (path / "test").exists()
    ):
        return

    kagglehub.dataset_download(
        handle="tolgadincer/labeled-chest-xray-images",
        output_dir=str(path),
        force_download=True,
    )

    # Flatten the folder.
    nested_folder = path / "chest_xray"

    for item in nested_folder.iterdir():
        dest = path / item.name

        if dest.exists():
            shutil.rmtree(dest) if dest.is_dir() else dest.unlink()

        shutil.move(str(item), str(dest))

    nested_folder.rmdir()

    # Force delete metadata.
    metadata_path = path / ".complete"
    shutil.rmtree(metadata_path, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and clean Kaggle dataset."
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force download the data even if it exists",
    )

    args = parser.parse_args()

    download_data(force_download=args.force)
