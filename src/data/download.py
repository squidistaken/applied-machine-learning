import kagglehub
import shutil
import argparse
from pathlib import Path
from src.constants import (
    DATA_DIR,
    LOGGER,
    DEBUG,
)


class DataDownloader:
    """Class to handle downloading and organizing the dataset."""

    def __init__(self, raw_data_path: Path) -> None:
        """
        Initialise the class.

        Args:
            raw_data_path (Path): The path to the raw data directory.
        """
        self.raw_data_path = raw_data_path

    def _download_from_kaggle(self) -> None:
        """Download the dataset from Kaggle."""
        LOGGER.info("Downloading data from Kaggle...")
        kagglehub.dataset_download(
            handle="tolgadincer/labeled-chest-xray-images",
            output_dir=str(self.raw_data_path),
            force_download=True,
        )
        LOGGER.info("Data downloaded successfully.")

    def _flatten_directory(self) -> None:
        """Flatten the nested directory structure from the Kaggle download."""
        LOGGER.info("Cleaning up the folders...")

        nested_folder = self.raw_data_path / "chest_xray"

        if not nested_folder.is_dir():
            LOGGER.info(
                "No nested 'chest_xray' folder found. Skipping flattening."
            )

            return

        for item in nested_folder.iterdir():
            dest = self.raw_data_path / item.name

            if dest.exists():
                shutil.rmtree(dest) if dest.is_dir() else dest.unlink()

            shutil.move(str(item), str(dest))

        nested_folder.rmdir()

        # Force delete any metadata that might be created by KaggleHub.
        metadata_path = self.raw_data_path / ".complete"

        shutil.rmtree(metadata_path, ignore_errors=True)
        LOGGER.info("Folder structure cleaned.")

    def _organise_classes(self) -> None:
        """Organises images from 'PNEUMONIA' into 'BACTERIA' and 'VIRUS'."""
        LOGGER.info(
            "Organizing raw data into three classes: NORMAL, BACTERIA, VIRUS..."
        )

        for split in ["train", "test"]:
            split_path = self.raw_data_path / split
            pneumonia_path = split_path / "PNEUMONIA"

            if not pneumonia_path.is_dir():
                LOGGER.info(
                    f"'PNEUMONIA' folder not found in '{split}' set. Skipping reorganisation."
                )

                continue

            bacteria_path = split_path / "BACTERIA"
            virus_path = split_path / "VIRUS"

            bacteria_path.mkdir(exist_ok=True)
            virus_path.mkdir(exist_ok=True)

            LOGGER.info(f"Processing images in: {pneumonia_path}")

            for image_path in pneumonia_path.glob("*.jpeg"):
                if "bacteria" in image_path.name.lower():
                    shutil.move(
                        str(image_path), str(bacteria_path / image_path.name)
                    )

                elif "virus" in image_path.name.lower():
                    shutil.move(
                        str(image_path), str(virus_path / image_path.name)
                    )

            try:
                pneumonia_path.rmdir()
                LOGGER.info(f"Removed empty directory: {pneumonia_path}")

            except OSError:
                LOGGER.warning(
                    f"Could not remove {pneumonia_path} as it might not be empty."
                )

        LOGGER.info("Successfully organised raw data.")

    def run(self, force_download: bool = False) -> None:
        """
        Run the full download and organisation pipeline.

        Args:
            force_download (bool): Whether to force download the data even if
                                   it already exists. Defaults to False.
        """
        if DEBUG:
            LOGGER.debug(f"Force Download: {force_download}")

        if (
            not force_download
            and (self.raw_data_path / "train").exists()
            and (self.raw_data_path / "test").exists()
        ):
            LOGGER.info("Data already exists. Skipping download.")

            return

        self._download_from_kaggle()
        self._flatten_directory()
        self._organise_classes()

        LOGGER.info("Data downloaded and prepared successfully.")


def main() -> None:
    """Run the script."""
    parser = argparse.ArgumentParser(
        description="Download and clean Kaggle dataset."
    )

    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Force download the data even if it exists",
    )

    args = parser.parse_args()

    raw_data_path = DATA_DIR / "raw"
    downloader = DataDownloader(raw_data_path)

    downloader.run(force_download=args.force)


if __name__ == "__main__":
    main()
