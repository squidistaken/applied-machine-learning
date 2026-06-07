from typing import Callable, Optional
from tqdm import tqdm
from src.constants import DATA_DIR, LOGGER
from src.features.image_preprocessor import ImagePreprocessor


def _count_raw_images(classes: list[str]) -> int:
    """Count all raw images across the train and test splits.

    Args:
        classes (list[str]): The list of classes.

    Returns:
        int: A count of the total number of images.
    """
    total = 0
    raw_dir = DATA_DIR / "raw"
    for split_name in ["train", "test"]:
        for cls in classes:
            class_dir = raw_dir / split_name / cls
            if class_dir.is_dir():
                total += len(list(class_dir.glob("*.jpeg")))
    return total


def preprocess_split(
    split_name: str,
    classes: list[str],
    preprocessor: ImagePreprocessor,
    on_image: Optional[Callable[[], None]] = None,
) -> None:
    """
    Preprocess a split.

    Args:
        split_name (str): The split to process.
        classes (list[str]): The list of classes to process.
        preprocessor (ImagePreprocessor): The preprocessor instance.
        on_image (Optional[Callable[[], None]]): Called once after each image
                                                 is processed.
    """
    LOGGER.info(f"Preprocessing '{split_name}' split...")

    input_dir = DATA_DIR / "raw"
    output_dir = DATA_DIR / "processed"
    input_split_dir = input_dir / split_name
    output_split_dir = output_dir / split_name

    if not input_split_dir.is_dir():
        LOGGER.warning(
            f"Split '{split_name}' not found in input directory. Skipping."
        )

        return

    for cls in classes:
        input_class_dir = input_split_dir / cls
        output_class_dir = output_split_dir / cls

        if not input_class_dir.is_dir():
            LOGGER.warning(
                f"Class '{cls}' not found in '{split_name}' split. Skipping."
            )

            continue

        output_class_dir.mkdir(parents=True, exist_ok=True)

        image_files = list(input_class_dir.glob("*.jpeg"))

        if not image_files:
            LOGGER.warning(f"No images found in {input_class_dir}. Skipping.")

            continue

        LOGGER.info(
            f"Preprocessing {len(image_files)} images in {input_class_dir}"
        )

        for image_path in tqdm(
            image_files, desc=f"Preprocessing {cls} in {split_name}"
        ):
            try:
                processed_image = preprocessor.run(str(image_path))
                output_filename = image_path.with_suffix(".pgm").name
                output_path = output_class_dir / output_filename

                preprocessor.save_image(
                    processed_image, str(output_path), format="PPM"
                )

            except Exception as e:
                LOGGER.error(f"Failed to preprocess {image_path}: {e}")

            finally:
                if on_image is not None:
                    on_image()


def preprocess_data(
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> None:
    """Preprocess the raw data.

    Args:
        progress_callback (Optional[Callable[[int, int, str], None]]): The
                                                                       progress
                                                                       callback.
    """
    input_dir = DATA_DIR / "raw"
    output_dir = DATA_DIR / "processed"
    if not input_dir.is_dir():
        LOGGER.error(f"Input directory not found: {input_dir}.")
        LOGGER.error("Please run the download script first.")

        return

    output_dir.mkdir(parents=True, exist_ok=True)

    train_dir = input_dir / "train"

    if not train_dir.is_dir():
        LOGGER.error(f"Training directory not found in {input_dir}.")

        return

    classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])

    if not classes:
        LOGGER.error(f"No class subdirectories found in {train_dir}.")

        return

    splits = ["train", "test"]
    preprocessor = ImagePreprocessor()

    total = _count_raw_images(classes) if progress_callback else 0
    counter = {"done": 0}

    def _on_image() -> None:
        counter["done"] += 1
        if progress_callback and total:
            progress_callback(
                counter["done"],
                total,
                f"Preprocessing images ({counter['done']}/{total})",
            )

    for split_name in splits:
        preprocess_split(
            split_name,
            classes,
            preprocessor,
            on_image=_on_image if progress_callback else None,
        )

    LOGGER.info("Pytorch dataset preprocessing complete.")


def main() -> None:
    """Run the script."""
    preprocess_data()


if __name__ == "__main__":
    main()
