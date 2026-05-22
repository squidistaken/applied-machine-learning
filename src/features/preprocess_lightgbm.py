import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
from skimage.feature import hog
from src.constants import DATA_DIR, LOGGER
from src.features.image_preprocessor import ImagePreprocessor


def extract_features(
    image: Image.Image, target_size: tuple[int, int]
) -> np.ndarray:
    """
    Extract features from an image.

    Args:
        image (Image.Image): The input image.
        target_size (tuple[int, int]): Downsampled size for the images.

    Returns:
        np.ndarray: An array of features.
    """
    # Downsampling is required in the feature extraction to reduce tabular
    # dimensionality.
    img_resized = image.resize(target_size)
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    stats = np.array(
        [
            np.mean(img_array),
            np.std(img_array),
            np.percentile(img_array, 25),
            np.percentile(img_array, 50),
            np.percentile(img_array, 75),
        ]
    )

    # We extract HOG (Histogram of Oriented Gradients) features instead of raw
    # pixels because it reduces dimensionality and captures structural textures.
    hog_features = hog(
        img_array,
        orientations=8,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        feature_vector=True,
    )

    return np.concatenate([stats, hog_features])


def get_feature_names(target_size: tuple[int, int]) -> list[str]:
    """
    Get the column names for the features.

    Args:
        target_size (tuple[int, int]): Downsampled size for the images.

    Returns:
        list[str]: A list of column names.
    """
    cols = ["pixel_mean", "pixel_std", "pixel_p25", "pixel_p50", "pixel_p75"]
    n_cells_row = target_size[0] // 8
    n_cells_col = target_size[1] // 8
    n_blocks_row = (n_cells_row - 2) + 1
    n_blocks_col = (n_cells_col - 2) + 1
    hog_length = n_blocks_row * n_blocks_col * 2 * 2 * 8

    cols += [f"hog_{i}" for i in range(hog_length)]

    return cols


def preprocess_split(
    split_name: str,
    classes: list[str],
    target_size: tuple[int, int],
    augment: bool = False,
) -> None:
    """
    Preprocess a split.

    Args:
        split_name (str): The split to process.
        classes (list[str]): The list of classes to process.
        target_size (tuple[int, int]): Downsampled size for the images.
        augment (bool): Whether to apply data augmentations. Defaults to
                        False.
    """
    LOGGER.info(f"Extracting features for '{split_name}' split...")

    pgm_dir = DATA_DIR / "processed" / split_name

    if not pgm_dir.is_dir():
        LOGGER.warning(
            f"Split '{split_name}' not found in processed directory. Skipping."
        )

        return

    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    images_by_class = {}

    for cls in classes:
        class_dir = pgm_dir / cls
        if not class_dir.is_dir():
            LOGGER.warning(
                f"Class '{cls}' not found in '{split_name}' split. Skipping."
            )

            continue

        image_files = list(class_dir.glob("*.pgm"))

        if not image_files:
            LOGGER.warning(f"No images found in {class_dir}. Skipping.")

            continue

        images_by_class[cls] = image_files

    if not images_by_class:
        LOGGER.warning(f"No images found to process in '{split_name}' split.")

        return

    # NOTE: The labelling is as follows:
    # 0 = BACTERIA
    # 1 = NORMAL
    # 2 = VIRUS
    class_counts = {cls: len(imgs) for cls, imgs in images_by_class.items()}
    max_count = max(class_counts.values())

    base_features_list = []
    base_labels_list = []
    aug_features_list = []
    aug_labels_list = []

    for cls, imgs in images_by_class.items():
        cls_idx = class_to_idx[cls]
        count = len(imgs)

        LOGGER.info(
            f"Extracting features for {count} base images in {pgm_dir / cls}"
        )

        for img_path in tqdm(imgs, desc=f"Extracting {cls} in {split_name}"):
            try:
                img = Image.open(img_path).convert("L")
                feats = extract_features(img, target_size)
                base_features_list.append(feats)
                base_labels_list.append(cls_idx)

            except Exception as e:
                LOGGER.error(f"Failed to extract features from {img_path}: {e}")

        if augment and count < max_count:
            num_to_add = max_count - count
            augmentations = transforms.Compose(
                ImagePreprocessor.get_augmentations()
            )

            LOGGER.info(
                f"Augmenting '{cls}' offline to match majority class (+{num_to_add} samples)"
            )

            for i in tqdm(
                range(num_to_add), desc=f"Augmenting {cls} in {split_name}"
            ):
                img_path = imgs[i % count]

                try:
                    img = Image.open(img_path).convert("L")
                    aug_img = augmentations(img)
                    feats = extract_features(aug_img, target_size)
                    aug_features_list.append(feats)
                    aug_labels_list.append(cls_idx)
                except Exception as e:
                    LOGGER.error(
                        f"Failed to augment and extract features from {img_path}: {e}"
                    )

    if not base_features_list:
        LOGGER.warning(f"No features were extracted for '{split_name}' split.")

        return

    # Due to the risk of data leakage from augmenting our data offline,
    # we create two versions of the splits: One with and one without data
    # augmentation. That way, we can run cross-validation safely.
    LOGGER.info("Converting base features to DataFrame...")
    X_base = np.vstack(base_features_list)
    y_base = np.array(base_labels_list)

    df_base = pd.DataFrame(X_base, columns=get_feature_names(target_size))
    df_base["label"] = y_base

    cv_dir = DATA_DIR / "features"
    cv_dir.mkdir(parents=True, exist_ok=True)
    cv_path = cv_dir / f"{split_name}.csv"

    LOGGER.info(f"Saving features to {cv_path}...")
    df_base.to_csv(cv_path, index=False)

    LOGGER.info("Converting finalised features to DataFrame...")
    if augment and aug_features_list:
        X_final = np.vstack(base_features_list + aug_features_list)
        y_final = np.array(base_labels_list + aug_labels_list)

    else:
        X_final = X_base
        y_final = y_base

    df_final = pd.DataFrame(X_final, columns=get_feature_names(target_size))
    df_final["label"] = y_final

    final_dir = DATA_DIR / "processed"
    final_dir.mkdir(parents=True, exist_ok=True)
    final_path = final_dir / f"{split_name}.csv"

    LOGGER.info(f"Saving finalized features to {final_path}...")
    df_final.to_csv(final_path, index=False)

    LOGGER.info(f"'{split_name}' split processing complete.")


def preprocess_data(target_size: tuple[int, int] = (64, 64)) -> None:
    """
    Preprocess the raw data.

    Args:
        target_size (tuple[int, int]): Downsampled size for the images. Defaults
                                       to (64, 64).
    """
    processed_dir = DATA_DIR / "processed"

    if not processed_dir.is_dir():
        LOGGER.error(f"Processed directory not found: {processed_dir}.")
        LOGGER.error("Please run the PyTorch preprocessing pipeline first.")

        return

    train_dir = processed_dir / "train"

    if not train_dir.is_dir():
        LOGGER.error(f"Training directory not found in {processed_dir}.")

        return

    classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])

    if not classes:
        LOGGER.error(f"No class subdirectories found in {train_dir}.")

        return

    splits = ["train", "test"]

    for split_name in splits:
        augment = True if split_name == "train" else False
        preprocess_split(split_name, classes, target_size, augment=augment)

    LOGGER.info("LightGBM dataset preprocessing complete.")


def main() -> None:
    """Run the script."""
    preprocess_data()


if __name__ == "__main__":
    main()
