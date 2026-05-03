from typing import Callable, Optional
from collections import Counter
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from src.constants import DATA_DIR, LOGGER, DEBUG


# NOTE: We should abstract this class to account for LightGBM, or just wrap it.
#       This is because LightGBM requires an np array as input. We are using
#       PIL images for this one.
class ChestXRayDataset(Dataset):
    """PyTorch Dataset Class for Chest X-Ray Images."""

    def __init__(
        self,
        split: str,
        augment: bool = False,
        image_size: int = 224,
        transform: Optional[Callable] = None,
    ) -> None:
        """
        Initialise the class.

        Args:
            split (str): The dataset split, e.g., 'train' or 'test'.
            augment (bool): If True, applies training augmentations. Defaults to False.
            image_size (int): The edge size of the preprocessed square images.
                              Defaults to 224.
            transform (callable, optional): Optional transform to be applied on a sample.
                                            If None, default transforms are used.
        """
        if DEBUG:
            LOGGER.debug(f"Split: {split}")
            LOGGER.debug(f"Augment: {augment}")
            LOGGER.debug(f"Image Size: {image_size}")
            LOGGER.debug(f"Transform: {transform}")

        self.split = split
        self.root_dir = DATA_DIR / "processed" / self.split
        self.image_size = image_size
        self.transform = (
            transform if transform else self.compose_transforms(augment)
        )
        self.classes = sorted(
            [d.name for d in self.root_dir.iterdir() if d.is_dir()]
        )
        self.class_to_idx = {
            cls_name: i for i, cls_name in enumerate(self.classes)
        }
        self.samples = self._create_dataset()

        if not self.samples:
            LOGGER.error(f"No images found in {self.root_dir}")

            raise RuntimeError(
                f"Found 0 images in subfolders of: {self.root_dir}"
            )

    @staticmethod
    def compose_transforms(augment: bool) -> transforms.Compose:
        """
        Apply a composition of transformations for the dataset.

        Args:
            augment (bool): Whether to augment the data.

        Returns:
            transforms.Compose: A composition of transformations.
        """
        transformations = []

        if augment:
            transformations.extend(
                [
                    transforms.RandomAffine(
                        degrees=5,
                        scale=(0.9, 1.1),
                        shear=5,
                    ),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                ]
            )

        transformations.append(transforms.ToTensor())

        return transforms.Compose(transformations)

    def _create_dataset(self) -> list[tuple[str, int]]:
        """
        Create the dataset.

        Returns:
            list[tuple[str, int]]: A list of (image_path, class_index).
        """
        images = []

        for target_class in self.classes:
            class_index = self.class_to_idx[target_class]
            class_dir = self.root_dir / target_class

            for img_path in class_dir.glob("*.pgm"):
                images.append((str(img_path), class_index))

        return images

    def __len__(self) -> int:
        """Get the number of samples in the dataset.

        Returns:
            int: A number of samples in the dataset.
        """
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Image.Image, int]:
        """Get the sample at the given index of the dataset.

        Args:
            idx (int): The index of the sample.

        Returns:
            tuple[Image.Image, int]: A sample containing the image and its
                                     corresponding target class.
        """
        img_path, target = self.samples[idx]

        try:
            image = Image.open(img_path).convert("L")

        except Exception as e:
            LOGGER.error(f"Error loading image {img_path}: {e}")

            raise ValueError(f"Error loading image {img_path}: {e}")

        if self.transform:
            image = self.transform(image)

        return image, target

    def compute_sample_weights(self) -> list[float]:
        """
        Compute weights for each sample. This is for use with a
        WeightedRandomSampler to handle class imabalance.

        Returns:
            list[float]: A list of weights for each sample.
        """
        class_counts = Counter(target for _, target in self.samples)
        total_samples = self.__len__()
        class_weights = {
            class_idx: total_samples / count
            for class_idx, count in class_counts.items()
        }
        if DEBUG:
            LOGGER.debug(f"Class weights: {class_weights}")

        sample_weights = [class_weights[target] for _, target in self.samples]

        return sample_weights
