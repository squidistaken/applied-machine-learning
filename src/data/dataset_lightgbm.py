import pandas as pd
import lightgbm as lgb
from src.constants import DATA_DIR, LOGGER, DEBUG


class ChestXRayDatasetLightGBM:
    """LightGBM Dataset Class for Chest X-Ray Images."""

    def __init__(self, split: str, augmented: bool = False) -> None:
        """
        Initialise the class.

        Args:
            split (str): The dataset split.
            augmented (bool): Whether to load the fully augmented dataset
                              from or the unaugmented dataset. Defaults to
                              False.
        """
        self.split = split
        self.augmented = augmented
        self.classes = ["BACTERIA", "NORMAL", "VIRUS"]

        folder = "processed" if augmented else "features"
        self.csv_path = DATA_DIR / folder / f"{split}.csv"

        if DEBUG:
            LOGGER.debug(f"Split: {split}")
            LOGGER.debug(f"Augmented: {augmented}")
            LOGGER.debug(f"CSV path: {self.csv_path}")

        if not self.csv_path.exists():
            LOGGER.error(f"Features file not found at {self.csv_path}")

            raise FileNotFoundError(
                f"Features missing for {split} in {self.csv_path}."
            )

        self.X, self.y = self._create_dataset()

        LOGGER.info(f"Loaded {split} dataset with shape {self.X.shape}")

    def _create_dataset(self) -> tuple[pd.DataFrame, pd.Series]:
        """
        Create the dataset by loading the CSV and separating features and labels.

        Returns:
            tuple[pd.DataFrame, pd.Series]: A tuple containing the features (X)
                                            and labels (y).
        """
        data = pd.read_csv(self.csv_path)
        X = data.drop(columns=["label"])
        y = data["label"]

        return X, y

    def __len__(self) -> int:
        """
        Get the number of samples in the dataset.

        Returns:
            int: The total number of rows/samples.
        """
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[pd.Series, int]:
        """
        Get the features and label at a specific index.

        Args:
            idx (int): The index of the sample.

        Returns:
            tuple[pd.Series, int]: A tuple containing the feature series and
                                   its corresponding target label.
        """
        features = self.X.iloc[idx]
        label = int(self.y.iloc[idx])

        return features, label

    def get_data(self) -> tuple[pd.DataFrame, pd.Series]:
        """
        Get the features and labels.

        Returns:
            tuple[pd.DataFrame, pd.Series]: A tuple containing the features
                                            (X) and labels (y).
        """
        return self.X, self.y

    def get_lgb_dataset(self, free_raw_data: bool = False) -> lgb.Dataset:
        """
        Get the data as a LightGBM Dataset object.

        Args:
            free_raw_data (bool): Whether to free memory of the raw data
                                  after constructing the Dataset.

        Returns:
            lgb.Dataset: A constructed LightGBM dataset.
        """
        return lgb.Dataset(self.X, label=self.y, free_raw_data=free_raw_data)
