import cv2
import numpy as np
from PIL import Image
import imutils
from src.constants import LOGGER


class ImagePreprocessor:
    """Class to handle image preprocessing."""

    def __init__(
        self,
        target_size: tuple[int, int] = (224, 224),
        morph_kernel_size: tuple[int, int] = (5, 5),
        morph_iterations: int = 2,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid_size: tuple[int, int] = (8, 8),
    ) -> None:
        """
        Initialise the class.

        Args:
            target_size (tuple[int, int]): The target size for the images
                                           (width, height). Defaults to
                                           (224, 224).
            morph_kernel_size (tuple[int, int]): The kernel size for the
                                                 morphological operations.
                                                 Defaults to (5, 5).
            morph_iterations (int): The number of iterations for morphological
                                    operations. Defaults to 2.
            clahe_clip_limit (float): The clip limit for CLAHE. Defaults to
                                      2.0.
            clahe_tile_grid_size (tuple[int, int]): The tile grid size for
                                                    CLAHE. Defaults to (8, 8).
        """
        self.target_size = target_size
        self.morph_kernel_size = morph_kernel_size
        self.morph_iterations = morph_iterations
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid_size = clahe_tile_grid_size

    def extract_lung_region(self, image: np.ndarray) -> np.ndarray:
        """
        Extract the lung region for a chest X-ray image using countour
        detection. This heuristic approach identifies the largest contours
        (e.g. lungs) and crops the image to their bounding box.

        Args:
            image (np.ndarray): The input grayscale image.

        Returns:
            np.ndarray: A cropped image containing the lung region.
        """
        # Otsu's method automatically determines the optimal threshold value.
        _, thresh = cv2.threshold(
            image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Erode and dilate to remove small noise.
        kernel = np.ones(self.morph_kernel_size, np.uint8)
        morphed = cv2.morphologyEx(
            thresh, cv2.MORPH_OPEN, kernel, iterations=self.morph_iterations
        )
        cnts = cv2.findContours(
            morphed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cnts = imutils.grab_contours(cnts)

        if not cnts:
            # If there are no countours, the image is fine as is.
            return image

        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

        # We assumme the two largest countours are the lungs, so we create a
        # bounding box to encompass them.
        if len(cnts) >= 2:
            x1, y1, w1, h1 = cv2.boundingRect(cnts[0])
            x2, y2, w2, h2 = cv2.boundingRect(cnts[1])
            x = min(x1, x2)
            y = min(y1, y2)
            w = max(x1 + w1, x2 + w2) - x
            h = max(y1 + h1, y2 + h2) - y
        else:
            # If only one significant contour is found, we use its bounding box
            # as we can assume the countour selected both lungs as one.
            x, y, w, h = cv2.boundingRect(cnts[0])

        return image[y : y + h, x : x + w]

    def resize(self, image: np.ndarray) -> np.ndarray:
        """
        Resize an image to the target size.

        Args:
            image (np.ndarray): The input image as a NumPy array.

        Returns:
            np.ndarray: A resized image.
        """
        return cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)

    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to a
        grayscale image.

        Args:
            image (np.ndarray): The input grayscale image.

        Returns:
            np.ndarray: An image with CLAHE applied.
        """
        # CLAHE should be applied on grayscale images.
        if len(image.shape) > 2 and image.shape[2] > 1:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=self.clahe_tile_grid_size,
        )

        return clahe.apply(image)

    def run(self, image_path: str) -> np.ndarray:
        """
        Run the image preprocessing pipeline.

        Args:
            image_path (str): The path to the image file.

        Returns:
            np.ndarray: A preprocessed 8-bit image.
        """
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            LOGGER.error(f"Failed to load image from {image_path}")

            raise ValueError(f"Failed to load image from {image_path}")

        cropped_image = self.extract_lung_region(image)
        resized_image = self.resize(cropped_image)
        clahe_image = self.apply_clahe(resized_image)

        return clahe_image

    @staticmethod
    def save_image(image: np.ndarray, path: str, format: str = "pgm") -> None:
        """
        Save the preprocessed image to the specified path.

        Args:
            image (np.ndarray): The preprocessed 8-bit image.
            path (str): The destination file path.
            format (str): The format to save the image in.
        """
        pil_image = Image.fromarray(image)

        pil_image.save(path, format=format)
