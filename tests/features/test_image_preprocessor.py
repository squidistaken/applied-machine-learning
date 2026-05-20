import pytest
from unittest.mock import patch, MagicMock
import numpy as np

from src.features.image_preprocessor import ImagePreprocessor


@pytest.fixture
def preprocessor() -> ImagePreprocessor:
    """Set up an ImagePreprocessor instance for testing.

    Returns:
        ImagePreprocessor: A preprocessor instance with a target size of
                           (128, 128).
    """
    return ImagePreprocessor(target_size=(128, 128))


@pytest.fixture
def dummy_image() -> np.ndarray:
    """Set up a dummy grayscale image as an NumPy array.

    Returns:
        np.ndarray: A random uint8 image of shape (200, 200).
    """
    return np.random.randint(0, 256, (200, 200), dtype=np.uint8)


def test_init(preprocessor: ImagePreprocessor) -> None:
    """Test the initialisation of the ImagePreprocessor class.

    Args:
        preprocessor (ImagePreprocessor): The preprocessor instance.
    """
    assert preprocessor.target_size == (128, 128)
    preprocessor_default = ImagePreprocessor()
    assert preprocessor_default.target_size == (224, 224)


@patch("src.features.image_preprocessor.cv2")
def test_extract_lung_region(
    mock_cv2: MagicMock,
    preprocessor: ImagePreprocessor,
    dummy_image: np.ndarray,
) -> None:
    """Test the lung region extraction logic.

    Args:
        mock_cv2 (MagicMock): The mocked OpenCV library.
        preprocessor (ImagePreprocessor): The preprocessor instance.
        dummy_image (np.ndarray): The dummy image array.
    """
    mock_cv2.threshold.return_value = (0, dummy_image)
    mock_cv2.morphologyEx.return_value = dummy_image
    mock_cv2.THRESH_BINARY = 0
    mock_cv2.THRESH_OTSU = 0
    mock_cv2.MORPH_OPEN = 0
    mock_cv2.RETR_EXTERNAL = 0
    mock_cv2.CHAIN_APPROX_SIMPLE = 0

    contour1 = np.array([[[0, 0]], [[0, 50]], [[50, 50]], [[50, 0]]])
    contour2 = np.array(
        [[[100, 100]], [[100, 150]], [[150, 150]], [[150, 100]]]
    )

    with patch(
        "src.features.image_preprocessor.cv2.contourArea",
        side_effect=[2500, 2500],
    ):
        with patch(
            "src.features.image_preprocessor.imutils.grab_contours",
            return_value=[contour1, contour2],
        ):
            mock_cv2.findContours.return_value = ([contour1, contour2], None)
            mock_cv2.boundingRect.side_effect = [
                (0, 0, 50, 50),
                (100, 100, 50, 50),
            ]

            cropped_image = preprocessor.extract_lung_region(dummy_image)

            # Heuristic expects min/max of the two largest contours
            assert cropped_image.shape[0] == 150
            assert cropped_image.shape[1] == 150


@patch("src.features.image_preprocessor.cv2.resize")
def test_resize(
    mock_resize: MagicMock,
    preprocessor: ImagePreprocessor,
    dummy_image: np.ndarray,
) -> None:
    """Test that the resize method correctly calls OpenCV resize.

    Args:
        mock_resize (MagicMock): The mocked OpenCV resize function.
        preprocessor (ImagePreprocessor): The preprocessor instance.
        dummy_image (np.ndarray): The dummy image array.
    """
    mock_resize.return_value = np.zeros((128, 128))
    resized_image = preprocessor.resize(dummy_image)

    mock_resize.assert_called_once()
    assert resized_image.shape == (128, 128)


@patch("src.features.image_preprocessor.cv2.createCLAHE")
def test_apply_clahe(
    mock_create_clahe: MagicMock,
    preprocessor: ImagePreprocessor,
    dummy_image: np.ndarray,
) -> None:
    """Test the application of CLAHE on a dummy image.

    Args:
        mock_create_clahe (MagicMock): The mocked CLAHE generator.
        preprocessor (ImagePreprocessor): The preprocessor instance.
        dummy_image (np.ndarray): The dummy image array.
    """
    mock_clahe_instance = MagicMock()
    mock_clahe_instance.apply.return_value = dummy_image
    mock_create_clahe.return_value = mock_clahe_instance

    clahe_image = preprocessor.apply_clahe(dummy_image)

    mock_create_clahe.assert_called_with(
        clipLimit=preprocessor.clahe_clip_limit,
        tileGridSize=preprocessor.clahe_tile_grid_size,
    )
    mock_clahe_instance.apply.assert_called_with(dummy_image)
    assert clahe_image is not None


@patch("src.features.image_preprocessor.cv2.imread")
@patch.object(ImagePreprocessor, "extract_lung_region")
@patch.object(ImagePreprocessor, "resize")
@patch.object(ImagePreprocessor, "apply_clahe")
def test_run(
    mock_apply_clahe: MagicMock,
    mock_resize: MagicMock,
    mock_extract_lung_region: MagicMock,
    mock_imread: MagicMock,
    preprocessor: ImagePreprocessor,
    dummy_image: np.ndarray,
) -> None:
    """Test the complete run pipeline of the preprocessor.

    Args:
        mock_apply_clahe (MagicMock): The mocked CLAHE method.
        mock_resize (MagicMock): The mocked resize method.
        mock_extract_lung_region (MagicMock): The mocked lung extraction
                                              method.
        mock_imread (MagicMock): The mocked OpenCV file reader.
        preprocessor (ImagePreprocessor): The preprocessor instance.
        dummy_image (np.ndarray): The dummy image array.
    """
    mock_imread.return_value = dummy_image
    mock_extract_lung_region.return_value = dummy_image
    mock_resize.return_value = dummy_image
    mock_apply_clahe.return_value = dummy_image

    result = preprocessor.run("path/to/image.jpg")

    mock_imread.assert_called_with("path/to/image.jpg", 0)
    assert result is not None


@patch("src.features.image_preprocessor.Image.fromarray")
def test_save_image(mock_fromarray: MagicMock, dummy_image: np.ndarray) -> None:
    """Test that save_image correctly interfaces with PIL.

    Args:
        mock_fromarray (MagicMock): The mocked PIL Image factory.
        dummy_image (np.ndarray): The dummy image array.
    """
    mock_pil_image = MagicMock()
    mock_fromarray.return_value = mock_pil_image

    ImagePreprocessor.save_image(dummy_image, "output.pgm", format="pgm")

    mock_fromarray.assert_called_once_with(dummy_image)
    mock_pil_image.save.assert_called_with("output.pgm", format="pgm")
