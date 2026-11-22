"""Loading, resizing and splitting the dataset.  Owner: TEAMMATE"""
import numpy as np


def resize_all():
    """Resize every photo in data/raw/ so its long side is IMAGE_LONG_SIDE,
    and save it to data/processed/ with the same folder and file name."""
    raise NotImplementedError


def load_labels():
    """Read data/labels.csv.

    Returns:
        pandas.DataFrame with columns
        filename, landmark, date, split, distance, angle, lighting
    """
    raise NotImplementedError


def load_image(filename, gray=False):
    """Load one processed image by file name.

    Returns:
        np.ndarray, BGR (H, W, 3) uint8, or grayscale (H, W) if gray=True
    """
    raise NotImplementedError


def load_dataset(split, gray=False):
    """Load all images of one split.

    Args:
        split: "train", "test" or "pano"
    Returns:
        images: list of np.ndarray
        labels: list of str (landmark names from config.LANDMARKS)
        filenames: list of str
    """
    raise NotImplementedError
