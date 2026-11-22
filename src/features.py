"""Keypoints (step 2) and global features (method B).  Owner: TEAMMATE"""
import numpy as np


# ---------- Step 2: keypoints ----------

def detect_harris(gray):
    """Harris corners.

    Returns:
        list of cv2.KeyPoint
    """
    raise NotImplementedError


def detect_sift(gray):
    """SIFT keypoints and descriptors. Used by both of us.

    Returns:
        keypoints: list of cv2.KeyPoint
        descriptors: np.ndarray (N, 128) float32, or None if no keypoints
    """
    raise NotImplementedError


def repeatability(img1, img2, H, detector, tol=3.0):
    """Share of keypoints in img1 found again in img2.

    Args:
        H: 3x3 homography mapping img1 -> img2 (from matching.match_good)
        detector: detect_harris or a function returning SIFT keypoints
        tol: px distance to count a keypoint as found again
    Returns:
        float in [0, 1]
    """
    raise NotImplementedError


# ---------- Method B: one global feature per image ----------

def color_histogram(img):
    """Normalized HSV color histogram with config.HIST_BINS bins.

    Returns:
        np.ndarray (n_bins,) float32
    """
    raise NotImplementedError


def tiny_image(img):
    """Grayscale image resized to TINY_SIZE x TINY_SIZE and flattened.

    Returns:
        np.ndarray (TINY_SIZE * TINY_SIZE,) float32
    """
    raise NotImplementedError


def fit_pca(X_train):
    """Fit PCA with config.PCA_COMPONENTS on training features only.

    Returns:
        fitted sklearn.decomposition.PCA
    """
    raise NotImplementedError
