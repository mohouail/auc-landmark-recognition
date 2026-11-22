"""SIFT matching with ratio test and RANSAC (step 3).  Owner: ME"""
import numpy as np


def ratio_matches(desc1, desc2):
    """Brute-force kNN matching (k=2) + Lowe's ratio test (config.RATIO_TEST).

    Returns:
        list of cv2.DMatch
    """
    raise NotImplementedError


def match_good(kp1, desc1, kp2, desc2):
    """Ratio-test matches, then RANSAC homography to drop outliers.

    Returns:
        inliers: list of cv2.DMatch (empty if fewer than 4 matches)
        H: 3x3 np.ndarray mapping image 1 -> image 2, or None
    """
    raise NotImplementedError


def count_good_matches(kp1, desc1, kp2, desc2):
    """Number of RANSAC inliers between two images (used by method A)."""
    raise NotImplementedError


def draw_matches(img1, kp1, img2, kp2, matches, title=""):
    """Side-by-side figure of the matches, returned for saving.

    Returns:
        matplotlib.figure.Figure
    """
    raise NotImplementedError
