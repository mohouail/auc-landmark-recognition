"""Keypoints (step 2) and global features (method B).  Owner: TEAMMATE"""
import cv2
import numpy as np
from sklearn.decomposition import PCA

from src import config


def _to_gray(img):
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# ---------- Step 2: keypoints ----------

def detect_harris(gray):
    """Harris corners.

    Returns:
        list of cv2.KeyPoint
    """
    gray = _to_gray(gray)
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=config.HARRIS_MAX_CORNERS,
        qualityLevel=config.HARRIS_QUALITY,
        minDistance=config.HARRIS_MIN_DISTANCE,
        blockSize=config.HARRIS_BLOCK_SIZE,
        useHarrisDetector=True,
        k=config.HARRIS_K,
    )
    if corners is None:
        return []
    size = float(config.HARRIS_BLOCK_SIZE * 2)
    return [cv2.KeyPoint(float(x), float(y), size) for x, y in corners.reshape(-1, 2)]


_sift = None


def detect_sift(gray):
    """SIFT keypoints and descriptors. Used by both of us.

    Returns:
        keypoints: list of cv2.KeyPoint
        descriptors: np.ndarray (N, 128) float32, or None if no keypoints
    """
    global _sift
    if _sift is None:
        _sift = cv2.SIFT_create()
    keypoints, descriptors = _sift.detectAndCompute(_to_gray(gray), None)
    return list(keypoints), descriptors


def sift_keypoints(gray):
    """Just the SIFT keypoints, so it can be passed to repeatability()."""
    return detect_sift(gray)[0]


def repeatability(img1, img2, H, detector, tol=None):
    """Share of keypoints in img1 found again in img2.

    Each keypoint of img1 is moved into img2 with H. Only points that land
    inside img2 count. A point is "found again" if img2 has a keypoint of
    its own within tol pixels of where it landed.

    Args:
        H: 3x3 homography mapping img1 -> img2 (from matching.match_good)
        detector: detect_harris or a function returning SIFT keypoints
                  (detect_sift also works, its descriptors are ignored)
        tol: px distance to count a keypoint as found again
             (config.REPEAT_TOL by default)
    Returns:
        float in [0, 1]
    """
    tol = config.REPEAT_TOL if tol is None else tol
    kp1, kp2 = detector(_to_gray(img1)), detector(_to_gray(img2))
    if isinstance(kp1, tuple):          # detect_sift returns (kp, desc)
        kp1, kp2 = kp1[0], kp2[0]
    if len(kp1) == 0 or len(kp2) == 0:
        return 0.0

    pts1 = np.float32([k.pt for k in kp1]).reshape(-1, 1, 2)
    pts2 = np.float32([k.pt for k in kp2])
    proj = cv2.perspectiveTransform(pts1, np.asarray(H, dtype=np.float64)).reshape(-1, 2)

    h2, w2 = img2.shape[:2]
    inside = ((proj[:, 0] >= 0) & (proj[:, 0] < w2) &
              (proj[:, 1] >= 0) & (proj[:, 1] < h2))
    proj = proj[inside]
    if len(proj) == 0:
        return 0.0

    found = 0
    for start in range(0, len(proj), 500):      # chunks keep memory small
        chunk = proj[start:start + 500]
        d2 = ((chunk[:, None, :] - pts2[None, :, :]) ** 2).sum(axis=2)
        found += int((d2.min(axis=1) <= tol ** 2).sum())
    return found / len(proj)


# ---------- Method B: one global feature per image ----------

def color_histogram(img):
    """Normalized HSV color histogram with config.HIST_BINS bins.

    Returns:
        np.ndarray (n_bins,) float32, sums to 1
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, list(config.HIST_BINS),
                        [0, 180, 0, 256, 0, 256])
    hist = hist.flatten().astype(np.float32)
    return hist / max(hist.sum(), 1e-8)


def tiny_image(img):
    """Grayscale image resized to TINY_SIZE x TINY_SIZE and flattened.

    It is also made zero-mean and unit-length, so a photo taken in brighter
    or darker light gives nearly the same feature.

    Returns:
        np.ndarray (TINY_SIZE * TINY_SIZE,) float32
    """
    small = cv2.resize(_to_gray(img), (config.TINY_SIZE, config.TINY_SIZE),
                       interpolation=cv2.INTER_AREA)
    v = small.astype(np.float32).flatten()
    v -= v.mean()
    return v / max(np.linalg.norm(v), 1e-8)


def fit_pca(X_train):
    """Fit PCA with config.PCA_COMPONENTS on training features only.

    Returns:
        fitted sklearn.decomposition.PCA
    """
    X_train = np.asarray(X_train)
    n = min(config.PCA_COMPONENTS, X_train.shape[0], X_train.shape[1])
    return PCA(n_components=n, random_state=config.SEED).fit(X_train)


def features_for(images, kind):
    """Stack one feature per image. kind: "hist" or "tiny"."""
    fn = {"hist": color_histogram, "tiny": tiny_image}[kind]
    return np.stack([fn(img) for img in images])
