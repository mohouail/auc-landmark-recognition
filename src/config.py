"""Shared settings. Change values here only, never hard-code them elsewhere."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
LABELS_CSV = ROOT / "data" / "labels.csv"
RESULTS_DIR = ROOT / "results"

# The 7 landmarks (from the 26 Nov shoot list). Folder names must match exactly.
LANDMARKS = [
    "pepsi_gate",
    "sports_gate",
    "seoudi",
    "waterfall",
    "library",
    "admin_building",
    "cilantro",
]
UNKNOWN = "unknown"         # label for photos outside the 7 (stretch goal)

IMAGE_LONG_SIDE = 1024      # px, long side after resizing
RATIO_TEST = 0.75           # Lowe's ratio test threshold
RANSAC_THRESHOLD = 5.0      # px, reprojection error for findHomography
MIN_INLIERS = 10            # below this, a pair counts as "no match"
HIST_BINS = (8, 8, 8)       # HSV color histogram bins
TINY_SIZE = 32              # tiny image is TINY_SIZE x TINY_SIZE grayscale
PCA_COMPONENTS = 50
KNN_K = 3
SEED = 42

# Keypoints (step 2)
HARRIS_MAX_CORNERS = 2000
HARRIS_QUALITY = 0.01       # fraction of the strongest corner response
HARRIS_MIN_DISTANCE = 5     # px between corners
HARRIS_BLOCK_SIZE = 3
HARRIS_K = 0.04
REPEAT_TOL = 3.0            # px, a keypoint counts as "found again" within this

# Method B classifiers
SVM_C = 10.0
SVM_KERNEL = "rbf"
MLP_HIDDEN = (128,)
MLP_ALPHA = 1e-3            # L2 regularization, we have few photos
MLP_MAX_ITER = 2000
