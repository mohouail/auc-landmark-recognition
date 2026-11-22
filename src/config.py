"""Shared settings. Change values here only, never hard-code them elsewhere."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
LABELS_CSV = ROOT / "data" / "labels.csv"
RESULTS_DIR = ROOT / "results"

# TODO: fix the 7 landmark names together. Folder names must match exactly.
LANDMARKS = [
    "main_gate",
    "library",
    "landmark_3",
    "landmark_4",
    "landmark_5",
    "landmark_6",
    "landmark_7",
]

IMAGE_LONG_SIDE = 1024      # px, long side after resizing
RATIO_TEST = 0.75           # Lowe's ratio test threshold
RANSAC_THRESHOLD = 5.0      # px, reprojection error for findHomography
MIN_INLIERS = 10            # below this, a pair counts as "no match"
HIST_BINS = (8, 8, 8)       # HSV color histogram bins
TINY_SIZE = 32              # tiny image is TINY_SIZE x TINY_SIZE grayscale
PCA_COMPONENTS = 50
KNN_K = 3
SEED = 42
