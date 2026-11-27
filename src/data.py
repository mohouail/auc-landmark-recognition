"""Loading, resizing and splitting the dataset.  Owner: Ouail"""
import cv2
import numpy as np
import pandas as pd

from src import config

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def landmark_from_filename(filename):
    """'main_gate_2026-11-20_014.jpg' -> 'main_gate' (names may contain _)."""
    stem = filename.rsplit(".", 1)[0]
    return stem.rsplit("_", 2)[0]


def resize_all(overwrite=False):
    """Resize every photo in data/raw/ so its long side is IMAGE_LONG_SIDE,
    and save it to data/processed/ with the same folder and file name.

    Photos already smaller than IMAGE_LONG_SIDE are copied unchanged.
    OpenCV applies the phone's EXIF rotation when reading, so portrait
    photos come out upright.

    Returns:
        int, number of photos written
    """
    written = 0
    for landmark in config.LANDMARKS:
        src_dir = config.RAW_DIR / landmark
        if not src_dir.is_dir():
            print(f"[resize_all] missing folder: {src_dir}")
            continue
        dst_dir = config.PROCESSED_DIR / landmark
        dst_dir.mkdir(parents=True, exist_ok=True)
        for path in sorted(src_dir.iterdir()):
            if path.suffix.lower() not in IMAGE_EXTS:
                if path.suffix.lower() == ".heic":
                    print(f"[resize_all] skipped HEIC (convert to jpg): {path.name}")
                continue
            dst = dst_dir / path.name
            if dst.exists() and not overwrite:
                continue
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                print(f"[resize_all] could not read: {path}")
                continue
            h, w = img.shape[:2]
            scale = config.IMAGE_LONG_SIDE / max(h, w)
            if scale < 1:
                img = cv2.resize(img, (round(w * scale), round(h * scale)),
                                 interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            written += 1
    print(f"[resize_all] wrote {written} photos to {config.PROCESSED_DIR}")
    return written


def load_labels():
    """Read data/labels.csv.

    Returns:
        pandas.DataFrame with columns
        filename, landmark, date, split, distance, angle, lighting
    """
    df = pd.read_csv(config.LABELS_CSV, dtype=str).fillna("")
    allowed = set(config.LANDMARKS) | {config.UNKNOWN}
    bad = sorted(set(df["landmark"]) - allowed)
    if bad:
        raise ValueError(f"labels.csv has landmarks not in config.LANDMARKS: {bad}")
    bad_split = sorted(set(df["split"]) - {"train", "test", "pano"})
    if bad_split:
        raise ValueError(f"labels.csv has unknown split values: {bad_split}")
    _check_split_days(df)
    return df


def _check_split_days(df):
    """Warn if a landmark has train and test photos from the same day."""
    for landmark, g in df.groupby("landmark"):
        train_days = set(g.loc[g["split"] == "train", "date"])
        test_days = set(g.loc[g["split"] == "test", "date"])
        shared = train_days & test_days
        if shared:
            print(f"[load_labels] WARNING {landmark}: train and test share "
                  f"day(s) {sorted(shared)}")


def _image_path(filename, landmark=None):
    landmark = landmark or landmark_from_filename(filename)
    return config.PROCESSED_DIR / landmark / filename


def load_image(filename, gray=False, landmark=None):
    """Load one processed image by file name.

    Returns:
        np.ndarray, BGR (H, W, 3) uint8, or grayscale (H, W) if gray=True
    """
    path = _image_path(filename, landmark)
    flag = cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR
    img = cv2.imread(str(path), flag)
    if img is None:
        raise FileNotFoundError(f"cannot read {path} (did you run resize_all?)")
    return img


def load_dataset(split, gray=False):
    """Load all images of one split.

    Args:
        split: "train", "test" or "pano"
    Returns:
        images: list of np.ndarray
        labels: list of str (landmark names from config.LANDMARKS)
        filenames: list of str
    """
    df = load_labels()
    df = df[df["split"] == split]
    images, labels, filenames = [], [], []
    for row in df.itertuples(index=False):
        images.append(load_image(row.filename, gray=gray, landmark=row.landmark))
        labels.append(row.landmark)
        filenames.append(row.filename)
    print(f"[load_dataset] {split}: {len(images)} photos")
    return images, labels, filenames
