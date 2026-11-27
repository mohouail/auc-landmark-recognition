"""SIFT matching with ratio test and RANSAC (step 3).  Owner: ME"""
import cv2
import matplotlib.pyplot as plt
import numpy as np

from src import config
from src.features import detect_sift

_matcher = cv2.BFMatcher(cv2.NORM_L2)   # L2 distance suits SIFT's float descriptors


def ratio_matches(desc1, desc2):
    """Brute-force kNN matching (k=2) + Lowe's ratio test (config.RATIO_TEST).

    For each descriptor in image 1 we find its two nearest descriptors in
    image 2. The match is kept only if the nearest is clearly closer than the
    second nearest, i.e. the match is unambiguous.

    Returns:
        list of cv2.DMatch
    """
    if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) < 2:
        return []
    good = []
    for pair in _matcher.knnMatch(desc1, desc2, k=2):
        if len(pair) == 2 and pair[0].distance < config.RATIO_TEST * pair[1].distance:
            good.append(pair[0])
    return good


def match_good(kp1, desc1, kp2, desc2):
    """Ratio-test matches, then RANSAC homography to drop outliers.

    RANSAC repeatedly fits a homography to 4 random matches and keeps the one
    that most other matches agree with (within config.RANSAC_THRESHOLD px).
    Matches that disagree are outliers and are dropped.

    Returns:
        inliers: list of cv2.DMatch (empty if fewer than 4 matches)
        H: 3x3 np.ndarray mapping image 1 -> image 2, or None
    """
    matches = one_to_one(ratio_matches(desc1, desc2), kp1, kp2)
    if len(matches) < 4:                 # a homography needs 4 points
        return [], None
    src = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, config.RANSAC_THRESHOLD)
    if H is None or mask is None or not plausible(H):
        return [], None
    inliers = [m for m, keep in zip(matches, mask.ravel()) if keep]
    return inliers, H


def one_to_one(matches, kp1, kp2):
    """Keep at most one match per location in each image (the closest one).

    Without this, many points of image 1 (e.g. the library's repeated
    windows) can all match the same point of image 2, and RANSAC happily
    fits a homography that squashes them onto that one point: a "match"
    between two different landmarks. SIFT can also put two keypoints on
    the same spot, so locations are compared rounded to the pixel.
    """
    best1, best2 = {}, {}
    for m in sorted(matches, key=lambda m: m.distance):
        p1 = tuple(np.round(kp1[m.queryIdx].pt).astype(int))
        p2 = tuple(np.round(kp2[m.trainIdx].pt).astype(int))
        if p1 not in best1 and p2 not in best2:
            best1[p1] = best2[p2] = m
    return list(best1.values())


def plausible(H):
    """Reject homographies no camera move could produce: mirrored (negative
    determinant) or squashing / stretching an area by more than
    config.MAX_SCALE_CHANGE times."""
    det = np.linalg.det(H[:2, :2] / H[2, 2])
    return 1 / config.MAX_SCALE_CHANGE < det < config.MAX_SCALE_CHANGE


def count_good_matches(kp1, desc1, kp2, desc2):
    """Number of RANSAC inliers between two images (used by method A)."""
    return len(match_good(kp1, desc1, kp2, desc2)[0])


def match_images(img1, img2):
    """Convenience for notebooks: SIFT on both images, then match_good.

    Returns:
        dict with kp1, kp2, ratio (ratio-test matches), inliers, H
    """
    kp1, d1 = detect_sift(img1)
    kp2, d2 = detect_sift(img2)
    inliers, H = match_good(kp1, d1, kp2, d2)
    return {"kp1": kp1, "kp2": kp2, "ratio": ratio_matches(d1, d2),
            "inliers": inliers, "H": H}


def _to_bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img


def _match_image(img1, kp1, img2, kp2, matches, max_draw):
    # Draw an even sample of at most max_draw matches, otherwise the lines hide
    # the photos. A sample (not the best ones) keeps the share of bad matches.
    step = max(1, -(-len(matches) // max_draw)) if max_draw else 1
    shown = matches[::step]
    out = cv2.drawMatches(_to_bgr(img1), kp1, _to_bgr(img2), kp2, shown, None,
                          matchColor=(0, 255, 0), singlePointColor=None,
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def draw_matches(img1, kp1, img2, kp2, matches, title="", max_draw=100):
    """Side-by-side figure of the matches, returned for saving.

    Returns:
        matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.imshow(_match_image(img1, kp1, img2, kp2, matches, max_draw))
    ax.set_title(f"{title} ({len(matches)} matches)" if title else f"{len(matches)} matches")
    ax.axis("off")
    fig.tight_layout()
    return fig


def draw_before_after(img1, img2, title="", max_draw=100):
    """Checkpoint figure: ratio-test matches on top, RANSAC inliers below.

    Returns:
        matplotlib.figure.Figure
    """
    r = match_images(img1, img2)
    fig, axes = plt.subplots(2, 1, figsize=(14, 11))
    for ax, matches, label in [(axes[0], r["ratio"], "after ratio test"),
                               (axes[1], r["inliers"], "after RANSAC")]:
        ax.imshow(_match_image(img1, r["kp1"], img2, r["kp2"], matches, max_draw))
        ax.set_title(f"{title}: {len(matches)} matches {label}".lstrip(": "))
        ax.axis("off")
    fig.tight_layout()
    return fig
