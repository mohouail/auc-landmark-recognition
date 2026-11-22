"""Panorama stitching (step 5).  Owner: ME"""


def stitch_pair(img_left, img_right):
    """Warp img_right onto img_left using matching.match_good's homography
    and blend the overlap.

    Returns:
        np.ndarray, the stitched image
    """
    raise NotImplementedError


def stitch_many(images):
    """Stitch a left-to-right ordered list of overlapping photos
    (stretch goal: 4 or more).

    Returns:
        np.ndarray, the panorama
    """
    raise NotImplementedError
