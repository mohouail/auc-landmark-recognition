"""Landmark recognition: method A (ME) and method B (TEAMMATE).

Every predict function returns a list of landmark names, so all methods
can go through evaluate.evaluate() the same way.
"""


# ---------- Method A: KNN on SIFT match counts.  Owner: ME ----------

def build_reference(train_images, train_labels):
    """Precompute SIFT keypoints and descriptors for every training image.

    Returns:
        list of (keypoints, descriptors, label)
    """
    raise NotImplementedError


def predict_by_matches(image, reference, k=None):
    """Count good matches against every reference image, take the k images
    with the most matches (config.KNN_K by default) and vote.

    Returns:
        label: str
        best_count: int (number of good matches of the best reference,
                    used for the "unknown place" stretch goal)
    """
    raise NotImplementedError


# ---------- Method B: global features + classifier.  Owner: TEAMMATE ----------

def train_svm(X_train, y_train):
    """Returns a fitted sklearn SVC (scale features first)."""
    raise NotImplementedError


def train_mlp(X_train, y_train):
    """Returns a fitted sklearn MLPClassifier (scale features first)."""
    raise NotImplementedError
