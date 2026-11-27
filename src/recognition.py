"""Landmark recognition: method A (Ouail) and method B (Ahmed).

Every predict function returns a list of landmark names, so all methods
can go through evaluate.evaluate() the same way.
"""
from collections import defaultdict

from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src import config
from src.features import detect_sift
from src.matching import count_good_matches


# ---------- Method A: KNN on SIFT match counts.  Owner: Ouail ----------

def _strongest_sift(image):
    """SIFT keypoints and descriptors, keeping only the config.MAX_KEYPOINTS
    strongest (by response). Matching every test photo against every
    reference photo is the slow part, and the weak points rarely match."""
    kps, desc = detect_sift(image)
    if desc is None or len(kps) <= config.MAX_KEYPOINTS:
        return kps, desc
    order = sorted(range(len(kps)), key=lambda i: -kps[i].response)[:config.MAX_KEYPOINTS]
    return [kps[i] for i in order], desc[order]


def build_reference(train_images, train_labels):
    """Precompute SIFT keypoints and descriptors for every training image.

    Returns:
        list of (keypoints, descriptors, label)
    """
    return [(*_strongest_sift(img), label) for img, label in zip(train_images, train_labels)]


def match_counts(image, reference):
    """Good-match (RANSAC inlier) count between image and every reference photo.

    Returns:
        list of (count, label), one per reference photo, in reference order
    """
    kp, desc = _strongest_sift(image)
    return [(count_good_matches(kp, desc, rkp, rdesc), label)
            for rkp, rdesc, label in reference]


def vote(counts, k=None):
    """KNN vote: the k reference photos with the most good matches each vote
    for their landmark. A tie goes to the landmark whose voters have more
    good matches in total.

    Returns:
        label: str
        best_count: int
    """
    k = k or config.KNN_K
    nearest = sorted(counts, key=lambda c: -c[0])[:k]
    votes, totals = defaultdict(int), defaultdict(int)
    for count, label in nearest:
        votes[label] += 1
        totals[label] += count
    label = max(votes, key=lambda lab: (votes[lab], totals[lab]))
    return label, nearest[0][0]


def predict_by_matches(image, reference, k=None):
    """Count good matches against every reference image, take the k images
    with the most matches (config.KNN_K by default) and vote.

    Returns:
        label: str
        best_count: int (number of good matches of the best reference,
                    used for the "unknown place" stretch goal)
    """
    return vote(match_counts(image, reference), k)


def predict_all_by_matches(images, reference, k=None, reject_below=None):
    """Method A on a list of photos, ready for evaluate.evaluate().

    Args:
        reject_below: stretch goal. If set, a photo whose best reference has
                      fewer good matches than this is labelled config.UNKNOWN.
    Returns:
        labels: list of str
        best_counts: list of int
    """
    labels, best = [], []
    for i, img in enumerate(images, 1):
        label, count = predict_by_matches(img, reference, k)
        if reject_below is not None and count < reject_below:
            label = config.UNKNOWN
        labels.append(label)
        best.append(count)
        if i % 10 == 0 or i == len(images):
            print(f"[method A] {i}/{len(images)} photos")
    return labels, best


# ---------- Method B: global features + classifier.  Owner: Ahmed ----------

def train_svm(X_train, y_train):
    """Returns a fitted sklearn SVC (scale features first).

    It comes as a Pipeline (scaler then SVC), so .predict() scales new
    photos the same way and returns landmark names directly.
    """
    model = make_pipeline(
        StandardScaler(),
        SVC(C=config.SVM_C, kernel=config.SVM_KERNEL, gamma="scale",
            random_state=config.SEED),
    )
    return model.fit(X_train, y_train)


def train_mlp(X_train, y_train):
    """Returns a fitted sklearn MLPClassifier (scale features first).

    Also a Pipeline (scaler then MLP); .predict() returns landmark names.
    """
    model = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=config.MLP_HIDDEN, alpha=config.MLP_ALPHA,
                      max_iter=config.MLP_MAX_ITER, random_state=config.SEED),
    )
    return model.fit(X_train, y_train)


def predict_with_rejection(model, X, threshold):
    """Stretch goal: like model.predict, but returns config.UNKNOWN when the
    model's top probability is below threshold (a place outside our 7).
    Needs a model with predict_proba, i.e. the MLP."""
    proba = model.predict_proba(X)
    names = model.classes_[proba.argmax(axis=1)]
    return [n if p >= threshold else config.UNKNOWN
            for n, p in zip(names, proba.max(axis=1))]
