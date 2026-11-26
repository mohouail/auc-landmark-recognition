"""Landmark recognition: method A (ME) and method B (TEAMMATE).

Every predict function returns a list of landmark names, so all methods
can go through evaluate.evaluate() the same way.
"""
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src import config


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
