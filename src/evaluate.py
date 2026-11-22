"""Evaluation shared by all methods.  Owner: TEAMMATE"""


def evaluate(y_true, y_pred, name):
    """Print accuracy, save a labelled confusion matrix to
    results/confusion_<name>.png, and return the accuracy.

    Args:
        y_true, y_pred: lists of landmark names
        name: method name, e.g. "knn_sift", "svm_hist", "mlp_pca"
    Returns:
        float accuracy
    """
    raise NotImplementedError


def results_table(results):
    """Turn {method name: accuracy} into a sorted table and save it to
    results/accuracy.csv.

    Returns:
        pandas.DataFrame
    """
    raise NotImplementedError
