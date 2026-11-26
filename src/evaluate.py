"""Evaluation shared by all methods.  Owner: TEAMMATE"""
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix

from src import config
from src.utils import save_figure


def evaluate(y_true, y_pred, name):
    """Print accuracy, save a labelled confusion matrix to
    results/confusion_<name>.png, and return the accuracy.

    Args:
        y_true, y_pred: lists of landmark names
        name: method name, e.g. "knn_sift", "svm_hist", "mlp_pca"
    Returns:
        float accuracy
    """
    acc = accuracy_score(y_true, y_pred)
    extra = sorted((set(y_true) | set(y_pred)) - set(config.LANDMARKS))
    labels = list(config.LANDMARKS) + extra          # e.g. "unknown"
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(
        ax=ax, cmap="Blues", colorbar=False, xticks_rotation=45)
    ax.set_title(f"{name}: accuracy {acc:.1%} ({len(y_true)} test photos)")
    ax.set_xlabel("Predicted landmark")
    ax.set_ylabel("True landmark")
    save_figure(fig, f"confusion_{name}")
    plt.show()
    plt.close(fig)

    print(f"{name}: accuracy {acc:.3f}")
    return acc


def results_table(results):
    """Turn {method name: accuracy} into a sorted table and save it to
    results/accuracy.csv.

    Returns:
        pandas.DataFrame
    """
    df = (pd.DataFrame(sorted(results.items(), key=lambda kv: -kv[1]),
                       columns=["method", "accuracy"])
          .reset_index(drop=True))
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.RESULTS_DIR / "accuracy.csv", index=False)
    return df
