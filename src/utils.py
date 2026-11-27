"""Small shared helpers.  Owner: BOTH"""
import cv2
import matplotlib.pyplot as plt

from src import config


def show(img, title="", ax=None):
    """Display a BGR or grayscale image with matplotlib (converts BGR to RGB)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    if img.ndim == 2:
        ax.imshow(img, cmap="gray")
    else:
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_title(title)
    ax.axis("off")
    return ax


def save_figure(fig, name):
    """Save a figure to results/<name>.png at 150 dpi."""
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.RESULTS_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path
