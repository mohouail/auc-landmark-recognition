"""Small shared helpers.  Owner: BOTH (agree before changing a signature)"""


def show(img, title="", ax=None):
    """Display a BGR or grayscale image with matplotlib (converts BGR to RGB)."""
    raise NotImplementedError


def save_figure(fig, name):
    """Save a figure to results/<name>.png at 150 dpi."""
    raise NotImplementedError
