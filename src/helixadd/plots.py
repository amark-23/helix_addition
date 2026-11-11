import numpy as np

from helixadd.scoring import N_FEATURES


def operand_grid(data, layer, feature):
    """100 x 100 grid of one feature's activation at "=": row a, column b.
    NaN where the feature did not fire (drawn white)."""
    rows = (data["layer"] == layer) & (data["feature"] == feature)
    p = data["prompt"][rows]
    grid = np.full((100, 100), np.nan)
    grid[data["a"][p], data["b"][p]] = data["value"][rows]
    return grid


def draw(ax, grid, title):
    ax.imshow(grid, origin="lower", cmap="viridis")
    ax.set_title(title, fontsize=8)
    ax.set_xticks([0, 50, 99])
    ax.set_yticks([0, 50, 99])
