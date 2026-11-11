"""How well a variable explains a feature: R^2 of group means (eta squared).

Every function here computes R^2 = 1 - SS_within / SS_tot for some way of
grouping the prompts. r2_groups is the plain definition (one feature, one
grouping); the others compute the same thing for every feature at once and are
tested against it (tests/test_labels.py).
"""
import numpy as np


def r2_groups(x, groups):
    """Fraction of the variance of x explained by the group means (eta squared).

    x: activations, one per prompt. groups: group number of each prompt.
    R^2 = 1 - SS_within / SS_tot.
    """
    total =    ((x - x.mean())**2).sum()       # SS_tot = Σ (xᵢ − x̄)²
    inside = 0.0
    for g in np.unique(groups):
        v = x[groups == g]                     #activations in group g
        inside += ((v - v.mean())**2).sum()    # Σ over group g of (xᵢ − x̄_g)²
    return 1-inside/total                      # R² = 1 − SS_within / SS_tot


N_FEATURES = 131072   # transcoder features per layer


def feature_matrix(data, min_prompts=20):
    """Unpack the sparse activations into a dense prompts x features matrix.

    Keeps features that fire on at least min_prompts prompts.
    Returns X (float32, zeros where a feature did not fire) and the unique id
    of each column (layer * N_FEATURES + feature).
    """
    uid = data["layer"].astype(np.int64) * N_FEATURES + data["feature"]
    ids, column, counts = np.unique(uid, return_inverse=True, return_counts=True)
    keep = counts >= min_prompts
    new_column = np.full(len(ids), -1)
    new_column[keep] = np.arange(keep.sum())
    col = new_column[column]
    rows = col >= 0
    X = np.zeros((len(data["a"]), keep.sum()), dtype=np.float32)
    X[data["prompt"][rows], col[rows]] = data["value"][rows]
    return X, ids[keep]


def r2_all(X, groups):
    """r2_groups for every column of X at once (same result, much faster)."""
    G = np.zeros((len(groups), groups.max() + 1), dtype=np.float32)
    G[np.arange(len(groups)), groups] = 1          # one-hot: a 1 in each prompt's group column
    n = G.sum(axis=0)                              # group sizes
    G, n = G[:, n > 0], n[n > 0]                   # drop group numbers that never occur
    mean = X.mean(axis=0)                          # overall mean of every feature
    M = (G.T @ X) / n[:, None]                     # group means: groups x features
    between = (n[:, None] * (M - mean) ** 2).sum(axis=0)
    total = ((X - mean) ** 2).sum(axis=0)
    return between / total                         # one R^2 per feature


def group_means(X, g):
    """Mean of every column of X in every group: array of shape (groups, features).
    Row k is group k (g must hold the integers 0, 1, ..., K-1)."""
    G = np.zeros((len(g), g.max() + 1), dtype=np.float32)
    G[np.arange(len(g)), g] = 1
    return (G.T @ X) / G.sum(axis=0)[:, None]


def r2_additive(X, g1, g2):
    """R^2 of the additive model: overall mean + effect of g1's group + effect of g2's group.

    Exact on a complete, balanced grid of g1 x g2 (ours misses 8 of 10,000 prompts).
    """
    mean = X.mean(axis=0)
    total = ((X - mean) ** 2).sum(axis=0)
    m1 = group_means(X, g1)          # row means: one row per group of g1
    m2 = group_means(X, g2)          # column means: one row per group of g2

    # the additive prediction for every prompt and feature,
    # shape (prompts, features). m1[g1] picks, for each prompt, the row of its g1 group.
    fitted = mean + m1[g1] + m2[g2] - m1.mean(axis=0) - m2.mean(axis=0)

    # the squared distances of X from fitted, summed over prompts (one value per feature)
    resid = ((X - fitted) ** 2).sum(axis=0)

    return 1 - resid / total


RANGE_LENGTHS = (5, 10, 20, 40)


def range_r2(X, values, lengths=RANGE_LENGTHS):
    """Best "value in [lo, lo + L)" split for every feature (Nikankin-style ranges).

    Tries every start lo and every length L, each as a 2-group variable
    (inside the range / outside). Returns, per feature, the best R^2 and the
    lo and L that achieved it. Same R^2 as r2_groups on the 2 groups.
    """
    N = len(values)
    mean = X.mean(axis=0)
    total = ((X - mean) ** 2).sum(axis=0)
    sum_x = X.sum(axis=0)
    starts, lens, cols = [], [], []
    for L in lengths:
        for lo in range(0, values.max() - L + 2):
            cols.append((values >= lo) & (values < lo + L))
            starts.append(lo)
            lens.append(L)
    R = np.stack(cols, axis=1).astype(np.float32)     # prompts x ranges, 1 = inside
    n1 = R.sum(axis=0)
    n0 = N - n1
    S1 = R.T @ X                                      # sum inside each range, per feature
    m1 = S1 / n1[:, None]                             # mean inside
    m0 = (sum_x - S1) / n0[:, None]                   # mean outside
    between = n1[:, None] * (m1 - mean) ** 2 + n0[:, None] * (m0 - mean) ** 2
    r2 = between / total
    best = r2.argmax(axis=0)
    return r2.max(axis=0), np.array(starts)[best], np.array(lens)[best]
