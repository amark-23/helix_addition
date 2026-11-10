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


# Candidate variables. Each splits the prompts into groups; the value is the
# group number of each prompt. "family" is the box a feature lands in when this
# variable explains it best.
VARIABLES = {
    # name           family
    "a mod 10":      "last digit",
    "b mod 10":      "last digit",
    "sum mod 10":    "last digit",
    "units (a, b)":  "last digit",   # lookup: both units digits together
    "carry":         "last digit",   # units digits add up to 10 or more
    "a // 10":       "magnitude",
    "b // 10":       "magnitude",
    "sum // 10":     "magnitude",
    "tens (a, b)":   "magnitude",    # lookup: both tens digits together
    "sum mod 100":   "mod 100",
    "min // 10":     "comparison",
    "max // 10":     "comparison",
    "a < b":         "comparison",
    "a = b":         "comparison",
    "max mod 10":    "last digit",   # last digit of the larger operand
    "min mod 10":    "last digit",   # last digit of the smaller operand
}

# Range variables (scored with range_r2, not as groups) and their family.
RANGES = {
    "a range":    "magnitude",
    "b range":    "magnitude",
    "sum range":  "magnitude",
    "diff range": "comparison",   # a - b
    "max range":  "comparison",   # the larger operand
    "min range":  "comparison",   # the smaller operand
}

# Additive models (scored with r2_additive): an effect of a's digit plus an
# effect of b's digit, with no interaction. Name -> (family, the two variables).
ADDITIVE = {
    "tens separate":  ("magnitude",  "a // 10",  "b // 10"),
    "units separate": ("last digit", "a mod 10", "b mod 10"),
}

# A joint (lookup) variable only counts if it beats its additive model by a
# margin: otherwise the feature has no real interaction between a and b.
INTERACTION = {
    "tens (a, b)":  "tens separate",
    "units (a, b)": "units separate",
}

FAMILY = {**VARIABLES, **RANGES, **{k: v[0] for k, v in ADDITIVE.items()}}

# Anthropic's role, from what the variable depends on (operand-plot shape):
# one operand alone -> bars ("add function"), the sum -> diagonals ("sum"),
# both operands jointly -> grid ("lookup table"). Comparisons depend on both
# operands but not through the sum; Anthropic has no role for them (they
# would be "miscellaneous" there), so they get their own.
ROLE = {
    "a mod 10":     "add function",
    "b mod 10":     "add function",
    "a // 10":      "add function",
    "b // 10":      "add function",
    "a range":      "add function",
    "b range":      "add function",
    "sum mod 10":   "sum",
    "sum // 10":    "sum",
    "sum mod 100":  "sum",
    "sum range":    "sum",
    "units (a, b)": "lookup table",
    "tens (a, b)":  "lookup table",
    "carry":        "lookup table",
    "min // 10":    "comparison",
    "max // 10":    "comparison",
    "a < b":        "comparison",
    "diff range":   "comparison",
    "max range":    "comparison",
    "min range":    "comparison",
    "a = b":        "comparison",
    "max mod 10":   "comparison",
    "min mod 10":   "comparison",
    "tens separate":  "add function",   # both operands, each on its own: bars in both directions
    "units separate": "add function",
}

# Finer variable -> the variables it contains. A finer variable only counts
# as the best explanation if it beats every variable it contains by a margin.
NESTED = {
    "sum mod 100":  ["sum mod 10"],
    "units (a, b)": ["a mod 10", "b mod 10", "sum mod 10", "carry"],
    "tens (a, b)":  ["a // 10", "b // 10", "min // 10", "max // 10"],
}


def variables(a, b):
    """Group number of every prompt, for every variable in VARIABLES."""
    s = a + b
    groups = {
        "a mod 10":     a % 10,
        "b mod 10":     b % 10,
        "sum mod 10":   s % 10,
        "units (a, b)": (a % 10) * 10 + b % 10,
        "carry":        ((a % 10) + (b % 10) >= 10).astype(int),
        "a // 10":      a // 10,
        "b // 10":      b // 10,
        "sum // 10":    s // 10,
        "tens (a, b)":  (a // 10) * 10 + b // 10,
        "sum mod 100":  s % 100,
        "min // 10":    np.minimum(a, b) // 10,
        "max // 10":    np.maximum(a, b) // 10,
        "a < b":        (a < b).astype(int),
        "a = b":        (a == b).astype(int),
        "max mod 10":   np.maximum(a, b) % 10,
        "min mod 10":   np.minimum(a, b) % 10,
    }
    assert groups.keys() == VARIABLES.keys()
    return groups


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


def contains(fine, coarse):
    """True if the coarse grouping is a function of the fine one: every fine
    group has a single coarse value (knowing the fine group tells you the coarse)."""
    return all(len(np.unique(coarse[fine == g])) == 1 for g in np.unique(fine))


def assign_label(best_r2, margin, axis_value, r2_min=0.6, margin_min=0.15):
    """Final label on one axis (role or family) for one feature, from its shape.

    Built for precision: a feature only gets a label when the evidence is clear.
    Order: unclear (no variable reaches r2_min) -> mixed (best value not ahead
    of the others by margin_min) -> axis_value.
    "Mostly active" (Anthropic's frequency rule) is a separate yes/no flag,
    see is_mostly_active, so dense features still get a role from their shape.
    """
    if best_r2 < r2_min:
        return "unclear"
    if margin < margin_min:
        return "mixed"
    return axis_value


def is_mostly_active(n_prompts, n_total, share=0.5):
    """Anthropic: active on the "=" token of most of the prompts."""
    return n_prompts > share * n_total


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
