"""Labeling features: the variable library, the rules, and the scoring pipeline.

Each variable splits the 100 x 100 grid of prompts (a, b) into groups. A
feature is labeled by the variable that explains it best (R^2, see scoring.py),
under rules that stop flexible variables from winning by default. Every
variable has a role (Anthropic's kinds, from the operand-plot shape) and a
family (the kind of condition).
"""
import numpy as np
import pandas as pd

from helixadd.scoring import N_FEATURES, RANGE_LENGTHS, feature_matrix, r2_additive, r2_all, range_r2


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


def contains(fine, coarse):
    """True if the coarse grouping is a function of the fine one: every fine
    group has a single coarse value (knowing the fine group tells you the coarse)."""
    return all(len(np.unique(coarse[fine == g])) == 1 for g in np.unique(fine))


def assign_label(best_r2, margin, axis_value, r2_min=0.6, margin_min=0.15):
    """Final label on one axis (role or family) for one feature, from its shape.

    Built for precision: a feature only gets a label when the evidence is clear.
    Order: unclear (no variable reaches r2_min) -> mixed (best value not ahead
    of the others by margin_min) -> axis_value.
    Features that fire on almost every prompt get the role "mostly active"
    afterwards, whatever this returns (see score_features).
    """
    if best_r2 < r2_min:
        return "unclear"
    if margin < margin_min:
        return "mixed"
    return axis_value


def is_mostly_active(n_prompts, n_total, share=0.9):
    """Anthropic: active on the "=" token of most of the prompts."""
    return n_prompts > share * n_total

# Thresholds. Chosen for precision (a wrong label in a supernode is worse than
# a missing one) and adjusted once after the first blind check (preregistration 3.5).
SETTINGS = {
    "min_prompts": 20,     # features firing on fewer prompts are not scored
    "delta": 0.05,         # nested rule: a finer variable must beat what it contains by this much
    "delta_int": 0.1,      # a joint (lookup) variable must beat its additive model by this much
    "delta_flex": 0.2,     # tens (a, b) must beat every other variable by this much
    "mostly_active": 0.9,  # fires on more than this share of prompts -> "mostly active"
    "r2_min": 0.6,         # below this, the evidence is too weak -> "unclear"
    "margin_min": 0.15,    # the best role/family must lead the next one by this much, else "mixed"
}


def raw_scores(X, a, b):
    """R^2 of every variable for every column of X (plus the best range's lo and L).
    A column that never varies (e.g. always on, in the shape basis) scores 0."""
    out = {}
    groups = variables(a, b)
    with np.errstate(invalid="ignore", divide="ignore"):
        for name, g in groups.items():
            out[name] = r2_all(X, g)
        for name, (_, v1, v2) in ADDITIVE.items():
            out[name] = r2_additive(X, groups[v1], groups[v2])
        for name, v, shift in [("a range", a, 0), ("b range", b, 0),
                               ("sum range", a + b, 0), ("diff range", a - b + 99, 99),
                               ("max range", np.maximum(a, b), 0), ("min range", np.minimum(a, b), 0)]:
            out[name], lo, out[name + " L"] = range_r2(X, v)
            out[name + " lo"] = lo - shift      # diff range: back to the real a - b
    return pd.DataFrame(out).fillna(0.0)


def effective(r, s=SETTINGS):
    """Apply the nested, interaction and tens rules: a variable that only wins by
    containing a simpler one (or by being very flexible) is set to 0 for that feature."""
    eff = r[list(FAMILY)].copy()
    for fine, contained in NESTED.items():
        eff[fine] = eff[fine].where(r[fine] - r[contained].max(axis=1) >= s["delta"], 0.0)
    for name, (_, v1, v2) in ADDITIVE.items():
        eff[name] = eff[name].where(r[name] - r[[v1, v2]].max(axis=1) >= s["delta"], 0.0)
    for joint, additive in INTERACTION.items():
        eff[joint] = eff[joint].where(r[joint] - r[additive] >= s["delta_int"], 0.0)
    # tens (a, b) has 100 cells and fits almost any pattern at decade resolution
    others = eff.drop(columns="tens (a, b)").max(axis=1)
    eff["tens (a, b)"] = eff["tens (a, b)"].where(r["tens (a, b)"] - others >= s["delta_flex"], 0.0)
    return eff


def best_and_margin(eff, groups_of):
    """Best group (role or family) per feature, and its lead over the next group."""
    by_group = pd.DataFrame({g: eff[[v for v in eff if groups_of[v] == g]].max(axis=1)
                             for g in sorted(set(groups_of.values()))})
    top2 = np.sort(by_group.to_numpy(), axis=1)[:, -2:]
    return by_group.idxmax(axis=1), top2[:, 1] - top2[:, 0]


def score_features(data, s=SETTINGS):
    """Score and label every feature that fires at "=" on at least min_prompts prompts.

    data: the sparse activations (results/eq_activations). Returns one row per
    feature: R^2 of every variable on values and on the on/off pattern, the best
    variable, and the labels role and family.
    """
    a, b = data["a"], data["b"]
    X, ids = feature_matrix(data, s["min_prompts"])
    value = raw_scores(X, a, b)
    shape = raw_scores((X > 0).astype(np.float32), a, b)

    df = pd.DataFrame({"layer": ids // N_FEATURES, "feature": ids % N_FEATURES,
                       "n_prompts": (X > 0).sum(axis=0)})
    for col in value.columns:
        df["value: " + col] = value[col]
        df["shape: " + col] = shape[col]

    eff_v, eff_s = effective(value, s), effective(shape, s)
    eff = np.maximum(eff_v, eff_s)              # per variable, the better basis counts
    df["best_var"] = eff.idxmax(axis=1)
    df["best_r2"] = eff.max(axis=1)
    df["best_basis"] = np.where(eff_s.max(axis=1) > eff_v.max(axis=1), "shape", "value")
    df["best_role"], df["role_margin"] = best_and_margin(eff, ROLE)
    df["best_family"], df["family_margin"] = best_and_margin(eff, FAMILY)
    df["mostly_active"] = is_mostly_active(df["n_prompts"], len(a), s["mostly_active"])
    for axis in ["role", "family"]:
        df[axis] = [assign_label(r.best_r2, r[f"{axis}_margin"], r[f"best_{axis}"], s["r2_min"], s["margin_min"])
                    for _, r in df.iterrows()]
    # a dense feature's shape (always on) says nothing: it gets Anthropic's own role;
    # its family still records what its value tracks
    df.loc[df["mostly_active"], "role"] = "mostly active"
    return df
