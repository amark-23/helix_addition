"""Score every feature that fires at "=" on every candidate variable, and label it.

Each feature is scored twice:
- "value": R^2 of its activation values (what values it takes),
- "shape": R^2 of its on/off pattern, 1 where it fires (where it fires; this is
  what Anthropic's operand-plot categories describe).
Per variable, the higher of the two counts. Labels: role (Anthropic's kinds;
features active on most prompts get "mostly active") and family.

Output: results/feature_scores/data.csv. Thresholds are working values.
"""
import numpy as np
import pandas as pd

from helixadd.labels import (ADDITIVE, FAMILY, INTERACTION, N_FEATURES, NESTED,
                             RANGE_LENGTHS, ROLE, assign_label, feature_matrix,
                             is_mostly_active, r2_additive, r2_all, range_r2, variables)
from helixadd.results import load_if_exists, save

MIN_PROMPTS = 20
DELTA = 0.05          # nested rule: a finer variable must beat what it contains by this much
DELTA_INT = 0.1       # a joint (lookup) variable must beat its additive model by this much
MOSTLY_ACTIVE = 0.5   # fires on more than this share of prompts -> mostly_active flag
R2_MIN = 0.6          # below this, the evidence is too weak -> "unclear"
MARGIN_MIN = 0.15     # best role/family must lead the next one by this much, else "mixed"
FOLDER = "results/feature_scores"
params = {"source": "results/eq_activations", "min_prompts": MIN_PROMPTS,
          "family": FAMILY, "role": ROLE, "range_lengths": list(RANGE_LENGTHS),
          "nested": NESTED, "delta": DELTA, "interaction": INTERACTION, "delta_int": DELTA_INT,
          "mostly_active": MOSTLY_ACTIVE, "r2_min": R2_MIN, "margin_min": MARGIN_MIN,
          "bases": ["value", "shape"], "dense_role": "mostly active"}


def raw_scores(X, a, b):
    """R^2 of every variable for every column of X. NaN-free: a column that never
    varies (e.g. always on, in the shape basis) scores 0."""
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


def effective(r):
    """Apply the nested and interaction rules: a variable that only wins by
    containing a simpler one is set to 0 for that feature."""
    eff = r[list(FAMILY)].copy()
    for fine, contained in NESTED.items():
        eff[fine] = eff[fine].where(r[fine] - r[contained].max(axis=1) >= DELTA, 0.0)
    for name, (_, v1, v2) in ADDITIVE.items():
        eff[name] = eff[name].where(r[name] - r[[v1, v2]].max(axis=1) >= DELTA, 0.0)
    for joint, additive in INTERACTION.items():
        eff[joint] = eff[joint].where(r[joint] - r[additive] >= DELTA_INT, 0.0)
    return eff


def best_and_margin(eff, groups_of):
    """Best group (role or family) per feature, and its lead over the next group."""
    by_group = pd.DataFrame({g: eff[[v for v in eff if groups_of[v] == g]].max(axis=1)
                             for g in sorted(set(groups_of.values()))})
    top2 = np.sort(by_group.to_numpy(), axis=1)[:, -2:]
    return by_group.idxmax(axis=1), top2[:, 1] - top2[:, 0]


df = load_if_exists(FOLDER, params)
if df is None:
    data = np.load("results/eq_activations/data.npz")
    a, b = data["a"], data["b"]
    X, ids = feature_matrix(data, MIN_PROMPTS)
    value = raw_scores(X, a, b)
    print("scored values")
    shape = raw_scores((X > 0).astype(np.float32), a, b)
    print("scored shapes")

    df = pd.DataFrame({"layer": ids // N_FEATURES, "feature": ids % N_FEATURES,
                       "n_prompts": (X > 0).sum(axis=0)})
    for col in value.columns:
        df["value: " + col] = value[col]
        df["shape: " + col] = shape[col]

    eff_v, eff_s = effective(value), effective(shape)
    eff = np.maximum(eff_v, eff_s)              # per variable, the better basis counts
    df["best_var"] = eff.idxmax(axis=1)
    df["best_r2"] = eff.max(axis=1)
    df["best_basis"] = np.where(eff_s.max(axis=1) > eff_v.max(axis=1), "shape", "value")
    df["best_role"], df["role_margin"] = best_and_margin(eff, ROLE)
    df["best_family"], df["family_margin"] = best_and_margin(eff, FAMILY)
    df["mostly_active"] = is_mostly_active(df["n_prompts"], len(a), MOSTLY_ACTIVE)
    for axis in ["role", "family"]:
        df[axis] = [assign_label(r.best_r2, r[f"{axis}_margin"], r[f"best_{axis}"], R2_MIN, MARGIN_MIN)
                    for _, r in df.iterrows()]
    # A dense feature's shape (always on) says nothing, so a shape-based role does
    # not apply; it gets Anthropic's own role. Its family still records what its
    # value tracks.
    df.loc[df["mostly_active"], "role"] = "mostly active"
    save(FOLDER, params, df)

print(f"{len(df)} features\n")
print(pd.crosstab(df["role"], df["family"], margins=True).to_string())
print("\nbasis of the best score, by role:")
print(pd.crosstab(df["role"], df["best_basis"]).to_string())

