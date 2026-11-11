"""Score and label every feature that fires at "=" (helixadd.labels.score_features).

Output: results/feature_scores/data.csv, one row per feature: R^2 of every
variable (on values and on the on/off pattern), the best variable, role and
family. Thresholds: helixadd.labels.SETTINGS.
"""
import numpy as np
import pandas as pd

from helixadd.labels import FAMILY, INTERACTION, NESTED, RANGE_LENGTHS, ROLE, SETTINGS, score_features
from helixadd.results import load_if_exists, save

FOLDER = "results/feature_scores"
s = SETTINGS
params = {"source": "results/eq_activations", "min_prompts": s["min_prompts"],
          "family": FAMILY, "role": ROLE, "range_lengths": list(RANGE_LENGTHS),
          "nested": NESTED, "delta": s["delta"], "interaction": INTERACTION, "delta_int": s["delta_int"],
          "delta_flex": s["delta_flex"], "mostly_active": s["mostly_active"], "r2_min": s["r2_min"],
          "margin_min": s["margin_min"], "bases": ["value", "shape"], "dense_role": "mostly active"}

df = load_if_exists(FOLDER, params)
if df is None:
    df = score_features(np.load("results/eq_activations/data.npz"))
    save(FOLDER, params, df)

print(f"{len(df)} features\n")
print(pd.crosstab(df["role"], df["family"], margins=True).to_string())
