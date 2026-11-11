"""Relaxed labels for the robustness variant (preregistration 4.2, 8.3).

The automatic labels with the R^2 bar at 0.5 instead of 0.6; every other rule
unchanged; no manual review applied (so every tens (a, b) lookup and every add
function feature counts as labeled).

Output: results/feature_labels_relaxed/data.csv (same columns as the final labels).
"""
import numpy as np
import pandas as pd

from helixadd.labels import SETTINGS, score_features
from helixadd.results import load_if_exists, save

FOLDER = "results/feature_labels_relaxed"
settings = {**SETTINGS, "r2_min": 0.5}
params = {"source": "results/eq_activations", "settings": settings, "manual_review": "none"}

df = load_if_exists(FOLDER, params)
if df is None:
    scores = score_features(np.load("results/eq_activations/data.npz"), settings)
    df = scores[["layer", "feature", "n_prompts", "best_var", "best_r2", "best_basis", "role", "family"]].assign(source="auto")
    save(FOLDER, params, df)

print(pd.crosstab(df["role"], df["family"], margins=True).to_string())
