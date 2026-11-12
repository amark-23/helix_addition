"""Exploratory: unclear features that fire mostly on one exact value.

For every scored feature labeled unclear: the share of its firings that fall on
the single most common operand value (a or b), on the most common sum, and on
the most common difference a - b. A feature is assigned the kind with the
highest share if that share is at least 0.4. Found after the Q1 freeze by
looking at unlabeled features; not part of any preregistered test.

Output: results/exact_kinds/data.csv (layer, feature, n_prompts, kind, value, share).
"""
import os

import numpy as np
import pandas as pd

from helixadd.scoring import N_FEATURES

MIN_SHARE = 0.4
labels = pd.read_csv("results/feature_labels/data.csv")
d = np.load("results/eq_activations/data.npz")

unclear = labels[labels["role"] == "unclear"]
uid = d["layer"].astype(np.int64) * N_FEATURES + d["feature"]
keep = np.isin(uid, (unclear["layer"].astype(np.int64) * N_FEATURES + unclear["feature"]).to_numpy())
p = d["prompt"][keep]
fire = pd.DataFrame({"uid": uid[keep], "a": d["a"][p], "b": d["b"][p]})
fire["sum"], fire["difference"] = fire["a"] + fire["b"], fire["a"] - fire["b"]

rows = []
for u, g in fire.groupby("uid"):
    n = len(g)
    operand = pd.concat([g["a"], g["b"].where(g["b"] != g["a"])]).value_counts()   # a = b counted once
    options = {"exact operand value": (operand.index[0], operand.iloc[0] / n),
               "exact sum": (g["sum"].value_counts().index[0], g["sum"].value_counts().iloc[0] / n),
               "exact difference": (g["difference"].value_counts().index[0], g["difference"].value_counts().iloc[0] / n)}
    kind, (value, share) = max(options.items(), key=lambda kv: kv[1][1])
    if share >= MIN_SHARE:
        rows.append({"layer": u // N_FEATURES, "feature": u % N_FEATURES, "n_prompts": n,
                     "kind": kind, "value": int(value), "share": round(share, 3)})
out = pd.DataFrame(rows)
os.makedirs("results/exact_kinds", exist_ok=True)
out.to_csv("results/exact_kinds/data.csv", index=False)
print(out["kind"].value_counts().to_string())
print(out.groupby("kind")["value"].agg(lambda v: v.value_counts().head(6).to_dict()).to_string())
