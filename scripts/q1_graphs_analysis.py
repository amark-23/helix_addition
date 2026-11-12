"""Preregistration 7.3: descriptive summary of the 200 attribution graphs.

Means with bootstrap 95% CIs over prompts, overall and split by carry.
Edges are reported twice: as written in the summary (share of a target
feature's direct input from the whole source group) and per source feature
(divided by the source group's size), because the last-digit group has about
twice as many nodes per graph as the magnitude group.

Output: printed report and results/q1_graphs/analysis.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("results/q1_graphs")
df = pd.read_csv(OUT / "summary.csv")
assert len(df) == 200 and df["carry"].sum() == 100


def mean_ci(x, n_boot=2000, seed=0):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    rng = np.random.default_rng(seed)
    boot = [rng.choice(x, len(x)).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return x.mean(), lo, hi


for src, n_src in [("last digit", "n last digit"), ("magnitude", "n magnitude")]:
    for tgt in ["last digit", "magnitude"]:
        df[f"edges per source feature {src} -> {tgt}"] = df[f"edges {src} -> {tgt}"] / df[n_src]

cols = [c for c in df.columns if c not in ("a", "b", "carry")]
rows = {}
for c in cols:
    m, lo, hi = mean_ci(df[c])
    rows[c] = {"mean": m, "ci low": lo, "ci high": hi,
               "carry": df.loc[df["carry"], c].mean(), "no-carry": df.loc[~df["carry"], c].mean()}
table = pd.DataFrame(rows).T
table.to_csv(OUT / "analysis.csv")

pd.set_option("display.width", 200)
print(table.to_string(float_format=lambda v: f"{v:.4f}"))
shares = [c for c in df.columns if c.startswith("influence ")]
print("\ninfluence shares sum to 1 in every graph:", bool(np.allclose(df[shares].sum(axis=1), 1)))
