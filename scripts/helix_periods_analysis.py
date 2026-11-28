"""Helix patching with 4, 5 and 7 periods: share of the real vector's effect kept, per point.

Input: results/helix_patching_periods/ (notebook 14). Output: printed table and
results/helix_patching_periods/shares.csv.
"""
import pandas as pd

from helixadd.helix import PERIOD_SETS

folder = "results/helix_patching_periods"
df = pd.read_csv(f"{folder}/data.csv")
p = df.pivot_table(index=["position", "point"], columns="method", values="logit difference")
shares = p[list(PERIOD_SETS)].div(p["layer patch"], axis=0)
shares.round(3).to_csv(f"{folder}/shares.csv")
print(shares.round(2).to_string())
print("\nMean over points 0-11:")
print(shares.groupby("position").mean().round(2).to_string())
