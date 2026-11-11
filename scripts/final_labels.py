"""Final feature labels: the automatic labels, with lookup tables found through
tens (a, b) replaced by the manual review.

- Lookup tables whose best variable is tens (a, b): kept as lookup / magnitude
  only if the manual review says "keep"; otherwise "unclear" (excluded).
- Every other label: the automatic label, unchanged.

Output: results/feature_labels/data.csv with role, family and source
("auto", "manual keep" or "manual drop") for every scored feature.
"""
import pandas as pd

from helixadd.results import load_if_exists, save

FOLDER = "results/feature_labels"
params = {"scores": "results/feature_scores", "review": "results/manual_review/tens_lookups.csv",
          "rule": "tens (a, b) lookup tables count only if kept in the manual review"}

if load_if_exists(FOLDER, params) is not None:
    raise SystemExit(f"{FOLDER} already exists with these parameters.")

scores = pd.read_csv("results/feature_scores/data.csv")
review = pd.read_csv("results/manual_review/tens_lookups.csv")
missing = review["decision"].isna().sum()
if missing:
    raise SystemExit(f"{missing} review rows have no decision yet.")

df = scores[["layer", "feature", "n_prompts", "best_var", "best_r2", "best_basis", "role", "family"]].copy()
df = df.merge(review[["layer", "feature", "decision", "note"]], on=["layer", "feature"], how="left")
df["source"] = "auto"
reviewed = df["decision"].notna()
df.loc[reviewed & (df["decision"] == "keep"), "source"] = "manual keep"
df.loc[reviewed & (df["decision"] == "drop"), "source"] = "manual drop"
df.loc[df["source"] == "manual drop", ["role", "family"]] = "unclear"
df = df.drop(columns="decision")
save(FOLDER, params, df)

print(f"{len(df)} features; source: {df['source'].value_counts().to_dict()}\n")
print(pd.crosstab(df["role"], df["family"], margins=True).to_string())
