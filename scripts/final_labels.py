"""Final feature labels: the automatic labels, with two groups replaced by
manual reviews (criteria: notes/manual_review.md).

- Lookup tables whose best variable is tens (a, b): results/manual_review/tens_lookups.csv
- Add function features: results/manual_review/add_functions.csv
In both, "keep" keeps the automatic label and "drop" makes the feature
"unclear" (excluded from every supernode). Every other label is unchanged.

Output: results/feature_labels/data.csv with role, family and source
("auto", "manual keep" or "manual drop") for every scored feature.
"""
import pandas as pd

from helixadd.results import load_if_exists, save

FOLDER = "results/feature_labels"
REVIEWS = ["results/manual_review/tens_lookups.csv", "results/manual_review/add_functions.csv"]
params = {"scores": "results/feature_scores", "reviews": REVIEWS,
          "rule": "reviewed features count only if kept in their manual review"}

if load_if_exists(FOLDER, params) is not None:
    raise SystemExit(f"{FOLDER} already exists with these parameters.")

df = pd.read_csv("results/feature_scores/data.csv")[
    ["layer", "feature", "n_prompts", "best_var", "best_r2", "best_basis", "role", "family"]]
reviews = pd.concat([pd.read_csv(f)[["layer", "feature", "decision", "note"]] for f in REVIEWS])
if reviews["decision"].isna().any():
    raise SystemExit("some review rows have no decision yet")
if reviews.duplicated(["layer", "feature"]).any():
    raise SystemExit("a feature appears in more than one review")

df = df.merge(reviews, on=["layer", "feature"], how="left")
df["source"] = "auto"
df.loc[df["decision"] == "keep", "source"] = "manual keep"
df.loc[df["decision"] == "drop", "source"] = "manual drop"
df.loc[df["source"] == "manual drop", ["role", "family"]] = "unclear"
df = df.drop(columns="decision")
save(FOLDER, params, df)

print(f"{len(df)} features; source: {df['source'].value_counts().to_dict()}\n")
print(pd.crosstab(df["role"], df["family"], margins=True).to_string())
