"""Compare hand labels (sheet.csv) with the automatic labels (key.csv) of one check.

python scripts/checks/blind_compare.py results/blind_check
python scripts/checks/blind_compare.py results/blind_check_2

Features labeled "mostly active" by hand have no family (they are defined by
how often they fire), so they are left out of the family comparison.
"""
import sys

import pandas as pd

folder = sys.argv[1] if len(sys.argv) > 1 else "results/blind_check"
m = pd.read_csv(f"{folder}/key.csv").merge(pd.read_csv(f"{folder}/sheet.csv"), on="id")
missing = m["my_role"].isna().sum()
if missing:
    print(f"warning: {missing} rows have no hand label yet")

for axis in ["role", "family"]:
    sub = m if axis == "role" else m[m["my_role"] != "mostly active"]
    agree = sub[axis] == sub[f"my_{axis}"]
    print(f"\n{axis}: hand and automatic labels agree on {agree.sum()} of {len(sub)} ({agree.mean():.0%})"
          + ("" if axis == "role" else "  (hand-labeled mostly active left out)"))
    print(pd.crosstab(sub[axis], sub[f"my_{axis}"], rownames=["automatic"], colnames=["hand"]).to_string())
    print("\nagreement per automatic label:")
    print(sub.assign(agree=agree).groupby(axis)["agree"].agg(["sum", "count", "mean"]).round(2).to_string())

print("\nrole disagreements:")
print(m[m["role"] != m["my_role"]][["id", "layer", "feature", "role", "my_role", "best_var", "best_r2", "note"]]
      .to_string(index=False))
