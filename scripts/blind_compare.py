"""Compare your labels (sheet.csv) with the automatic ones (key.csv).

python scripts/blind_compare.py            -> the blind round (results/blind_check)
python scripts/blind_compare.py 2          -> the second blind round (results/blind_check_2)
python scripts/blind_compare.py practice   -> the practice round (results/practice)
"""
import sys

import pandas as pd

folder = {"practice": "results/practice", "2": "results/blind_check_2"}.get(
    (sys.argv[1:] or [""])[0], "results/blind_check")
sheet = pd.read_csv(f"{folder}/sheet.csv")
key = pd.read_csv(f"{folder}/key.csv")
m = key.merge(sheet, on="id")
missing = m["my_role"].isna().sum()
if missing:
    print(f"warning: {missing} rows have no my_role yet")

for axis in ["role", "family"]:
    # mostly active features have no family (your convention): leave them out of the family check
    sub = m if axis == "role" else m[m["my_role"] != "mostly active"]
    agree = sub[axis] == sub[f"my_{axis}"]
    print(f"\n{axis}: you agree with the automatic label on {agree.sum()} of {len(sub)} ({agree.mean():.0%})"
          + ("" if axis == "role" else "  (features you labeled mostly active left out)"))
    print(pd.crosstab(sub[axis], sub[f"my_{axis}"], rownames=["automatic"], colnames=["you"]).to_string())
    print("\nagreement per automatic label:")
    print(sub.assign(agree=agree).groupby(axis)["agree"].agg(["sum", "count", "mean"]).round(2).to_string())

print("\ndisagreements on role:")
print(m[m["role"] != m["my_role"]][["id", "layer", "feature", "role", "my_role", "best_var", "best_r2", "note"]]
      .to_string(index=False))
