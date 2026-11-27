"""Follow-up to Q1: comparison, mixed family and the three tens groups together, each against
100 layer-matched random groups of its own size. Same rule as for magnitude: on all prompts,
M (tens only) above the random p95 and U (units only) at most the random p95.

Output: printed table and results/q1_tens_runs/decisions.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from helixadd.q1 import N_RANDOM, TENS_FOLDER, TENS_GROUPS, file_name

rows = []
for g in TENS_GROUPS:
    real = pd.read_csv(TENS_FOLDER / "runs" / file_name(g))
    rand = [pd.read_csv(TENS_FOLDER / "runs" / file_name(f"random | {g} | {k:03d}")) for k in range(N_RANDOM)]
    share = lambda df, o: (df["outcome"] == o).mean()
    pM, pU = np.percentile([share(r, "M") for r in rand], 95), np.percentile([share(r, "U") for r in rand], 95)
    rows.append({"group": g, "M": share(real, "M"), "random p95 M": pM, "U": share(real, "U"), "random p95 U": pU,
                 "B": share(real, "B"), "passes": share(real, "M") > pM and share(real, "U") <= pU})
df = pd.DataFrame(rows)
df.to_csv(TENS_FOLDER / "decisions.csv", index=False)
print(df.round(3).to_string(index=False))
