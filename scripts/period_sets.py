"""Q2 step 1: which set of periods fits the number vectors best.

For each period set, operand and point: fit the helix on 80 numbers
(helixadd.helix.fit), predict the other 20, and record the share of their
variance explained. 10 random splits per point.

Output: results/period_sets/data.csv, params.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from helixadd.helix import fit, predict

SETS = {
    "4: 100, 10, 5, 2": [100, 10, 5, 2],
    "5: 100, 50, 10, 5, 2": [100, 50, 10, 5, 2],
    "7: all": [100, 50, 10, 5, 10 / 3, 2.5, 2],
}
N_SPLITS = 10
N_TRAIN = 80
SEED = 0

data = np.load("results/number_acts/data.npz")
rng = np.random.default_rng(SEED)
rows = []
for name, periods in SETS.items():
    for position in ["a", "b"]:
        for point in range(data[position].shape[0]):
            X = data[position][point].astype(float)
            for split in range(N_SPLITS):
                order = rng.permutation(100)
                train, test = order[:N_TRAIN], order[N_TRAIN:]
                mean, C = fit(train, X[train], periods)
                error = X[test] - predict(test, mean, C, periods)
                spread = X[test] - X[test].mean(axis=0)
                rows.append({"set": name, "position": position, "point": point, "split": split,
                             "explained": 1 - (error ** 2).sum() / (spread ** 2).sum()})

folder = Path("results/period_sets")
folder.mkdir(parents=True, exist_ok=True)
df = pd.DataFrame(rows)
df.to_csv(folder / "data.csv", index=False)
json.dump({"sets": SETS, "n_splits": N_SPLITS, "n_train": N_TRAIN, "seed": SEED,
           "measure": "share of the test numbers' variance explained by the fit",
           "source": "results/number_acts/data.npz"}, open(folder / "params.json", "w"), indent=2)

summary = df[df.point <= 11].groupby(["set", "position"]).explained.mean().unstack().round(3)
print(summary)
print(df.groupby(["set", "point"]).explained.mean().unstack(0).round(2))
