"""Q2 results: the decision rules of preregistration section 9.7, applied as written.

Per operand (a, b) and swap kind (digit, slow): the swap's mean shift of its own
supernode and mean off-target change of the other supernode, against the p95
of the 100 random sets; the ceiling; outcomes as context.

Output: printed report and results/q2_analysis/decisions.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RUNS = Path("results/q2_runs/runs")
OUT = Path("results/q2_analysis")
OUT.mkdir(parents=True, exist_ok=True)
OWN = {"digit": "last digit", "slow": "magnitude"}
OTHER = {"digit": "magnitude", "slow": "last digit"}


def load(name):
    return pd.read_csv(RUNS / f"{name.replace(' ', '_')}.csv")


def mean(x):
    x = np.asarray(x, float)
    return x[np.isfinite(x)].mean()


def boot_ci(x, n=1000, seed=0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    rng = np.random.default_rng(seed)
    return tuple(np.percentile([rng.choice(x, len(x)).mean() for _ in range(n)], [2.5, 97.5]).round(3))


rows = []
for op in ["a", "b"]:
    for kind in ["digit", "slow"]:
        swap, ceil = load(f"{op} {kind} swap"), load(f"{op} {kind} ceiling")
        rand = [load(f"{op} {kind} random {r:03d}") for r in range(100)]
        own, other = f"shift {OWN[kind]}", f"off-target {OTHER[kind]}"
        shift_r = np.array([mean(r[own]) for r in rand])
        off_r = np.array([mean(r[other]) for r in rand])
        pred_r = np.array([r["predicted"].mean() for r in rand])
        row = {"operand": op, "swap": kind, "n": len(swap),
               "shift own": mean(swap[own]), "shift CI": boot_ci(swap[own]),
               "shift p95": np.percentile(shift_r, 95), "shift random median": np.median(shift_r),
               "shift ceiling": mean(ceil[own]),
               "primary passes": mean(swap[own]) > np.percentile(shift_r, 95),
               "off-target other": mean(swap[other]), "off-target CI": boot_ci(swap[other]),
               "off-target p95": np.percentile(off_r, 95), "off-target random median": np.median(off_r),
               "other moved": mean(swap[other]) > np.percentile(off_r, 95),
               "predicted answer": swap["predicted"].mean(), "predicted p95": np.percentile(pred_r, 95),
               "predicted ceiling": ceil["predicted"].mean()}
        row.update({f"rate {o}": (swap["outcome"] == o).mean() for o in "NUMBX"})
        rows.append(row)
df = pd.DataFrame(rows)
df.to_csv(OUT / "decisions.csv", index=False)
pd.set_option("display.width", 200)
print(df.set_index(["operand", "swap"]).T.to_string())

for op in ["a", "b"]:
    d = df[df["operand"] == op].set_index("swap")
    primary = bool(d.loc["digit", "primary passes"] and d.loc["slow", "primary passes"])
    dm, sm = bool(d.loc["digit", "other moved"]), bool(d.loc["slow", "other moved"])
    claim = ("no evidence that the supernodes depend on these circles" if not primary else
             {(False, False): "each supernode depends on its own circles",
              (True, False): "the digit circles feed both supernodes; the slow circles only magnitude",
              (False, True): "the slow circles feed both supernodes; the digit circles only last digit",
              (True, True): "both supernodes depend on both groups; no split along the helix"}[(dm, sm)])
    print(f"\noperand {op}: primary {'pass' if primary else 'fail'}; digit swap moves magnitude: {dm}; "
          f"slow swap moves last digit: {sm}\n  claim (9.7): {claim}")
