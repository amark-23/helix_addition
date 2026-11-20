"""Q2 descriptive runs (preregistration 9.8, 9.12): summary tables.

Output: printed tables and results/q2_analysis/descriptive_*.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DESC = Path("results/q2_descriptive/runs")
MAIN = Path("results/q2_runs/runs")
OUT = Path("results/q2_analysis")
OUT.mkdir(parents=True, exist_ok=True)
COLS = ["predicted", "shift last digit", "off-target last digit", "shift magnitude", "off-target magnitude",
        "shift comparison", "off-target comparison"]
pd.set_option("display.width", 220)


def load(name, folder=DESC):
    return pd.read_csv(folder / f"{name.replace(' ', '_')}.csv")


def mean(x):
    x = np.asarray(x, float)
    return x[np.isfinite(x)].mean()


def summarize(df):
    row = {c: (mean(df[c]) if c in df else np.nan) for c in COLS}
    row.update({f"rate {o}": (df["outcome"] == o).mean() for o in "UMB"})
    row["n"] = len(df)
    return row


PERIODS = ["10", "5", "3.33333", "2.5", "2", "100", "50"]
for op in ["a", "b"]:
    swaps = {"main digit swap": summarize(load(f"{op} digit swap", MAIN)),
             "digit swap, last digit clamped": summarize(load(f"{op} digit swap clamped")),
             "digit swap, carry flipped": summarize(load(f"{op} digit swap carry flip")),
             "fast group (odd-difference pairs)": summarize(load(f"{op} fast group swap")),
             "slow group (tens difference not 5)": summarize(load(f"{op} slow group swap"))}
    swaps.update({f"period {T} alone": summarize(load(f"{op} period {T} swap")) for T in PERIODS})
    t = pd.DataFrame(swaps).T
    t.to_csv(OUT / f"descriptive_swaps_{op}.csv")
    print(f"\n=== operand {op}: swaps\n{t.round(3).to_string()}")

    sub = {lab: summarize(load(f"{op} sub-helix {lab}"))
           for lab in ["digit group", "slow group"] + [f"period {T}" for T in PERIODS]}
    t = pd.DataFrame(sub).T
    t.to_csv(OUT / f"descriptive_subhelix_{op}.csv")
    print(f"\n=== operand {op}: sub-helix patching (prompt a' + b; predicted = the answer a + b)\n{t.round(3).to_string()}")

    d = load(f"{op} digit swap carry flip")
    d = d[d["pred"].notna()]
    target, clean = d["a2"] + d["b2"], d["a"] + d["b"]
    switched = (d["pred"] % 10) == (target % 10)
    print(f"\ncarry flip ({op}): units switched {switched.mean():.3f}; tens follow the new carry "
          f"{((d['pred'] // 10) == (target // 10)).mean():.3f}, stay as clean {((d['pred'] // 10) == (clean // 10)).mean():.3f}; "
          f"among prompts whose units switched, tens follow the carry {((d['pred'] // 10) == (target // 10))[switched].mean():.3f}")

    rows = []
    for g, own in [("digit", "last digit"), ("slow", "magnitude")]:
        for i in range(12):
            sw = load(f"{op} {g} point {i:02d} swap")
            rand = [mean(load(f"{op} {g} point {i:02d} random {r:02d}")[f"shift {own}"]) for r in range(20)]
            rows.append({"group": g, "point": "embedding" if i == 0 else f"after L{i - 1}",
                         "shift own": mean(sw[f"shift {own}"]), "random p95": np.percentile(rand, 95),
                         "predicted": sw["predicted"].mean()})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / f"descriptive_points_{op}.csv", index=False)
    print(f"\n=== operand {op}: one point at a time (200 prompts)\n{t.round(3).to_string(index=False)}")
