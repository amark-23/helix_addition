"""Blind check, round 2: tests the two fixes made after round 1
(mostly active from 50% to 90%, and the stricter rule for tens (a, b)).

New features only: none from round 1, the taxonomy examples or the practice set.
Sample: 10 lookup tables, 10 mostly active, and 10 features whose role changed
because of the fixes. Same procedure as round 1: label the pages in
results/blind_check_2/sheet.csv, do not open key.csv, then run
python scripts/blind_compare.py 2
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from helixadd.plots import draw, operand_grid

SEED = 1
OUT = Path("results/blind_check_2")
FIG = Path("figures/blind_check_2")
if (OUT / "sheet.csv").exists():
    raise SystemExit(f"{OUT / 'sheet.csv'} already exists; not overwriting your labels.")

new = pd.read_csv("results/feature_scores/data.csv")
old = pd.read_csv("results/feature_scores_round1/data.csv")[["layer", "feature", "role"]]
data = np.load("results/eq_activations/data.npz")

seen = pd.concat([pd.read_csv(f)[["layer", "feature"]] for f in
                  ["results/blind_check/key.csv", "figures/taxonomy/examples.csv", "results/practice/key.csv"]])
pool = new.merge(seen.drop_duplicates(), how="left", indicator=True)
pool = pool[pool["_merge"] == "left_only"].drop(columns="_merge")
pool = pool.merge(old.rename(columns={"role": "old_role"}), on=["layer", "feature"])

changed = pool[(pool["role"] != pool["old_role"]) & ~pool["role"].isin(["unclear", "mixed"])]
parts = [
    pool[pool["role"] == "lookup table"].sample(10, random_state=SEED),
    pool[pool["role"] == "mostly active"].sample(10, random_state=SEED),
]
taken = pd.concat(parts)[["layer", "feature"]]
changed = changed.merge(taken, how="left", indicator=True)
changed = changed[changed["_merge"] == "left_only"].drop(columns="_merge")
parts.append(changed.sample(10, random_state=SEED))
sample = pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)
sample["id"] = np.arange(1, len(sample) + 1)

OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
sample[["id", "layer", "feature", "role", "old_role", "family", "best_var", "best_r2", "best_basis",
        "n_prompts"]].to_csv(OUT / "key.csv", index=False)
pd.DataFrame({"id": sample["id"], "my_role": "", "my_family": "", "note": ""}).to_csv(OUT / "sheet.csv", index=False)

for page, start in enumerate(range(0, len(sample), 15), start=1):
    chunk = sample.iloc[start:start + 15]
    fig, axes = plt.subplots(3, 5, figsize=(15, 9.5))
    for ax in axes.flat:
        ax.axis("off")
    for ax, (_, f) in zip(axes.flat, chunk.iterrows()):
        ax.axis("on")
        draw(ax, operand_grid(data, f["layer"], f["feature"]), f"#{f['id']}")
    fig.supxlabel("b")
    fig.supylabel("a")
    fig.tight_layout()
    fig.savefig(FIG / f"page_{page}.png", dpi=80)
    plt.close(fig)
print(f"{len(sample)} features on {page} pages in {FIG}/; fill in {OUT / 'sheet.csv'}")
