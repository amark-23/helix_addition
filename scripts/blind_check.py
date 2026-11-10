"""Blind check of the automatic labels.

Draws a random sample of labeled features (N_PER_ROLE per role), shuffles them,
and plots them with only a number as the title. You label each by eye in
results/blind_check/sheet.csv; the automatic labels are in key.csv, which you
should NOT open until the sheet is filled in. Then run scripts/blind_compare.py.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from helixadd.plots import draw, operand_grid

N_PER_ROLE = 15
PER_PAGE = 20
SEED = 0
OUT = Path("results/blind_check")
FIG = Path("figures/blind_check")

if (OUT / "sheet.csv").exists():
    raise SystemExit("results/blind_check/sheet.csv already exists; not overwriting your labels.")

scores = pd.read_csv("results/feature_scores/data.csv")
data = np.load("results/eq_activations/data.npz")

labeled = scores[~scores["role"].isin(["unclear", "mixed"])]
sample = (labeled.groupby("role", group_keys=False)
          .apply(lambda g: g.sample(min(N_PER_ROLE, len(g)), random_state=SEED))
          .sample(frac=1, random_state=SEED)          # shuffle, so roles are mixed together
          .reset_index(drop=True))
sample["id"] = np.arange(1, len(sample) + 1)

OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
sample[["id", "layer", "feature", "role", "family", "best_var", "best_r2", "best_basis"]].to_csv(
    OUT / "key.csv", index=False)
pd.DataFrame({"id": sample["id"], "my_role": "", "my_family": "", "note": ""}).to_csv(
    OUT / "sheet.csv", index=False)

for page, start in enumerate(range(0, len(sample), PER_PAGE), start=1):
    chunk = sample.iloc[start:start + PER_PAGE]
    fig, axes = plt.subplots(4, 5, figsize=(15, 12.5))
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
