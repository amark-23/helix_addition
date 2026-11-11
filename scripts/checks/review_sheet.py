"""Pages for the manual reviews (criteria: notes/manual_review.md).

python scripts/checks/review_sheet.py tens   lookup tables found through tens (a, b)
                                             (blind checks: 11/24 agreement)
python scripts/checks/review_sheet.py add    add function features (the dense ones
                                             were never blind-checked)

Draws figures/review_<kind>/page_*.png and, if the decisions file
results/manual_review/<file>.csv does not exist yet, creates it with an empty
decision column.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from helixadd.plots import draw, operand_grid

kind = sys.argv[1] if len(sys.argv) > 1 else "tens"
CSV = Path({"tens": "results/manual_review/tens_lookups.csv",
            "add": "results/manual_review/add_functions.csv"}[kind])
FIG = Path(f"figures/review_{kind}")
data = np.load("results/eq_activations/data.npz")

if CSV.exists():
    todo = pd.read_csv(CSV)
else:
    scores = pd.read_csv("results/feature_scores/data.csv")
    pick = ((scores["role"] == "lookup table") & (scores["best_var"] == "tens (a, b)") if kind == "tens"
            else scores["role"] == "add function")
    todo = scores[pick].sort_values(["layer", "feature"]).reset_index(drop=True)
    todo["id"] = np.arange(1, len(todo) + 1)
    CSV.parent.mkdir(parents=True, exist_ok=True)
    todo[["id", "layer", "feature", "n_prompts", "best_r2"]].assign(decision="", note="").to_csv(CSV, index=False)

FIG.mkdir(parents=True, exist_ok=True)
for page, start in enumerate(range(0, len(todo), 20), start=1):
    fig, axes = plt.subplots(4, 5, figsize=(15, 12.5))
    for ax in axes.flat:
        ax.axis("off")
    for ax, (_, f) in zip(axes.flat, todo.iloc[start:start + 20].iterrows()):
        ax.axis("on")
        draw(ax, operand_grid(data, f["layer"], f["feature"]), f"#{f['id']}   (L{f['layer']} #{f['feature']})")
    fig.supxlabel("b")
    fig.supylabel("a")
    fig.tight_layout()
    fig.savefig(FIG / f"page_{page}.png", dpi=80)
    plt.close(fig)
print(f"{len(todo)} features on {page} pages in {FIG}/")
