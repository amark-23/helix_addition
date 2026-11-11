"""Manual review of the lookup tables whose label comes from tens (a, b).

The blind checks showed these agree with manual inspection only about half the
time (11/24), so they do not count automatically. You review each one and keep
it as a magnitude lookup only if it clearly shows the blob pattern.

Outputs:
- figures/review/page_*.png: the features, titled with a review number and the feature ID
- results/manual_review/tens_lookups.csv: fill in decision (keep / drop) and an optional note
Instructions: notes/manual_review.md
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from helixadd.plots import draw, operand_grid

OUT = Path("results/manual_review")
FIG = Path("figures/review")
if (OUT / "tens_lookups.csv").exists():
    raise SystemExit(f"{OUT / 'tens_lookups.csv'} already exists; not overwriting your decisions.")

scores = pd.read_csv("results/feature_scores/data.csv")
data = np.load("results/eq_activations/data.npz")
todo = (scores[(scores["role"] == "lookup table") & (scores["best_var"] == "tens (a, b)")]
        .sort_values(["layer", "feature"]).reset_index(drop=True))
todo["id"] = np.arange(1, len(todo) + 1)

OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
todo[["id", "layer", "feature", "n_prompts", "best_r2"]].assign(decision="", note="").to_csv(
    OUT / "tens_lookups.csv", index=False)

for page, start in enumerate(range(0, len(todo), 20), start=1):
    chunk = todo.iloc[start:start + 20]
    fig, axes = plt.subplots(4, 5, figsize=(15, 12.5))
    for ax in axes.flat:
        ax.axis("off")
    for ax, (_, f) in zip(axes.flat, chunk.iterrows()):
        ax.axis("on")
        draw(ax, operand_grid(data, f["layer"], f["feature"]), f"#{f['id']}   (L{f['layer']} #{f['feature']})")
    fig.supxlabel("b")
    fig.supylabel("a")
    fig.tight_layout()
    fig.savefig(FIG / f"page_{page}.png", dpi=80)
    plt.close(fig)
print(f"{len(todo)} features on {page} pages in {FIG}/; decisions go in {OUT / 'tens_lookups.csv'}")
