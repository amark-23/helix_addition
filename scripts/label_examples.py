"""One figure with a random sample of each final label (role / family).

Output: figures/label_examples.png
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from helixadd.plots import draw, operand_grid

N = 6
SEED = 0
GROUPS = [
    ("add function", "magnitude"), ("sum", "last digit"), ("sum", "mod 100"), ("sum", "magnitude"),
    ("lookup table", "last digit"), ("lookup table", "magnitude"), ("comparison", "comparison"),
    ("comparison", "last digit"), ("mostly active", None), ("mixed", "mixed"), ("unclear", "unclear"),
]

labels = pd.read_csv("results/feature_labels/data.csv")
data = np.load("results/eq_activations/data.npz")

fig, axes = plt.subplots(len(GROUPS), N, figsize=(2.3 * N, 2.5 * len(GROUPS)))
for row, (role, family) in zip(axes, GROUPS):
    g = labels[labels["role"] == role]
    if family is not None:
        g = g[g["family"] == family]
    if role == "unclear":
        g = g[g["n_prompts"] >= 200]          # skip the near-empty ones
    sample = g.sample(min(N, len(g)), random_state=SEED)
    for ax in row:
        ax.axis("off")
    for ax, (_, f) in zip(row, sample.iterrows()):
        ax.axis("on")
        draw(ax, operand_grid(data, f["layer"], f["feature"]), f"L{f['layer']} #{f['feature']}")
    name = role if family is None else f"{role} / {family}"
    row[0].text(0, 1.25, f"{name}  ({len(g)} features)", transform=row[0].transAxes,
                fontsize=11, fontweight="bold", va="bottom")
fig.subplots_adjust(left=0.04, right=0.99, top=0.97, bottom=0.02, hspace=0.75, wspace=0.3)
fig.savefig("figures/label_examples.png", dpi=70)
print("saved figures/label_examples.png")
