"""Six labeled example features at "=", for slides: three last-digit, three tens.

Output: figures/feature_examples.png
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from helixadd.plots import operand_grid

EXAMPLES = [  # (layer, feature, plain label)
    (15, 70977, "units digit of a + b"),
    (14, 61054, "units of a and b (lookup)"),
    (11, 113488, "units of the smaller operand"),
    (14, 53961, "tens of a and b (lookup)"),
    (15, 107310, "a + b in a range"),
    (10, 87749, "a vs b (fires when a > b)"),
]
COLORS = {"last digit": "#2E6FB0", "magnitude": "#D9822B", "comparison": "#7A55A6"}

labels = pd.read_csv("results/feature_labels/data.csv").set_index(["layer", "feature"])
data = np.load("results/eq_activations/data.npz")

fig, axes = plt.subplots(2, 3, figsize=(11, 7.4))
for ax, (layer, feature, text) in zip(axes.flat, EXAMPLES):
    f = labels.loc[(layer, feature)]
    ax.imshow(operand_grid(data, layer, feature), origin="lower", cmap="viridis")
    ax.set_xticks([0, 50, 99]); ax.set_yticks([0, 50, 99])
    ax.set_xlabel("b", fontsize=10); ax.set_ylabel("a", fontsize=10)
    ax.set_title(f"{f['family']}\n", fontsize=11, fontweight="bold", color=COLORS[f["family"]], loc="left")
    ax.text(0, 1.03, f"{text}   R² = {f['best_r2']:.2f}", transform=ax.transAxes, fontsize=10)
    for s in ax.spines.values():
        s.set_color("#999999")
fig.text(0.5, 0.01, "Each panel: one feature's activation at \"=\" for every a, b in 0-99 (white = off).",
         ha="center", fontsize=9, color="#555555")
fig.subplots_adjust(left=0.05, right=0.99, top=0.92, bottom=0.1, hspace=0.45, wspace=0.25)
fig.savefig("figures/feature_examples.png", dpi=150)
print("saved figures/feature_examples.png")
