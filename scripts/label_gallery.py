"""Operand plots of a sample of features from each label, to check the labels by eye."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from helixadd.plots import draw, operand_grid

N_PER_ROW = 8
data = np.load("results/eq_activations/data.npz")

scores = pd.read_csv("results/feature_scores/data.csv")

# One row of panels per group. Each entry: (row name, the features in that group).
groups = []
for role in ["add function", "sum", "lookup table", "comparison"]:
    # the features labeled with this role (the label already includes the R^2 and margin rules)
    chosen = scores[scores["role"] == role]
    groups.append((role, chosen.sample(N_PER_ROW, random_state=0)))

# frequent features that no variable explains: n_prompts at least 500
# AND best_r2 below 0.3. Take 2 rows' worth (2 * N_PER_ROW) of them.
low = scores[(scores["n_prompts"] >= 500) & (scores["best_r2"] < 0.3)]
groups.append(("unexplained", low.sample(2 * N_PER_ROW, random_state=0)))

n_rows = sum(len(g) // N_PER_ROW for _, g in groups)
fig, axes = plt.subplots(n_rows, N_PER_ROW, figsize=(2.5 * N_PER_ROW, 2.6 * n_rows))
panels = iter(axes.flat)
for name, g in groups:
    for _, f in g.iterrows():
        # a title showing the row name, the layer and feature,
        # the best variable and its R^2 (rounded to 2 decimals)
        dense = "  [mostly active]" if f["mostly_active"] else ""
        title = (f"{name} / {f['family']}{dense}\nlayer {f['layer']}, feature {f['feature']}\n"
                 f"{f['best_var']} ({f['best_basis']}, R²={f['best_r2']:.2f})")
        draw(next(panels), operand_grid(data, f["layer"], f["feature"]), title)
fig.tight_layout()
fig.savefig("figures/label_gallery.png", dpi=80)
print("saved figures/label_gallery.png")
