"""Operand plots of features labeled "unclear": what is in that pile?"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from helixadd.plots import draw, operand_grid

scores = pd.read_csv("results/feature_scores/data.csv")
data = np.load("results/eq_activations/data.npz")
unclear = scores[scores["role"] == "unclear"]

rows = [
    ("frequent", unclear[(unclear["n_prompts"] >= 500) & ~unclear["mostly_active"]].sample(16, random_state=0)),
    ("near the bar", unclear[unclear["best_r2"] >= 0.45].sample(16, random_state=0)),
    ("random", unclear.sample(16, random_state=1)),
]
fig, axes = plt.subplots(6, 8, figsize=(20, 16))
panels = iter(axes.flat)
for name, g in rows:
    for _, f in g.iterrows():
        title = (f"{name}  L{f['layer']} #{f['feature']}  n={f['n_prompts']}\n"
                 f"{f['best_var']} ({f['best_basis']}, R²={f['best_r2']:.2f})")
        draw(next(panels), operand_grid(data, f["layer"], f["feature"]), title)
fig.tight_layout()
fig.savefig("figures/unclear_gallery.png", dpi=80)
print("saved figures/unclear_gallery.png")
