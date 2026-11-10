"""Operand plots for a random sample of features that fire at "=".

Each panel is one feature: a on the vertical axis, b on the horizontal axis,
color = the feature's activation at "=" on the prompt a+b (white = not active).
"""
import matplotlib.pyplot as plt
import numpy as np

N_FEATURES = 131072
MIN_PROMPTS = 20      # a feature must fire on at least this many prompts
N_SAMPLE = 48
SEED = 0

d = np.load("results/eq_activations/data.npz")
uid = d["layer"].astype(np.int64) * N_FEATURES + d["feature"]
ids, counts = np.unique(uid, return_counts=True)
eligible = ids[counts >= MIN_PROMPTS]
print(f"features firing on >= {MIN_PROMPTS} prompts: {len(eligible)}")

rng = np.random.default_rng(SEED)
sample = np.sort(rng.choice(eligible, size=N_SAMPLE, replace=False))

fig, axes = plt.subplots(6, 8, figsize=(20, 15))
for ax, fid in zip(axes.flat, sample):
    rows = uid == fid
    a = d["a"][d["prompt"][rows]]
    b = d["b"][d["prompt"][rows]]
    grid = np.full((100, 100), np.nan)       # NaN = not active, drawn white
    grid[a, b] = d["value"][rows]
    ax.imshow(grid, origin="lower", cmap="viridis")
    ax.set_title(f"L{fid // N_FEATURES} #{fid % N_FEATURES}  n={rows.sum()}", fontsize=9)
    ax.set_xticks([0, 50, 99]); ax.set_yticks([0, 50, 99])
fig.supxlabel("b"); fig.supylabel("a")
fig.tight_layout()
fig.savefig("figures/operand_sample.png", dpi=80)
print("saved figures/operand_sample.png")
