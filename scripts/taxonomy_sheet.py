"""Reference examples for the blind check (the "codebook") and a practice round.

All features here are outside the blind sample in results/blind_check/key.csv,
so nothing in the blind round is shown in advance.

Outputs:
- figures/taxonomy/page_1.png, page_2.png: N_EXAMPLES clear examples per category
- figures/taxonomy/practice.png: practice features, titled only with a letter
- results/practice/sheet.csv (you fill in), results/practice/key.csv (answers)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from helixadd.plots import draw, operand_grid

N_EXAMPLES = 8
N_PRACTICE = 20
SEED = 0
FIG = Path("figures/taxonomy")
PRACTICE = Path("results/practice")

scores = pd.read_csv("results/feature_scores/data.csv")
data = np.load("results/eq_activations/data.npz")
blind = pd.read_csv("results/blind_check/key.csv")
pool = scores.merge(blind[["layer", "feature"]], how="left", indicator=True)
pool = pool[pool["_merge"] == "left_only"].drop(columns="_merge")

# (row title, what makes it this category, which features)
CATEGORIES = [
    ("add function / magnitude", "bars: depends on one operand only (here mostly: b is small)",
     (pool["role"] == "add function")),
    ("sum / last digit", "diagonal lines every 10: the sum's last digit",
     (pool["role"] == "sum") & (pool["family"] == "last digit")),
    ("sum / mod 100", "diagonal lines 100 apart, or one thin line: the sum's last two digits",
     (pool["role"] == "sum") & (pool["family"] == "mod 100")),
    ("sum / magnitude", "one wide diagonal band or corner: the sum is in a range",
     (pool["role"] == "sum") & (pool["family"] == "magnitude")),
    ("lookup table / last digit", "a grid repeating every 10 both ways: both units digits together",
     (pool["role"] == "lookup table") & (pool["family"] == "last digit")),
    ("lookup table / magnitude", "one or a few blobs: a in some range AND b in some range",
     (pool["role"] == "lookup table") & (pool["family"] == "magnitude")),
    ("comparison / comparison", "L-shapes, corners, a = b diagonal: min, max or a - b",
     (pool["role"] == "comparison") & (pool["family"] == "comparison")),
    ("comparison / last digit", "nested squares: the last digit of min(a, b) or max(a, b)",
     (pool["role"] == "comparison") & (pool["family"] == "last digit")),
    ("mostly active", "active on more than half of all prompts: a large filled area, whatever its shape",
     (pool["role"] == "mostly active")),
    ("unclear", "specks, several shapes mixed, or a shape only in part of the grid",
     (pool["role"] == "unclear") & (pool["n_prompts"] >= 200)),
]


# Removed by hand after viewing: examples that are not textbook cases of their
# category (a codebook must not teach the code's own mistakes).
NOT_TEXTBOOK = {
    (14, 62293),                                       # sum / last digit: dense dots, lines hard to see
    (14, 3767), (12, 112333), (14, 70953), (14, 24399),  # lookup / last digit: dense dots, no clear grid
    (12, 31058), (11, 12461), (8, 102279),             # lookup / magnitude: L-shape or bar, not blobs
    (15, 75681), (14, 73345),                          # sum / last digit: dense dots
    (12, 113524), (12, 90428),                         # lookup / last digit: dense dots
    (11, 20170), (8, 86534),                           # lookup / magnitude: L-shape / diagonal region
    (14, 102022), (8, 114163),                         # lookup / magnitude: sum band / a < b triangle
    (11, 98772),                                       # lookup / last digit: dense dots
}


def pick(mask, n, seed):
    """Clear examples: among the top half by score in the category (all of them
    when the category is small, or for "unclear" / "mostly active"), a random n."""
    keep = [(l, f) not in NOT_TEXTBOOK for l, f in zip(pool["layer"], pool["feature"])]
    g = pool[mask & keep]
    if g["role"].iloc[0] not in ("unclear", "mostly active") and len(g) > 2 * n:
        g = g[g["best_r2"] >= g["best_r2"].median()]
    return g.sample(min(n, len(g)), random_state=seed)


FIG.mkdir(parents=True, exist_ok=True)
used = []
rows_per_page = 5
for page in range(2):
    cats = CATEGORIES[page * rows_per_page:(page + 1) * rows_per_page]
    fig, axes = plt.subplots(len(cats), N_EXAMPLES, figsize=(2.4 * N_EXAMPLES, 2.9 * len(cats)))
    for row, (title, why, mask) in zip(axes, cats):
        ex = pick(mask, N_EXAMPLES, SEED)
        used.append(ex.assign(category=title))
        for ax in row:
            ax.axis("off")
        for ax, (_, f) in zip(row, ex.iterrows()):
            ax.axis("on")
            draw(ax, operand_grid(data, f["layer"], f["feature"]), f"L{f['layer']} #{f['feature']}")
        row[0].text(0, 1.32, f"{title}:  {why}", transform=row[0].transAxes,
                    fontsize=12, fontweight="bold", va="bottom")
    fig.subplots_adjust(left=0.03, right=0.99, top=0.95, bottom=0.03, hspace=0.75, wspace=0.25)
    fig.savefig(FIG / f"page_{page + 1}.png", dpi=80)
    plt.close(fig)

pd.concat(used)[["category", "layer", "feature", "role", "family", "best_var", "best_r2"]].to_csv(
    FIG / "examples.csv", index=False)

# practice round: other labeled or unclear features, not in the blind set or the examples
shown = pd.concat(used)[["layer", "feature"]]
rest = pool.merge(shown, how="left", indicator=True)
rest = rest[rest["_merge"] == "left_only"]
rest = rest[(rest["role"] != "mixed") & ((rest["role"] != "unclear") | (rest["n_prompts"] >= 200))]
practice = (rest.groupby("role", group_keys=False)
            .apply(lambda g: g.sample(min(4, len(g)), random_state=SEED))
            .sample(n=N_PRACTICE, random_state=SEED).reset_index(drop=True))
practice["id"] = [chr(ord("A") + i) for i in range(len(practice))]

PRACTICE.mkdir(parents=True, exist_ok=True)
practice[["id", "layer", "feature", "role", "family", "best_var", "best_r2", "best_basis"]].to_csv(
    PRACTICE / "key.csv", index=False)
if not (PRACTICE / "sheet.csv").exists():
    pd.DataFrame({"id": practice["id"], "my_role": "", "my_family": "", "note": ""}).to_csv(
        PRACTICE / "sheet.csv", index=False)
fig, axes = plt.subplots(4, 5, figsize=(15, 12.5))
for ax, (_, f) in zip(axes.flat, practice.iterrows()):
    draw(ax, operand_grid(data, f["layer"], f["feature"]), f["id"])
fig.supxlabel("b")
fig.supylabel("a")
fig.tight_layout()
fig.savefig(FIG / "practice.png", dpi=80)
print("saved", FIG, "and", PRACTICE)
