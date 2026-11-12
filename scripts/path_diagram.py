"""Illustration: path diagram for one carry and one no-carry prompt (36+59, 36+53).

Boxes are groups of nodes of the pruned attribution graph (helixadd.diagram).
An arrow's number is the share of the target box's positive direct input that
comes from the source box. Inputs from mostly active, unclear, rare and error
nodes are merged as "unlabeled" and written inside each box. Negative totals
(mostly the prefix token embeddings) are not drawn.
One prompt each: an illustration of the Q1 results, not evidence for them.

Output: figures/path_diagram.png, results/path_diagram/box_shares.csv.
"""
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from helixadd import diagram, graphs
from helixadd.groups import load_labels

FOLDER = "results/path_diagram"
PROMPTS = [(36, 59, "carry"), (36, 53, "no carry")]
MIN_SHARE = 0.05
MERGE = {"a token": "a", "features on a": "a", "b token": "b", "features on b": "b",
         "mostly active": "unlabeled", "unclear": "unlabeled", "error": "unlabeled", "rare (unscored)": "unlabeled"}
# box: (x, y, fill)
BOXES = {"a": (1.5, 0.0, "#eeeeee"), "b": (4.9, 0.0, "#eeeeee"),
         "comparison": (3.2, 2.3, "#e6d9f2"), "last digit": (0.6, 3.0, "#cfe2f3"),
         "magnitude": (5.8, 3.4, "#fde2c4"), "mod 100": (3.2, 5.0, "#d9ead3"), "answer": (3.2, 6.7, "#ffffff")}
EDGE_COLOR = {"a": "#888888", "b": "#888888", "last digit": "#2f6fb0", "comparison": "#7a4fa3",
              "magnitude": "#d9822b", "mod 100": "#4a8f3c"}
W, H = 1.9, 0.86


def shares(a, b, categories):
    e = diagram.box_edges(f"{FOLDER}/{a}_{b}", categories)
    e = e.replace({"source box": MERGE, "target box": MERGE})
    e = e[e["source box"] != e["target box"]]
    e = e.groupby(["source box", "target box"])["weight"].sum().reset_index()
    positive = e[e["weight"] > 0].groupby("target box")["weight"].sum()
    e["share"] = e["weight"] / e["target box"].map(positive)
    return e[e["target box"].isin(BOXES)]


def draw(ax, a, b, kind, e, off):
    names = {"a": f'"{a}"\ntoken and its features', "b": f'"{b}"\ntoken and its features',
             "last digit": "last digit", "comparison": "comparison", "magnitude": "magnitude",
             "mod 100": "sum mod 100", "answer": f'answer "{a + b}"'}
    unlabeled = e[e["source box"] == "unlabeled"].set_index("target box")["share"]
    patches = {}
    for box, (x, y, fill) in BOXES.items():
        patches[box] = FancyBboxPatch((x - W / 2, y - H / 2), W, H, boxstyle="round,pad=0.02,rounding_size=0.08",
                                      fc=fill, ec="#444444", lw=1.2 if box != "answer" else 2, zorder=3)
        ax.add_patch(patches[box])
        lines = [names[box]]
        small = []
        if box in off:
            small.append(f"switched off: answer {off[box]}")
        if box in unlabeled.index:
            small.append(f"{unlabeled[box]:.0%} of input unlabeled")
        ax.text(x, y + (0.16 if small else 0), lines[0], ha="center", va="center", fontsize=10, weight="bold", zorder=4)
        ax.text(x, y - 0.2, "\n".join(small), ha="center", va="center", fontsize=7.5, color="#444444", zorder=4)

    shown = e[e["source box"].isin(BOXES) & (e["share"] >= MIN_SHARE)]
    bend = {("last digit", "answer"): -0.25, ("magnitude", "answer"): 0.25,
            ("last digit", "comparison"): 0.15, ("comparison", "last digit"): 0.15,
            ("b", "last digit"): -0.22, ("a", "magnitude"): 0.38, ("b", "magnitude"): 0.15,
            ("last digit", "magnitude"): -0.3}
    label_at = {("last digit", "magnitude"): 0.22, ("a", "magnitude"): 0.7, ("b", "last digit"): 0.3}
    for _, r in shown.iterrows():
        s, t = r["source box"], r["target box"]
        (x0, y0, _), (x1, y1, _) = BOXES[s], BOXES[t]
        rad = bend.get((s, t), 0.0)
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>",
                                     mutation_scale=10 + 14 * r["share"], lw=0.8 + 9 * r["share"],
                                     color=EDGE_COLOR[s], alpha=0.85, patchA=patches[s], patchB=patches[t],
                                     shrinkA=2, shrinkB=2, zorder=2))
        cx, cy = (x0 + x1) / 2 + rad * (y1 - y0), (y0 + y1) / 2 - rad * (x1 - x0)   # control point of the bend
        f = label_at.get((s, t), 0.5)
        mx = (1 - f) ** 2 * x0 + 2 * f * (1 - f) * cx + f ** 2 * x1
        my = (1 - f) ** 2 * y0 + 2 * f * (1 - f) * cy + f ** 2 * y1
        ax.text(mx, my, f"{r['share']:.0%}", fontsize=8, ha="center", va="center", color=EDGE_COLOR[s], zorder=5,
                bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.9))
    ax.set_title(f"{a} + {b} = {a + b}  ({kind})", fontsize=12)
    ax.set_xlim(-0.6, 7.0), ax.set_ylim(-0.7, 7.4), ax.axis("off")


categories = graphs.feature_categories(load_labels())
sup = pd.read_csv(f"{FOLDER}/suppression.csv")
sup["output"] = sup["output"].str.split("\n").str[0]
fig, axes = plt.subplots(1, 2, figsize=(14, 8))
tables = []
for ax, (a, b, kind) in zip(axes, PROMPTS):
    e = shares(a, b, categories)
    off = sup[(sup["a"] == a) & (sup["b"] == b)].set_index("switched off")["output"].to_dict()
    draw(ax, a, b, kind, e, off)
    tables.append(e.assign(a=a, b=b))
fig.text(0.5, 0.015, "Arrow: share of the target box's positive direct input from the source box (shares of 5% or more). "
         "All boxes above the two operand boxes are features at \"=\".\nOne pruned attribution graph per prompt: an illustration, not evidence.",
         ha="center", fontsize=8.5, color="#444444")
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig("figures/path_diagram.png", dpi=200)
pd.concat(tables).round(4).to_csv(f"{FOLDER}/box_shares.csv", index=False)
print("saved figures/path_diagram.png")
