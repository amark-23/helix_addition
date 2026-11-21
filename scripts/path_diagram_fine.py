"""Anthropic-style path diagram for 36 + 59 and 36 + 53, with finer boxes.

Boxes: the operand tokens; the features on each operand token, by the property of
the number they track (results/path_diagram/operand_features.csv); the features at
"=", by the variable they track (diagram.EQ_BOX_OF_VARIABLE); the answer. An arrow
is the share of the target box's positive direct input that comes from the source
box (shares of 10% or more). Mostly active and unclear features, features at other
positions and error nodes count in the total but are not drawn.

Output: figures/path_diagram_fine.png, results/path_diagram/fine_box_shares.csv.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from helixadd import diagram

FOLDER = "results/path_diagram"
PROMPTS = [(36, 59, "carry"), (36, 53, "no carry")]
MIN_SHARE = 0.10
labels = pd.read_csv("results/feature_labels/data.csv").set_index(["layer", "feature"])
op = pd.read_csv(f"{FOLDER}/operand_features.csv").set_index(["position", "layer", "feature"])
inter = pd.read_csv(f"{FOLDER}/box_interventions.csv")
OP_KIND = {"tens": "tens", "size": "tens", "units": "units", "parity": "units", "mod 5": "units"}
SLOW, FAST, COMP, M100, GREY = "#d9822b", "#2f6fb0", "#7a4fa3", "#4a8f3c", "#888888"
PATHWAY = {"units of a, b": FAST, "smaller ends in": FAST, "larger ends in": FAST, "sum ends in": FAST,
           "tens of a, b": SLOW, "sum in the": SLOW, "smaller in the": COMP, "larger in the": COMP, "a < b": COMP,
           "sum mod 100": M100}


def box_of(r):
    if r.kind == "logit":
        return "answer"
    if r.kind == "token":
        return {40: "a token", 42: "b token"}.get(r.pos, "other")
    if r.kind != "feature":
        return "other"
    if r.pos in (40, 42):
        side = "a" if r.pos == 40 else "b"
        key = (side, r.layer, r.feature)
        kind = OP_KIND.get(op.loc[key, "label"]) if key in op.index else None
        return f"{side} {kind}" if kind else "other"
    if r.pos == 43 and (r.layer, r.feature) in labels.index:
        L = labels.loc[(r.layer, r.feature)]
        if L["role"] not in ("mostly active", "unclear") and L["best_var"] in diagram.EQ_BOX_OF_VARIABLE:
            return diagram.EQ_BOX_OF_VARIABLE[L["best_var"]]
    return "other"


def shares(a, b):
    nodes = pd.read_csv(f"{FOLDER}/{a}_{b}/nodes.csv").set_index("index")
    edges = pd.read_csv(f"{FOLDER}/{a}_{b}/edges.csv")
    nodes["box"] = [box_of(r) for r in nodes.itertuples()]
    edges["s"] = nodes.loc[edges["source"], "box"].values
    edges["t"] = nodes.loc[edges["target"], "box"].values
    edges = edges[edges["s"] != edges["t"]]
    e = edges.groupby(["s", "t"])["weight"].sum().reset_index()
    total = e[e["weight"] > 0].groupby("t")["weight"].sum()
    e["share"] = e["weight"] / e["t"].map(total)
    return e, nodes["box"].value_counts()


# layout: three columns (last digit, comparison, magnitude), stages from bottom to top
COLX = {"blue": (0.8, 2.3), "purple": (3.8, 5.3), "orange": (6.8, 8.3)}
ROWS = [0.0, 1.35, 2.75, 4.05, 5.35, 6.75]
POS = {"a token": (3.8, ROWS[0]), "b token": (5.3, ROWS[0]),
       "a units": (0.8, ROWS[1]), "b units": (2.3, ROWS[1]), "a tens": (6.8, ROWS[1]), "b tens": (8.3, ROWS[1]),
       "smaller ends in": (0.8, ROWS[2]), "larger ends in": (2.3, ROWS[2]),
       "smaller in the": (3.8, ROWS[2]), "larger in the": (5.3, ROWS[2]),
       "units of a, b": (1.55, ROWS[3]), "a < b": (5.4, ROWS[3]), "tens of a, b": (7.55, ROWS[3]),
       "sum ends in": (1.55, ROWS[4]), "sum mod 100": (4.55, ROWS[4]), "sum in the": (7.55, ROWS[4]),
       "answer": (4.55, ROWS[5])}
W, H = 1.32, 0.78
COLUMNS = [("last digit", 1.55, "#2f6fb0"), ("comparison", 4.55, "#7a4fa3"), ("magnitude", 7.55, "#d9822b")]
STAGES = [(ROWS[0], "tokens"), (ROWS[1], "operand\nfeatures"), (ROWS[2], "at \"=\":\noperands"),
          (ROWS[3], "at \"=\":\ncombined"), (ROWS[4], "at \"=\":\nsum"), (ROWS[5], "output")]


def draw(ax, a, b, kind):
    e, counts = shares(a, b)
    off = inter[(inter["a"] == a) & (inter["b"] == b)].set_index("box")
    unlabeled = e[(e["t"] == "answer") & (e["s"] == "other")]["share"].sum()
    nodes = pd.read_csv(f"{FOLDER}/{a}_{b}/nodes.csv")
    circle_note = {}
    for side, pos in [("a", 40), ("b", 42)]:
        here = nodes[(nodes["kind"] == "feature") & (nodes["pos"] == pos)]
        keys = [(side, l, f) for l, f in zip(here["layer"], here["feature"]) if (side, l, f) in op.index]
        sub = op.loc[keys].reset_index()
        sub["kind"] = sub["label"].map(OP_KIND)
        for k, col, rand, name in [("tens", "slow share", 4 / 2048, "slow"), ("units", "fast share", 9 / 2048, "fast")]:
            m = sub.loc[sub["kind"] == k, col].mean()
            if np.isfinite(m):
                circle_note[f"{side} {k}"] = f"{name} circles {m / rand:.0f}x"

    # column bands and stage labels
    for title, x, c in COLUMNS:
        ax.add_patch(plt.Rectangle((x - 1.55, ROWS[1] - 0.6), 3.1, ROWS[4] - ROWS[1] + 1.2, fc=c, alpha=0.05, ec="none", zorder=0))
        ax.text(x, ROWS[4] + 0.75, title, ha="center", va="bottom", fontsize=10, color=c, weight="bold")
    for y, t in STAGES:
        ax.text(-0.95, y, t, ha="right", va="center", fontsize=7.5, color="#666666")

    names = {"a token": f'"{a}"', "b token": f'"{b}"', "a tens": f"{a}: tens\n({10 * (a // 10)}s)",
             "a units": f"{a}: last digit\n({a % 10})", "b tens": f"{b}: tens\n({10 * (b // 10)}s)",
             "b units": f"{b}: last digit\n({b % 10})", "answer": f"{a + b}"}
    patches = {}
    for box, (x, y) in POS.items():
        if box.startswith(("a ", "b ")) and box.split()[1] in ("tens", "units"):
            color = SLOW if box.endswith("tens") else FAST
        else:
            color = PATHWAY.get(box, GREY)
        plain = box in ("answer", "a token", "b token")
        patches[box] = FancyBboxPatch((x - W / 2, y - H / 2), W, H, boxstyle="round,pad=0.02,rounding_size=0.1",
                                      fc="white" if plain else color, ec=color if not plain else "#333333",
                                      lw=2 if box == "answer" else 1.3, zorder=3)
        if not plain:
            patches[box].set_alpha(None)
            patches[box].set_facecolor((*plt.matplotlib.colors.to_rgb(color), 0.16))
        ax.add_patch(patches[box])
        text = names.get(box) or diagram.eq_box_label(box, a, b).replace(": ", ":\n").replace(" (mod", "\n(mod")
        small = []
        if box in circle_note:
            small.append(circle_note[box])
        if box in off.index and str(off.loc[box, "answer"]).strip() != str(a + b):
            small.append(f"off: {off.loc[box, 'answer']}")
        if box == "answer":
            small.append(f"{unlabeled:.0%} input unlabeled")
        ax.text(x, y + (0.12 if small else 0), text, ha="center", va="center",
                fontsize=11 if box == "answer" else 7.6, weight="bold", zorder=4)
        if small:
            ax.text(x, y - 0.25, "\n".join(small), ha="center", va="center", fontsize=6.6,
                    color="#b22222" if any(t.startswith("off") for t in small) else "#333333", zorder=4)

    shown = e[e["s"].isin(POS) & e["t"].isin(POS)
              & ((e["share"] >= MIN_SHARE) | ((e["t"] == "answer") & (e["share"] >= 0.03)))]
    for r in shown.itertuples():
        (x0, y0), (x1, y1) = POS[r.s], POS[r.t]
        if y1 <= y0 + 0.1:                      # only upward arrows (no same-row or backward ones)
            continue
        if r.s.endswith("token") and not r.t.startswith(("a ", "b ")):
            continue                            # tokens: only their arrows into the operand features
        src_color = SLOW if r.s.endswith("tens") else FAST if r.s.endswith("units") else PATHWAY.get(r.s, GREY)
        same_row = abs(y0 - y1) < 0.1
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                     connectionstyle=f"arc3,rad={-0.3 if same_row else 0.0}",
                                     patchA=patches[r.s], patchB=patches[r.t], shrinkA=2, shrinkB=2,
                                     mutation_scale=9 + 10 * r.share, lw=0.8 + 5 * r.share, color=src_color,
                                     alpha=0.7, zorder=2))
        if r.share >= 0.2 and not same_row:
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            ax.text(mx, my, f"{r.share:.0%}", fontsize=6.5, color=src_color, ha="center", va="center", zorder=5,
                    bbox=dict(fc="white", ec="none", pad=0.5, alpha=0.85))
    ax.set_title(f"{a} + {b} = {a + b}  ({kind})", fontsize=12)
    ax.set_xlim(-1.9, 9.1), ax.set_ylim(-0.6, 7.4), ax.axis("off")
    return e.assign(a=a, b=b)


fig, axes = plt.subplots(1, 2, figsize=(17, 8.4))
tables = [draw(ax, a, b, k) for ax, (a, b, k) in zip(axes, PROMPTS)]
fig.text(0.5, 0.012, "Arrow: share of the target box's positive direct input from the source box (10% or more; % shown from 20%). "
         "Inputs to the answer from 3%. Not drawn: inputs from unlabeled nodes, direct inputs from the tokens to \"=\", "
         "and arrows within a row.\n\"off\" (red): the answer with that box "
         "switched off at \"=\", when it changes. \"circles Nx\": share of the features' input direction along the Q2 circles, "
         "relative to a random direction. One pruned graph per prompt: an illustration.", ha="center", fontsize=8, color="#444444")
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig("figures/path_diagram_fine.png", dpi=170)
pd.concat(tables).round(4).to_csv(f"{FOLDER}/fine_box_shares.csv", index=False)
print("saved figures/path_diagram_fine.png")
