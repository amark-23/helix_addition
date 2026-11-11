"""The taxonomy as a PDF to keep open while labeling: figures/taxonomy/taxonomy.pdf.

One page per category: the math (which variable defines it), what it means for
addition, how it looks in an operand plot, what it gets confused with, and
examples (from figures/taxonomy/examples.csv, all outside the blind set).
"""
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from helixadd.plots import draw, operand_grid

A4 = (8.27, 11.69)
data = np.load("results/eq_activations/data.npz")
examples = pd.read_csv("figures/taxonomy/examples.csv")
# keep only examples whose current label still matches their category (the rules changed after blind check 1)
current = pd.read_csv("results/feature_scores/data.csv")[["layer", "feature", "role"]]
examples = examples.merge(current.rename(columns={"role": "now"}), on=["layer", "feature"])
examples = examples[(examples["category"] != "mostly active") | (examples["now"] == "mostly active")]
# and drop lookup examples that the manual review rejected (e.g. blobs with an L attached)
dropped = pd.read_csv("results/manual_review/tens_lookups.csv")
dropped = dropped[dropped["decision"] == "drop"][["layer", "feature"]].assign(rejected=True)
examples = examples.merge(dropped, on=["layer", "feature"], how="left")
examples = examples[examples["rejected"].isna()]

INTRO = [
    ("How to read an operand plot",
     "Each plot is one feature at the \"=\" token of the prompt \"calc: a+b=\". "
     "Vertical axis: a (first operand). Horizontal axis: b (second operand). Both run 0 to 99, "
     "so the plot covers all 10,000 prompts. Color: how strongly the feature fires on that prompt. "
     "White: it does not fire."),
    ("Directions in the plot",
     "Top-left to bottom-right: a + b is constant (sum). "
     "Bottom-left to top-right: a - b is constant (the a = b diagonal is a - b = 0). "
     "Vertical lines: b is constant. Horizontal lines: a is constant."),
    ("Two labels per feature",
     "Role (Anthropic's kinds): how the feature combines a and b, read from the shape. "
     "Family: what kind of condition it checks (last digit, magnitude, mod 100, comparison)."),
    ("Step 1, the role: ask in order, stop at the first yes",
     "1. Almost the whole plot filled (more than 90%)? -> mostly active.  "
     "2. Diagonal lines or bands, top-left to bottom-right? -> sum.  "
     "3. Only vertical bars, or only horizontal bars? -> add function.  "
     "4. L-shapes, corners, nested squares, or the a = b diagonal? -> comparison.  "
     "5. A regular grid, or a few separate blobs? -> lookup table.  "
     "6. None, several mixed, or a shape only in part of the plot? -> unclear."),
    ("Step 2, the family",
     "last digit: the pattern repeats every 10.  magnitude: one or a few smooth regions.  "
     "mod 100: diagonal lines exactly 100 apart, or one thin diagonal line.  "
     "comparison: about which operand is larger, or their difference.  unclear: none of these."),
    ("How the code decides (for reference)",
     "For each candidate variable v(a, b) the code computes R^2: the share of the feature's variation "
     "explained by the average activation within each value of v. It does this on the activation values "
     "and on the on/off pattern, and keeps the better one. A feature gets the category of its best variable "
     "only if R^2 >= 0.6 and it leads every other category by >= 0.15. A variable that contains a simpler "
     "one (e.g. (a+b) mod 100 contains (a+b) mod 10) must beat it by 0.05, and a joint lookup must beat "
     "the additive model (effect of a + effect of b) by 0.1. Otherwise: mixed or unclear."),
    ("When unsure",
     "Write unclear and say why in the note column. A careful \"unclear\" is better than a guess."),
]

PAGES = {
    "add function / magnitude": dict(
        math="Anthropic: unions of horizontal lines (a condition on a) and vertical lines (a condition on b).\n"
             "Typical: 1[a mod 10 = 9  OR  b mod 10 = 9]  (\"add 9\"), scored here by the additive model\n"
             "g1(a mod 10) + g2(b mod 10).  Ours: 1[b in [lo, lo+L)] or floor(b/10) (or the same for a).",
        means="Reads one operand on its own, before the two are combined. Anthropic's typical example, \"add 9\", "
              "fires whenever either operand ends in 9 (bottom left: a synthetic plot of that pattern). "
              "At \"=\" in Llama we found NONE of that kind: the method can detect it (a synthetic \"add 9\" "
              "scores 0.95 on the additive model), but no real feature does. The add function features we did "
              "find are the other kind: \"b is a single digit\" (b < 10), a single thick bar (examples below). "
              "The closest relative of \"add 9\" in Llama may be the nested squares (comparison / last digit): "
              "the same lines, cut off at the a = b diagonal.",
        looks="Anthropic's kind: thin full-length lines every 10, in both directions (bottom left). "
              "Llama's kind: one vertical bar at the left edge (b small), or one horizontal bar at the bottom (a small).",
        confused="Nested squares (comparison / last digit): lines every 10 that stop at the a = b diagonal. "
                 "Full-length lines -> add function; lines that turn a corner at a = b -> comparison."),
    "sum / last digit": dict(
        math="v(a, b) = (a + b) mod 10",
        means="The units digit of the answer: 36 + 59 = 95 -> 5. This is the output side of the last-digit pathway.",
        looks="Thin diagonal lines (top-left to bottom-right), one every 10 steps of a + b.",
        confused="mod 100 features: count the gap between lines. Every 10 -> last digit; 100 apart -> mod 100."),
    "sum / mod 100": dict(
        math="v(a, b) = (a + b) mod 100",
        means="The last two digits of the answer: 95 and 195 are the same value (95). "
              "Fits the period-100 components found in the number embeddings.",
        looks="Diagonal lines or bands exactly 100 apart (e.g. a + b = 55 and 155), or one thin line "
              "(an exact sum below 100, e.g. a + b = 99).",
        confused="sum / magnitude: a single wide band is a range of sums, not mod 100."),
    "sum / magnitude": dict(
        math="v(a, b) = 1[a + b in [lo, lo+L)]  or  floor((a + b)/10)",
        means="The rough size of the answer: \"the sum is about 30\". The low-precision side of addition.",
        looks="One wide diagonal band, or a corner (bottom-left: small sums; top-right: sums near 198).",
        confused="mod 100 (several lines 100 apart) and mostly active (a band that fills over half the plot)."),
    "lookup table / last digit": dict(
        math="v(a, b) = (a mod 10, b mod 10) jointly: 100 combinations (e.g. \"_6 + _9\");\n"
             "also carry = 1[(a mod 10) + (b mod 10) >= 10].",
        means="Combines the two units digits before the sum is formed, like a times table for units. "
              "Anthropic's \"_6 + _9\" features. Part of the last-digit pathway.",
        looks="A regular grid of dots or blobs repeating every 10 in both directions.",
        confused="Dense dot textures that fill the plot (mostly active), and nested squares (comparison / last digit)."),
    "lookup table / magnitude": dict(
        math="v(a, b) = (floor(a/10), floor(b/10)) jointly, and it must clearly beat the additive model\n"
             "g1(floor(a/10)) + g2(floor(b/10)): a real interaction between a and b.",
        means="\"a is about X AND b is about Y\": a low-precision lookup (Anthropic's \"~36 + ~60\").",
        looks="One or a few separate, compact blobs at specific (a, b) locations. Mirror pairs, e.g. "
              "(a about 75, b about 30) and (a about 30, b about 75), are normal: addition is commutative. "
              "All lookups of this kind were checked by hand (results/manual_review/).",
        confused="L-shapes whose arms meet on the a = b diagonal, even when drawn as beads (comparison); "
                 "bands from top-left to bottom-right (sum); bars crossing the whole plot (add function). "
                 "Test: can it be described as \"a about X and b about Y\", and nothing else?"),
    "comparison / comparison": dict(
        math="v(a, b) = min(a, b) or max(a, b) in a range or in decades;  a - b in a range;  1[a < b];  1[a = b]",
        means="Which operand is larger, or how far apart they are: \"the smaller operand is about 20\".",
        looks="L-shapes or corners that bend exactly at the a = b diagonal; the a = b diagonal itself "
              "(bottom-left to top-right); a region above or below that diagonal.",
        confused="Blobs (lookup table). An L-shape whose corner lies on a = b is comparison."),
    "comparison / last digit": dict(
        math="v(a, b) = max(a, b) mod 10  or  min(a, b) mod 10",
        means="The last digit of the larger (or smaller) operand. The pilot found \"max(a, b) ends in d\". "
              "\"max(a, b) ends in 9\" = \"a ends in 9 and a is larger, or b ends in 9 and b is larger\": "
              "Anthropic's \"add 9\" lines cut off at the a = b diagonal (hypothesis: Llama's version of add function).",
        looks="Nested squares: lines that turn a corner at the a = b diagonal, one every 10.",
        confused="Lookup grids (dots in rows and columns, no corners), and Anthropic-style add function "
                 "features (lines every 10 that run the full length without turning at a = b)."),
    "mostly active": dict(
        math="n_prompts / 9,992 > 0.9   (Anthropic: \"active on the '=' token of most of our 10,000 prompts\";\n"
             "we use 90%: at 50%, the first blind check showed clear shapes such as a < b being hidden)",
        means="A feature that is on for most addition prompts. Its value can still vary a little with a or b, "
              "but its shape says nothing, so it gets this role whatever it looks like.",
        looks="Almost the whole plot filled, often with a texture or a faint a = b diagonal.",
        confused="A plot filled on about half (e.g. one side of the a = b diagonal) is NOT mostly active: "
                 "label its shape (that one is comparison)."),
    "unclear": dict(
        math="No variable reaches R^2 >= 0.6, or two categories score about the same (\"mixed\").",
        means="Either noise (scattered specks), two conditions at once (a lattice only in part of the grid), "
              "or a kind the variable library does not cover.",
        looks="Specks, several shapes mixed, or a clear shape that only covers part of the plot.",
        confused="Nothing: when in doubt, this is the right answer."),
}


def wrap(text, width=95):
    return "\n".join(textwrap.fill(line, width) for line in text.split("\n"))


with PdfPages("figures/taxonomy/taxonomy.pdf") as pdf:
    fig = plt.figure(figsize=A4)
    fig.text(0.07, 0.95, "Feature taxonomy for addition at \"=\"", fontsize=17, weight="bold")
    y = 0.91
    for head, body in INTRO:
        fig.text(0.07, y, head, fontsize=11, weight="bold")
        body = wrap(body)
        fig.text(0.07, y - 0.012, body, fontsize=9, va="top")
        y -= 0.05 + 0.0145 * body.count("\n")
    pdf.savefig(fig)
    plt.close(fig)

    for cat, info in PAGES.items():
        fig = plt.figure(figsize=A4)
        fig.text(0.07, 0.95, cat, fontsize=17, weight="bold")
        y = 0.91
        for head, key in [("Math", "math"), ("What it means", "means"),
                          ("How it looks", "looks"), ("Confused with", "confused")]:
            fig.text(0.07, y, head, fontsize=11, weight="bold")
            body = wrap(info[key], 80 if key == "math" else 95)
            fig.text(0.07, y - 0.012, body, fontsize=9.5, va="top",
                     family="monospace" if key == "math" else None)
            y -= 0.04 + 0.016 * body.count("\n")
        if cat == "add function / magnitude":    # Anthropic's typical pattern, drawn from the math
            a_, b_ = np.meshgrid(np.arange(100), np.arange(100), indexing="ij")
            add9 = np.where((a_ % 10 == 9) | (b_ % 10 == 9), 1.0, np.nan)
            draw(fig.add_axes([0.07, 0.11, 0.19, 0.19 * A4[0] / A4[1]]), add9,
                 "synthetic: a or b ends in 9\n(Anthropic's \"add 9\")")
        ex = examples[examples["category"] == cat].head(8)
        for i, (_, f) in enumerate(ex.iterrows()):
            ax = fig.add_axes([0.07 + (i % 4) * 0.225, 0.36 - (i // 4) * 0.25, 0.19, 0.19 * A4[0] / A4[1] * 1.0])
            draw(ax, operand_grid(data, f["layer"], f["feature"]), f"L{f['layer']} #{f['feature']}")
        if cat == "add function / magnitude":
            fig.text(0.30, 0.17, "Bottom left: Anthropic's pattern, drawn from the math (not a Llama feature).\n"
                     "Top row: the add function features we found in Llama.", fontsize=8.5)
        fig.text(0.07, 0.035, "a on the vertical axis, b on the horizontal axis, white = not active. "
                 "Examples are outside the blind set.", fontsize=8, color="gray")
        pdf.savefig(fig)
        plt.close(fig)
print("saved figures/taxonomy/taxonomy.pdf")
