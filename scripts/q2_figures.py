"""Q2 figures (main, descriptive and post hoc runs).

figures/q2_results.png, four panels:
A. each supernode follows its own circles: swap shift vs random floor and ceiling;
B. the grid: exact a' + b answer for fast parts plus slow periods;
C. removal: which digit breaks when one part is set to the average;
D. each period swapped alone: shift of the supernode it should drive.
Operand a (b gives nearly the same numbers; see results/q2_analysis/).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MAIN, DESC, POST = (Path(f"results/{f}/runs") for f in ["q2_runs", "q2_descriptive", "q2_posthoc"])
OP = "a"


def load(folder, name):
    return pd.read_csv(folder / f"{name.replace(' ', '_')}.csv")


def mean(x):
    x = np.asarray(x, float)
    return x[np.isfinite(x)].mean()


fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
LD, MAG = "#2f6fb0", "#d9822b"

# A: swaps
ax = axes[0, 0]
bars = [
    ("fast\n(10, 5, 3.3, 2.5)\nmain run", load(MAIN, f"{OP} digit swap"), "last digit", LD,
     [load(MAIN, f"{OP} digit random {r:03d}") for r in range(100)], load(MAIN, f"{OP} digit ceiling")),
    ("fast + T = 2\npost hoc", load(POST, f"{OP} digit+2 swap main pairs"), "last digit", LD,
     [load(POST, f"{OP} digit+2 random {r:02d} main pairs") for r in range(20)], load(POST, f"{OP} ceiling main pairs")),
    ("slow\n(100, 50)\nmain run", load(MAIN, f"{OP} slow swap"), "magnitude", MAG,
     [load(MAIN, f"{OP} slow random {r:03d}") for r in range(100)], load(MAIN, f"{OP} slow ceiling")),
]
for i, (label, df, sn, color, rand, ceil) in enumerate(bars):
    ax.bar(i, mean(df[f"shift {sn}"]), color=color, width=0.6)
    p95 = np.percentile([mean(r[f"shift {sn}"]) for r in rand], 95)
    ax.plot([i - 0.35, i + 0.35], [p95, p95], color="black", lw=2)
    ax.plot([i - 0.35, i + 0.35], [mean(ceil[f"shift {sn}"])] * 2, color="black", lw=1, ls="--")
    ax.text(i, mean(df[f"shift {sn}"]) - 0.04, f"{mean(df[f'shift {sn}']):.2f}\n{sn}", ha="center", va="top",
            fontsize=9, color="white")
ax.plot([], [], color="black", lw=2, label="random directions, p95")
ax.plot([], [], color="black", lw=1, ls="--", label="whole-vector swap (ceiling)")
ax.set_xticks(range(3), [b[0] for b in bars], fontsize=8.5)
ax.set_ylim(0, 1.2)
ax.set_ylabel("shift of the supernode the swap should drive")
ax.set_title("A. Each supernode follows its own circles")
ax.legend(fontsize=8, loc="upper left")

# B: grid
ax = axes[0, 1]
rows, cols = ["10, 5, 2", "10, 5, 3.3, 2.5, 2"], ["none", "+ T = 100", "+ T = 50", "+ both"]
keys = [("10,5,2", ["", "+100", "+50", "+100+50"]), ("digit+2", ["", "+100", "+50", "+100+50"])]
grid = np.array([[load(POST, f"{OP} grid {f}{s} main pairs")["predicted"].mean() for s in ss] for f, ss in keys])
im = ax.imshow(grid, cmap="Blues", vmin=0.8, vmax=1.0)
for r in range(2):
    for c in range(4):
        ax.text(c, r, f"{grid[r, c]:.0%}", ha="center", va="center", fontsize=11,
                color="white" if grid[r, c] > 0.93 else "black")
ax.set_xticks(range(4), cols)
ax.set_yticks(range(2), rows)
ax.set_xlabel("slow periods swapped as well")
ax.set_ylabel("fast periods swapped")
ax.set_title("B. Units swap: exact a' + b answer (ceiling 100%)")

# C: removal
ax = axes[1, 0]
parts = [("T = 100", "period 100"), ("T = 50", "period 50"), ("T = 10", "period 10"), ("T = 5", "period 5"),
         ("T = 3.3", "period 3.33333"), ("T = 2.5", "period 2.5"), ("T = 2", "period 2"),
         ("all fast", "digit+2"), ("all slow", "slow")]
U = [(load(POST, f"{OP} remove {k}")["outcome"].isin(["U", "B"])).mean() for _, k in parts]
M = [(load(POST, f"{OP} remove {k}")["outcome"].isin(["M", "B"])).mean() for _, k in parts]
x = np.arange(len(parts))
ax.bar(x - 0.2, U, 0.4, color=LD, label="units digit wrong")
ax.bar(x + 0.2, M, 0.4, color=MAG, label="tens wrong")
rand = max((load(POST, f"{OP} remove random{k} {r:02d}")["outcome"] != "N").mean() for k in (9, 4) for r in range(5))
ax.axhline(rand, color="black", lw=1, ls=":")
ax.text(len(parts) - 0.5, rand + 0.02, f"random directions: at most {rand:.0%}", ha="right", fontsize=8)
ax.set_xticks(x, [p[0] for p in parts], fontsize=8.5)
ax.set_ylabel("share of prompts")
ax.set_ylim(0, 1)
ax.set_title("C. Removal: one part set to the average")
ax.legend(fontsize=8, loc="upper left")

# D: single periods swapped
ax = axes[1, 1]
fast = [("10", "10"), ("5", "5"), ("3.3", "3.33333"), ("2.5", "2.5"), ("2", "2")]
slow = [("100", "100"), ("50", "50")]
vals = [mean(load(DESC, f"{OP} period {k} swap")["shift last digit"]) for _, k in fast]
vals += [mean(load(DESC, f"{OP} period {k} swap")["shift magnitude"]) for _, k in slow]
ax.bar(range(7), vals, color=[LD] * 5 + [MAG] * 2)
for i, v in enumerate(vals):
    ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8.5)
ax.set_xticks(range(7), [f"T = {l}" for l, _ in fast + slow], fontsize=8.5)
ax.set_ylabel("shift (blue: last digit, orange: magnitude)")
ax.set_ylim(0, 1)
ax.set_title("D. Each period swapped alone")
ax.text(3.0, 0.92, "fast periods: a' with the same tens, units differing by an odd amount other than 5\n"
        "slow periods: a' with the same units, tens differing by anything but 5", ha="center", fontsize=7.5, color="#444444")

fig.text(0.5, 0.012, f"Llama-3.2-1B, operand {OP}, 998 evaluation prompts; swaps at every point from the embedding to after "
         "layer 10. Shift: 0 = unchanged, 1 = like the real a' + b run.\nA: the main run's random directions were drawn "
         "from the number space minus the fast and slow groups, so they could include T = 2 and the line; the post-hoc "
         "ones exclude every circle and the line.", ha="center", fontsize=8, color="#444444")
fig.tight_layout(rect=(0, 0.045, 1, 1))
fig.savefig("figures/q2_results.png", dpi=170)
print("saved figures/q2_results.png")
