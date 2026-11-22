"""Q2 step 1 figure: how much of the real vector's effect a fitted vector keeps.

figures/helix_patching.png. Kantamneni and Tegmark's setup: run a' + b, put a
vector for the clean a on its token at one point, read the clean answer's logit
difference; share of the real vector's (layer patch) value. Fit on 80 numbers,
patched only for the other 20.
"""
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("results/helix_patching_corrupted/data.csv")
POINTS = list(range(12))
names = ["embedding"] + [f"L{i}" for i in range(11)]
style = {"helix": "#2f6fb0", "circles only": "#4a8f3c", "line only": "#d9822b", "pca 9": "#888888"}

fig, ax = plt.subplots(figsize=(8.5, 4.4))
for pos, ls in [("a", "-"), ("b", "--")]:
    d = df[(df["position"] == pos) & df["point"].isin(POINTS)].pivot(index="point", columns="method", values="logit difference")
    for m, c in style.items():
        ax.plot(POINTS, d[m] / d["layer patch"], ls, color=c, marker="o", ms=3, label=m if pos == "a" else None)
ax.set_xticks(POINTS, names, fontsize=8)
ax.set_ylim(0, 1)
ax.set_xlabel("point where the vector is replaced (after layer)")
ax.set_ylabel("share of the real vector's effect")
ax.set_title("Fitted vectors patched into a' + b (solid a, dashed b)")
ax.legend(fontsize=8, ncol=2, loc="lower right")
fig.text(0.5, 0.01, "Fits on 80 numbers, patched for the other 20. After layer 10 the operand token has little effect on the answer.",
         ha="center", fontsize=8, color="#444444")
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig("figures/helix_patching.png", dpi=170)
print("saved figures/helix_patching.png")
