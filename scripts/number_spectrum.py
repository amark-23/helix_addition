"""Q2 step 1: spectrum of the number representations across layers.

For each of the 17 points (embedding, after each layer) and each operand
position: the share of the variance in the straight-line part, then the
spectrum (helixadd.helix.spectrum) of what is left after removing that line.

Output: figures/number_spectrum.png, results/number_spectrum/data.csv.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from helixadd.helix import spectrum

data = np.load("results/number_acts/data.npz")
n = np.arange(100)
t = (n - n.mean())[:, None]
POINTS = ["emb"] + [f"L{L}" for L in range(16)]
SHOWN = {"T=100": 1, "T=50": 2, "T=10": 10, "T=5": 20, "T=2": 50}   # period: frequency k


def split_line(X):
    """Share of the variance in the straight line, and X with the line removed."""
    X = X - X.mean(axis=0)
    line = t @ (t.T @ X) / (t.T @ t).item()
    return (line ** 2).sum() / (X ** 2).sum(), X - line


rows, maps = [], {}
for pos in ["a", "b"]:
    spectra = []
    for i, point in enumerate(POINTS):
        share, rest = split_line(data[pos][i])
        s = spectrum(rest)
        spectra.append(s)
        rows.append({"position": pos, "point": point, "linear share": share,
                     **{f"k={k}": s[k] for k in range(51)}})
    maps[pos] = np.array(spectra)
df = pd.DataFrame(rows)
Path("results/number_spectrum").mkdir(parents=True, exist_ok=True)
df.to_csv("results/number_spectrum/data.csv", index=False)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1.25, 1]})

# left: full spectrum of a at every point (line removed)
im = ax1.imshow(maps["a"][:, 1:].T, aspect="auto", origin="lower", cmap="viridis",
                extent=(-0.5, 16.5, 0.5, 50.5), vmin=0, vmax=0.15)
for name, k in SHOWN.items():
    ax1.text(16.8, k, name, va="center", fontsize=8)
ax1.set_xticks(range(17), POINTS, fontsize=8)
ax1.set_xlabel("point in the model")
ax1.set_ylabel("frequency k (repeats over 0-99; period = 100 / k)")
ax1.set_title("Spectrum of a's representation, straight line removed")
fig.colorbar(im, ax=ax1, label="share of the remaining variance", pad=0.09)

# right: the chosen periods and the linear part, a (solid) and b (dashed)
colors = {"linear": "black", "T=100": "#4a8f3c", "T=50": "#9cc78f", "T=10": "#2f6fb0", "T=5": "#7fb2e5", "T=2": "#d9822b"}
for pos, style in [("a", "-"), ("b", "--")]:
    d = df[df["position"] == pos]
    ax2.plot(range(17), d["linear share"], style, color=colors["linear"], label="linear part" if pos == "a" else None)
    for name, k in SHOWN.items():
        ax2.plot(range(17), d[f"k={k}"], style, color=colors[name], label=name if pos == "a" else None)
ax2.axhline(0.02, color="grey", lw=0.8, ls=":")
ax2.text(0, 0.024, "flat level (1/50)", fontsize=8, color="grey")
ax2.set_xticks(range(17), POINTS, fontsize=8)
ax2.set_xlabel("point in the model")
ax2.set_ylabel("share of variance")
ax2.set_title("Linear part and chosen periods (solid a, dashed b)")
ax2.legend(fontsize=8, ncol=2)

fig.text(0.5, 0.01, "Linear part: share of all variance. Periods: share of the variance left after removing the line. "
         "Llama-3.2-1B, residual stream on the number token, few-shot prompts.", ha="center", fontsize=8.5, color="#444444")
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig("figures/number_spectrum.png", dpi=180)
print("saved figures/number_spectrum.png")

# a single representative spectrum: a after layer 10, where the periods are strongest
POINT = "L10"
s = df[(df["position"] == "a") & (df["point"] == POINT)][[f"k={k}" for k in range(1, 51)]].to_numpy()[0]
fig, ax = plt.subplots(figsize=(10, 4.2))
ks = np.arange(1, 51)
named = {1: "T=100", 2: "T=50", 10: "T=10", 20: "T=5", 30: "T=3.3", 40: "T=2.5", 50: "T=2"}
ax.bar(ks, s, color=["#2f6fb0" if k in named else "#bbbbbb" for k in ks], width=0.8)
for k, name in named.items():
    x, ha = {1: (1.3, "right"), 2: (1.7, "left")}.get(k, (k, "center"))
    ax.text(x, s[k - 1] + 0.003, name, ha=ha, fontsize=8.5)
ax.axhline(1 / 50, color="grey", lw=0.8, ls=":")
ax.text(26, 1 / 50 + 0.002, "flat level (1/50)", fontsize=8, color="grey")
ax.set_xlabel("frequency k (repeats over 0-99; period T = 100 / k)")
ax.set_ylabel("share of the variance (line removed)")
ax.set_title(f"Spectrum of a's representation after layer {POINT[1:]}")
ax.set_xlim(-4, 51.5)
fig.text(0.5, 0.01, f"Llama-3.2-1B, residual stream on the a token, a = 0..99; straight-line part ({df[(df['position'] == 'a') & (df['point'] == POINT)]['linear share'].iloc[0]:.0%} of the variance) removed first.",
         ha="center", fontsize=8.5, color="#444444")
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig("figures/number_spectrum_L10.png", dpi=180)
print("saved figures/number_spectrum_L10.png")
