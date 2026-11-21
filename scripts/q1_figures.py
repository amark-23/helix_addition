"""Q1 figure: outcomes when each supernode is switched off at "=".

figures/q1_suppression.png. Left: the preregistered test (intended digit vs the
other digit, against the p95 of 100 layer-matched random groups). Right: outcome
shares on all 998 prompts for every supernode.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

prim = pd.read_csv("results/q1_analysis/primary_lenient.csv").set_index("group")
rates = pd.read_csv("results/q1_analysis/rates_all_lenient.csv").set_index("run")
LD, MAG, BOTH = "#2f6fb0", "#d9822b", "#7a4fa3"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw={"width_ratios": [1, 1.4]})

ld, mg = prim.loc["last digit"], prim.loc["magnitude"]
bars = [("last digit off\nunits wrong (U)\nno-carry prompts", ld["effect (U)"], ld["random p95 (U)"], LD),
        ("last digit off\ntens wrong (M)\nno-carry prompts", ld["other (M)"], ld["random p95 (M)"], MAG),
        ("magnitude off\ntens wrong (M)\nall prompts", mg["effect (M)"], mg["random p95 (M)"], MAG),
        ("magnitude off\nunits wrong (U)\nall prompts", mg["other (U)"], mg["random p95 (U)"], LD)]
for i, (label, v, p95, c) in enumerate(bars):
    ax1.bar(i, v, color=c, width=0.6)
    ax1.plot([i - 0.35, i + 0.35], [p95, p95], color="black", lw=2)
    ax1.text(i, v + 0.015, f"{v:.0%}", ha="center", fontsize=9)
ax1.plot([], [], color="black", lw=2, label="random groups, p95")
ax1.set_xticks(range(4), [b[0] for b in bars], fontsize=8)
ax1.set_ylim(0, 0.6)
ax1.set_ylabel("share of prompts")
ax1.set_title("Preregistered test (section 7.1)")
ax1.legend(fontsize=8)

groups = ["last digit", "magnitude", "comparison", "mod 100", "mixed family"]
bottom = np.zeros(len(groups))
for o, c, name in [("U", LD, "units only (U)"), ("M", MAG, "tens only (M)"), ("B", BOTH, "both (B)")]:
    v = rates.loc[groups, o].to_numpy()
    ax2.bar(range(len(groups)), v, bottom=bottom, color=c, label=name, width=0.6)
    bottom += v
for i, t in enumerate(bottom):
    ax2.text(i, t + 0.015, f"{t:.0%}", ha="center", fontsize=9)
ax2.set_xticks(range(len(groups)), groups, fontsize=9)
ax2.set_ylim(0, 1)
ax2.set_ylabel("share of prompts with a changed answer")
ax2.set_title("Every supernode switched off, all prompts")
ax2.legend(fontsize=8)

fig.text(0.5, 0.01, "Llama-3.2-1B, 998 evaluation prompts; features switched off at \"=\" (set to 0).",
         ha="center", fontsize=8.5, color="#444444")
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig("figures/q1_suppression.png", dpi=170)
print("saved figures/q1_suppression.png")
