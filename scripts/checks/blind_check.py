"""Prepare a new blind check of the labels.

python scripts/checks/blind_check.py NAME [N_PER_ROLE]

Samples N_PER_ROLE features per role from the final labels (default 15),
leaving out every feature already shown in an earlier check, in the taxonomy
examples or in the manual review. The plots are titled with a number only.

Outputs: figures/blind_check_NAME/page_*.png, results/blind_check_NAME/
sheet.csv (to fill in by hand) and key.csv (the automatic labels; not to be
opened before the sheet is done). Compare with scripts/checks/blind_compare.py.
"""
import sys
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from helixadd.plots import draw, operand_grid

name = sys.argv[1]
n_per_role = int(sys.argv[2]) if len(sys.argv) > 2 else 15
OUT, FIG = Path(f"results/blind_check_{name}"), Path(f"figures/blind_check_{name}")
if OUT.exists():
    raise SystemExit(f"{OUT} already exists; not overwriting.")

labels = pd.read_csv("results/feature_labels/data.csv")
data = np.load("results/eq_activations/data.npz")
shown = [pd.read_csv(f)[["layer", "feature"]] for f in
         glob("results/blind_check*/key.csv") + ["figures/taxonomy/examples.csv",
                                                  "results/manual_review/tens_lookups.csv"]]
pool = labels.merge(pd.concat(shown).drop_duplicates(), how="left", indicator=True)
pool = pool[(pool["_merge"] == "left_only") & ~pool["role"].isin(["unclear", "mixed"])]

sample = (pd.concat([g.sample(min(n_per_role, len(g)), random_state=0) for _, g in pool.groupby("role")])
          .sample(frac=1, random_state=0).reset_index(drop=True))
sample["id"] = np.arange(1, len(sample) + 1)

OUT.mkdir(parents=True)
FIG.mkdir(parents=True, exist_ok=True)
sample[["id", "layer", "feature", "role", "family", "best_var", "best_r2", "source"]].to_csv(OUT / "key.csv", index=False)
pd.DataFrame({"id": sample["id"], "my_role": "", "my_family": "", "note": ""}).to_csv(OUT / "sheet.csv", index=False)

for page, start in enumerate(range(0, len(sample), 20), start=1):
    fig, axes = plt.subplots(4, 5, figsize=(15, 12.5))
    for ax in axes.flat:
        ax.axis("off")
    for ax, (_, f) in zip(axes.flat, sample.iloc[start:start + 20].iterrows()):
        ax.axis("on")
        draw(ax, operand_grid(data, f["layer"], f["feature"]), f"#{f['id']}")
    fig.supxlabel("b")
    fig.supylabel("a")
    fig.tight_layout()
    fig.savefig(FIG / f"page_{page}.png", dpi=80)
    plt.close(fig)
print(f"{len(sample)} features on {page} pages in {FIG}/")
