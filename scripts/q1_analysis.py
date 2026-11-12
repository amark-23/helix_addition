"""Q1 results: the decision rules of preregistration section 7, applied as written.

Output: printed report and tables in results/q1_analysis/.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from helixadd import analysis as A

OUT = Path("results/q1_analysis")
OUT.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 200)

for rule_name, strict in [("lenient (primary)", False), ("strict (sensitivity)", True)]:
    df = A.load_runs(strict=strict)
    tag = "strict" if strict else "lenient"
    print(f"\n==================== readout rule: {rule_name} ====================")

    # outcome rates of the real groups
    real = ["clean", "last digit", "magnitude", "last digit without max-min mod 10", "magnitude without hand lookups",
            "relaxed last digit", "relaxed magnitude", "mod 100", "comparison", "mixed family"]
    tables = {s: A.rates(df[df["run"].isin(real)], s).loc[real] for s in ["all", "no-carry", "carry"]}
    for s, t in tables.items():
        t.round(3).to_csv(OUT / f"rates_{s}_{tag}.csv")
        print(f"\noutcome rates, {s} prompts:\n{t.round(3).to_string()}")
    drops = pd.Series({g: A.logit_drop(df, g) for g in real[1:]}, name="mean logit drop")
    print(f"\nmean drop in the correct answer's logit:\n{drops.round(2).to_string()}")

    # 7.1 primary rule (and the same rule for the robustness variants, 4.2)
    rows = [A.primary_rule(df, g, "last digit") for g in
            ["last digit", "last digit without max-min mod 10", "relaxed last digit"]]
    rows += [A.primary_rule(df, g, "magnitude") for g in
             ["magnitude", "magnitude without hand lookups", "relaxed magnitude"]]
    prim = pd.DataFrame(rows)
    prim.to_csv(OUT / f"primary_{tag}.csv", index=False)
    print("\n7.1 primary rule (first two rows of each block are what the rule is about):")
    print(prim.round(3).to_string(index=False))
    both = prim.set_index("group").loc[["last digit", "magnitude"], "passes"]
    print(f"\nPRIMARY RULE ({rule_name}): {'PASS' if both.all() else 'FAIL'}  "
          f"(last digit: {both['last digit']}, magnitude: {both['magnitude']})")

    if strict:
        continue
    # 7.2 internal separation
    internal = pd.DataFrame([A.internal_rule(df, "last digit", "magnitude"),
                             A.internal_rule(df, "magnitude", "last digit")])
    internal.to_csv(OUT / "internal.csv", index=False)
    print(f"\n7.2 internal separation:\n{internal.round(3).to_string(index=False)}")
    print(f"SECONDARY RULE: {'PASS' if internal['passes'].all() else 'FAIL'}")

    # 7.4 carry prediction
    ld = df[df["run"] == "last digit"]
    b_carry, b_nocarry = (ld[ld["carry"]]["outcome"] == "B").mean(), (ld[~ld["carry"]]["outcome"] == "B").mean()
    rng = np.random.default_rng(0)
    c = (ld[ld["carry"]]["outcome"] == "B").to_numpy()
    n = (ld[~ld["carry"]]["outcome"] == "B").to_numpy()
    diffs = [c[rng.integers(0, len(c), len(c))].mean() - n[rng.integers(0, len(n), len(n))].mean() for _ in range(1000)]
    print(f"\n7.4 carry prediction: B rate with last digit off, carry {b_carry:.3f} vs no-carry {b_nocarry:.3f}; "
          f"difference 95% CI {np.percentile(diffs, [2.5, 97.5]).round(3)}")
