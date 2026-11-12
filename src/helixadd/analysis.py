"""Q1 analysis: the decision rules of preregistration section 7, applied as written."""
from pathlib import Path

import numpy as np
import pandas as pd

from helixadd import q1
from helixadd.readout import outcome, parse_answer_strict

CATS = ["N", "U", "M", "B", "X"]
RUNS = Path("results/q1_runs/runs")


def load_runs(names=None, strict=False):
    """All runs as one table with a 'run' column. strict=True re-reads every
    answer with the strict rule (section 5, sensitivity check)."""
    names = list(q1.plan()) if names is None else names
    parts = []
    for name in names:
        df = pd.read_csv(RUNS / q1.file_name(name), keep_default_na=False, na_values=[""])
        df["run"] = name
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    if strict:
        df["outcome"] = [outcome(parse_answer_strict(str(r)), c) for r, c in zip(df["raw"], df["correct"])]
    return df


def rates(df, subset="all"):
    """Share of each outcome per run, on all, carry or no-carry prompts."""
    if subset == "carry":
        df = df[df["carry"]]
    elif subset == "no-carry":
        df = df[~df["carry"]]
    r = pd.crosstab(df["run"], df["outcome"], normalize="index")
    return r.reindex(columns=CATS, fill_value=0.0)


def random_names(group):
    return [f"random | {group} | {k:03d}" for k in range(q1.N_RANDOM)]


def bootstrap_rate(df, cat, n=1000, seed=0):
    """95% interval of the share of `cat` in one run, resampling prompts."""
    hits = (df["outcome"] == cat).to_numpy()
    rng = np.random.default_rng(seed)
    means = hits[rng.integers(0, len(hits), (n, len(hits)))].mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def primary_rule(df, group, kind):
    """Section 7.1. kind 'last digit': no-carry prompts, U above random p95 and M
    at most random p95. kind 'magnitude': all prompts, M above p95, U at most p95."""
    subset = "no-carry" if kind == "last digit" else "all"
    effect, other = ("U", "M") if kind == "last digit" else ("M", "U")
    r = rates(df[df["run"].isin([group] + random_names(group))], subset)
    rand = r.loc[random_names(group)]
    real = r.loc[group]
    p95_eff, p95_oth = np.percentile(rand[effect], 95), np.percentile(rand[other], 95)
    sub = df[(df["run"] == group) & (~df["carry"] if subset == "no-carry" else True)]
    return {
        "group": group, "prompts": subset, "n_prompts": len(sub),
        f"effect ({effect})": real[effect], f"effect CI": tuple(bootstrap_rate(sub, effect).round(3)),
        f"random p95 ({effect})": p95_eff,
        f"other ({other})": real[other], f"other CI": tuple(bootstrap_rate(sub, other).round(3)),
        f"random p95 ({other})": p95_oth,
        "effect passes": real[effect] > p95_eff, "other passes": real[other] <= p95_oth,
        "passes": (real[effect] > p95_eff) and (real[other] <= p95_oth),
    }


def internal_change(df, run, watched):
    """Section 7.2: mean over prompts of |S_after - S_before| / S_before for a watched group."""
    col = f"S {watched}"
    before = df[df["run"] == "clean"].set_index(["a", "b"])[col]
    after = df[df["run"] == run].set_index(["a", "b"])[col]
    return float(((after - before).abs() / before).mean())


def internal_rule(df, group, watched):
    real = internal_change(df, group, watched)
    rand = [internal_change(df, r, watched) for r in random_names(group)]
    p95 = np.percentile(rand, 95)
    return {"suppressed": group, "watched": watched, "change": real, "random p95": p95,
            "random median": float(np.median(rand)), "passes": real <= p95}


def logit_drop(df, run):
    """Mean drop in the correct answer's logit at the first generated position (clean - run)."""
    clean = df[df["run"] == "clean"].set_index(["a", "b"])["logit_correct"]
    after = df[df["run"] == run].set_index(["a", "b"])["logit_correct"]
    return float((clean - after).mean())
