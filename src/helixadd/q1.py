"""Q1 runs: switch off each planned group on the evaluation prompts and record the answers.

The plan (preregistration 4.1, 4.2, 6, 8.2, 8.3):
- clean (nothing switched off)
- primary: last digit, magnitude
- variants: last digit without max/min mod 10, magnitude without the hand-confirmed lookups
- relaxed: last digit, magnitude (relaxed labels)
- secondary: mod 100, comparison, mixed family
- 100 layer-matched random groups for each primary, variant and relaxed group
Every run is saved as its own CSV in results/q1_runs/runs/ and skipped if it
exists, so an interrupted session resumes where it stopped.
"""
import json
from pathlib import Path

import pandas as pd

from helixadd.groups import load_labels, random_group, supernode
from helixadd.prompts import is_carry, make_prompt, sample_pairs
from helixadd.readout import outcome, parse_answer

FOLDER = Path("results/q1_runs")
N_PROMPTS, PROMPT_SEED, N_RANDOM = 1000, 0, 100
COMPARED = ["last digit", "magnitude", "last digit without max-min mod 10",
            "magnitude without hand lookups", "relaxed last digit", "relaxed magnitude"]
WATCHED = ["last digit", "magnitude"]


def eval_pairs(clean_path="results/clean_accuracy/data.csv"):
    """sample_pairs(1000, seed=0), minus the pairs the model answers wrongly in the clean run."""
    clean = pd.read_csv(clean_path)
    wrong = set(zip(clean.loc[~clean["correct"], "a"], clean.loc[~clean["correct"], "b"]))
    return [(a, b) for a, b in sample_pairs(N_PROMPTS, seed=PROMPT_SEED) if (a, b) not in wrong]


def real_groups(labels, relaxed):
    """The named groups of the plan, each a DataFrame (layer, feature)."""
    ld, mag = supernode(labels, "last digit"), supernode(labels, "magnitude")
    ld_full = ld.merge(labels, on=["layer", "feature"])
    mag_full = mag.merge(labels, on=["layer", "feature"])
    return {
        "clean": ld.iloc[:0],
        "last digit": ld,
        "magnitude": mag,
        "last digit without max-min mod 10": ld_full[ld_full["role"] != "comparison"][["layer", "feature"]],
        "magnitude without hand lookups": mag_full[mag_full["role"] != "lookup table"][["layer", "feature"]],
        "relaxed last digit": supernode(relaxed, "last digit"),
        "relaxed magnitude": supernode(relaxed, "magnitude"),
        "mod 100": supernode(labels, "mod 100"),
        "comparison": supernode(labels, "comparison"),
        "mixed family": supernode(labels, "mixed"),
    }


def plan(labels=None, relaxed=None, n_random=N_RANDOM):
    """Every run as {name: group}. Random groups are named 'random | <group> | <k>'."""
    labels = load_labels() if labels is None else labels
    relaxed = load_labels("results/feature_labels_relaxed/data.csv") if relaxed is None else relaxed
    runs = real_groups(labels, relaxed)
    for i, name in enumerate(COMPARED):
        for k in range(n_random):
            runs[f"random | {name} | {k:03d}"] = random_group(labels, runs[name], seed=1000 * (i + 1) + k)
    return runs


def file_name(name):
    return name.replace(" | ", "__").replace(" ", "_") + ".csv"


def run_one(model, group, pairs, watch, batch=64):
    """One run: the group switched off on every pair. One row per prompt."""
    from helixadd import intervene      # needs torch: imported here so the plan can be built without it

    g = intervene.by_layer(group["layer"], group["feature"])
    rows = []
    for i in range(0, len(pairs), batch):
        chunk = pairs[i:i + batch]
        texts, logits, sums = intervene.suppress_generate(model, [make_prompt(a, b) for a, b in chunk], g,
                                                          watch=watch)
        for j, (a, b) in enumerate(chunk):
            pred = parse_answer(texts[j])
            answer_id = model.tokenizer.convert_tokens_to_ids(str(a + b))
            row = {"a": a, "b": b, "carry": is_carry(a, b), "correct": a + b, "raw": texts[j], "pred": pred,
                   "outcome": outcome(pred, a + b), "logit_correct": float(logits[j, answer_id])}
            for name in watch:
                row[f"S {name}"] = float(sums[name][j])
            rows.append(row)
    return pd.DataFrame(rows)


def run_all(model, runs=None, pairs=None, folder=FOLDER):
    """Run every planned group not saved yet. Safe to call again after an interruption."""
    from helixadd import intervene

    runs = plan() if runs is None else runs
    pairs = eval_pairs() if pairs is None else pairs
    params = {"n_prompts": N_PROMPTS, "prompt_seed": PROMPT_SEED, "n_eval": len(pairs), "n_random": N_RANDOM,
              "compared": COMPARED, "watched": WATCHED, "position": intervene.EQ, "value": 0.0,
              "labels": "results/feature_labels", "relaxed": "results/feature_labels_relaxed"}
    (folder / "runs").mkdir(parents=True, exist_ok=True)
    pfile = folder / "params.json"
    if pfile.exists() and json.loads(pfile.read_text()) != params:
        raise ValueError(f"{folder} holds runs with different parameters")
    pfile.write_text(json.dumps(params, indent=2))
    pd.DataFrame([(n, len(g)) for n, g in runs.items()], columns=["run", "n_features"]).to_csv(
        folder / "plan.csv", index=False)

    watch = {name: intervene.by_layer(runs[name]["layer"], runs[name]["feature"]) for name in WATCHED}
    todo = [n for n in runs if not (folder / "runs" / file_name(n)).exists()]
    print(f"{len(runs) - len(todo)} of {len(runs)} runs already saved; {len(todo)} to go")
    for k, name in enumerate(todo):
        run_one(model, runs[name], pairs, watch).to_csv(folder / "runs" / file_name(name), index=False)
        if k % 25 == 0:
            print(k, "of", len(todo), name)
