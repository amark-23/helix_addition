"""Q2 runs: swap circle groups on an operand token and measure the Q1 supernodes at "="
(preregistration section 9).

For each operand (a, b) and each swap kind (digit, slow), on the Q1 evaluation prompts:
- the circle swap: the operand's vector takes a''s part along the group's
  directions, at every point from the embedding to after layer 10;
- the ceiling: the whole vector of a' at the same points;
- N_RANDOM floor runs: random number directions of the same size, swapped the same way.
Plus the reference runs without any swap: the clean prompts a + b and the
real target prompts a' + b. Every run is saved as its own CSV in
results/q2_runs/runs/ (resumable); the supernode activations of the
reference runs in results/q2_runs/reference/.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from helixadd.helix import (A_POS, B_POS, Q2_PERIODS, circle_directions, fit, point_name)
from helixadd.prompts import EXCLUDED, is_carry, make_prompt
from helixadd.readout import outcome, parse_answer

FOLDER = Path("results/q2_runs")
POINTS = list(range(12))                       # embedding .. after layer 10 (9.3)
GROUPS = {"digit": [10, 5, 10 / 3, 2.5], "slow": [100, 50]}
N_RANDOM = 100
MAX_OVERLAP = 0.1                              # squared cosine above which slow is made orthogonal to digit (9.2)
POS = {"a": A_POS, "b": B_POS}


def orthonormal(M, tol=1e-6):
    """Orthonormal rows spanning the rows of M."""
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    return Vt[S > tol * S[0]]


def point_directions(data, position):
    """For each point: digit and slow directions, the rest of the number space, and the vectors.

    rest: orthonormal directions spanning the 100 number vectors (centered), with
    the digit and slow directions removed; the floor's random directions come from it.
    """
    out = []
    for i in POINTS:
        X = data[position][i]
        mean, C = fit(np.arange(100), X, Q2_PERIODS)
        digit = circle_directions(C, GROUPS["digit"])
        slow = circle_directions(C, GROUPS["slow"])
        overlap = float((np.linalg.svd(digit @ slow.T, compute_uv=False) ** 2).max())
        if overlap > MAX_OVERLAP:
            slow = orthonormal(slow - slow @ digit.T @ digit)
        span = orthonormal(X - mean)
        both = orthonormal(np.vstack([digit, slow]))      # digit and slow need not be orthogonal
        rest = orthonormal(span - span @ both.T @ both)
        out.append({"digit": digit, "slow": slow, "rest": rest, "X": X, "overlap": overlap})
    return out


def random_directions(rest, k, rng):
    """k orthonormal random directions inside the space spanned by `rest`."""
    return orthonormal(rng.normal(size=(k, rest.shape[0])) @ rest)


def swap_pairs(kind, position, pairs, seed=0):
    """(a, b, a2, b2) per prompt: (a2, b2) is the target a' + b (or a + b').

    digit: same tens, other units, same carry status. slow: same units, other tens.
    Prompts with no qualifying partner are dropped.
    """
    rng = np.random.default_rng(seed)
    out = []
    for a, b in pairs:
        x = a if position == "a" else b
        if kind == "digit":
            options = [10 * (x // 10) + u for u in range(10) if u != x % 10]
        else:
            options = [10 * t + x % 10 for t in range(10) if t != x // 10]
        ok = []
        for y in options:
            a2, b2 = (y, b) if position == "a" else (a, y)
            if (a2, b2) in EXCLUDED or (b2, a2) in EXCLUDED:
                continue
            if kind == "digit" and is_carry(a2, b2) != is_carry(a, b):
                continue
            ok.append((a2, b2))
        if ok:
            a2, b2 = ok[rng.integers(len(ok))]
            out.append((a, b, a2, b2))
    return out


def supernode_groups():
    """The two Q1 supernodes as {layer: feature ids}."""
    from helixadd.groups import load_labels, supernode
    from helixadd.intervene import by_layer

    labels = load_labels()
    return {name: by_layer(g["layer"], g["feature"])
            for name, g in [("last digit", supernode(labels, "last digit")), ("magnitude", supernode(labels, "magnitude"))]}


def make_read_hooks(model, watch, store):
    """Read-only hooks: each supernode's feature activations at "=", {name: (batch, n_features)}.
    Features are in layer order, then in the order of the group."""
    from helixadd.intervene import EQ

    layers = sorted({layer for g in watch.values() for layer in g})
    hooks = []
    for layer in layers:
        here = {name: g[layer].to(model.cfg.device) for name, g in watch.items() if layer in g}

        def read(x, hook, layer=layer, here=here):
            acts = model.transcoders.encode_layer(x[:, EQ, :], layer)
            for name, feats in here.items():
                store.setdefault(name, []).append(acts[:, feats].float())
            return x

        hooks.append((f"blocks.{layer}.{model.feature_input_hook}", read))
    return hooks


def make_swap_hooks(model, pos, sources, directions):
    """Hooks that swap at every point: sources[i] (batch, d) is a''s vector at point i,
    directions[i] (k, d) the directions to swap there, or None for the whole vector."""
    import torch

    hooks = []
    for i in POINTS:
        src = torch.tensor(sources[i], dtype=torch.float32, device=model.cfg.device)
        D = None if directions is None else torch.tensor(directions[i], dtype=torch.float32, device=model.cfg.device)

        def swap(act, hook, src=src, D=D):
            x = act[:, pos, :].float()
            new = src if D is None else x - ((x - src) @ D.T) @ D
            act[:, pos, :] = new.to(act.dtype)
            return act

        hooks.append((point_name(i), swap))
    return hooks


def generate(model, prompts, hooks, watch, max_new_tokens=4):
    """Greedy generation with the hooks; returns the texts and each supernode's activations
    at "=" during the first step ({name: (batch, n_features)} as numpy)."""
    import torch

    from helixadd.model import tokenize

    store = {}
    tokens = tokenize(model, prompts)
    n_prompt = tokens.shape[1]
    first = None
    with torch.no_grad(), model.hooks(fwd_hooks=hooks + make_read_hooks(model, watch, store)):
        for step in range(max_new_tokens):
            store.clear()
            logits = model(tokens)[:, -1, :]
            if step == 0:
                first = {name: torch.cat(v, dim=1).cpu().numpy() for name, v in store.items()}
            tokens = torch.cat([tokens, logits.argmax(dim=-1)[:, None]], dim=1)
    texts = [model.tokenizer.decode(row) for row in tokens[:, n_prompt:]]
    return texts, first


def reference(model, quads, watch, batch=64):
    """Supernode activations in the clean (a + b) and target (a' + b) runs, no swap."""
    out = {}
    for which, make in [("clean", lambda q: make_prompt(q[0], q[1])), ("target", lambda q: make_prompt(q[2], q[3]))]:
        parts = {}
        for i in range(0, len(quads), batch):
            _, s = generate(model, [make(q) for q in quads[i:i + batch]], [], watch, max_new_tokens=1)
            for name, v in s.items():
                parts.setdefault(name, []).append(v)
        out[which] = {name: np.concatenate(v) for name, v in parts.items()}
    return out


def run_one(model, quads, position, dirs, directions, ref, watch, batch=64):
    """One run: swap on every prompt; one row per prompt with outcomes, shift and change."""
    from helixadd.helix import change, off_target, shift

    rows, acts = [], {}
    pos = POS[position]
    for i in range(0, len(quads), batch):
        chunk = quads[i:i + batch]
        partner = [q[2] if position == "a" else q[3] for q in chunk]
        sources = [dirs[p]["X"][partner] for p in range(len(POINTS))]
        hooks = make_swap_hooks(model, pos, sources, directions)
        texts, s = generate(model, [make_prompt(a, b) for a, b, _, _ in chunk], hooks, watch)
        for name, v in s.items():
            acts.setdefault(name, []).append(v)
        for j, (a, b, a2, b2) in enumerate(chunk):
            pred = parse_answer(texts[j])
            rows.append({"a": a, "b": b, "a2": a2, "b2": b2, "carry": is_carry(a, b), "raw": texts[j],
                         "pred": pred, "outcome": outcome(pred, a + b), "predicted": pred == a2 + b2})
    df = pd.DataFrame(rows)
    for name in watch:
        s_swap = np.concatenate(acts[name])
        with np.errstate(divide="ignore", invalid="ignore"):
            df[f"shift {name}"] = shift(s_swap, ref["clean"][name], ref["target"][name])
        df[f"change {name}"] = change(s_swap, ref["clean"][name])
        df[f"off-target {name}"] = off_target(s_swap, ref["clean"][name], ref["target"][name])
    return df


def plan(n_random=N_RANDOM):
    """(kind, position, run name, which directions): the swap, the ceiling and the floor runs."""
    runs = []
    for position in ["a", "b"]:
        for kind in GROUPS:
            runs.append((kind, position, f"{position} {kind} swap", "swap"))
            runs.append((kind, position, f"{position} {kind} ceiling", "ceiling"))
            runs += [(kind, position, f"{position} {kind} random {r:03d}", r) for r in range(n_random)]
    return runs


def run_all(model, data, pairs, seed=0, folder=FOLDER, n_random=N_RANDOM):
    """Every run of plan(), skipping the ones already saved."""
    import json

    folder = Path(folder)
    (folder / "runs").mkdir(parents=True, exist_ok=True)
    (folder / "reference").mkdir(parents=True, exist_ok=True)
    (folder / "params.json").write_text(json.dumps({
        "points": POINTS, "groups": GROUPS, "periods": Q2_PERIODS, "n_random": n_random, "seed": seed, "n_prompts": len(pairs),
        "max_overlap": MAX_OVERLAP, "prompts": "q1.eval_pairs()", "activations": "results/number_acts"}, indent=2))
    watch = supernode_groups()
    dirs = {p: point_directions(data, p) for p in ["a", "b"]}
    for p in dirs:
        print(p, "digit-slow overlap per point:", [round(d["overlap"], 3) for d in dirs[p]])

    for kind, position, name, which in plan(n_random):
        f = folder / "runs" / f"{name.replace(' ', '_')}.csv"
        if f.exists():
            continue
        quads = swap_pairs(kind, position, pairs, seed)
        ref_file = folder / "reference" / f"{position}_{kind}.npz"
        if ref_file.exists():
            z = np.load(ref_file)
            ref = {w: {n: z[f"{w}|{n}"] for n in watch} for w in ["clean", "target"]}
        else:
            ref = reference(model, quads, watch)
            np.savez_compressed(ref_file, **{f"{w}|{n}": v for w, d in ref.items() for n, v in d.items()})
        d = dirs[position]
        if which == "swap":
            directions = [p[kind] for p in d]
        elif which == "ceiling":
            directions = None
        else:
            rng = np.random.default_rng(1000 * (position == "b") + 100 * (kind == "slow") + which)
            k = len(d[0][kind])
            directions = [random_directions(p["rest"], k, rng) for p in d]
        run_one(model, quads, position, d, directions, ref, watch).to_csv(f, index=False)
        print(name, "saved")


def summary(folder=FOLDER):
    """Mean of every measure per run (quick look; the decision rules are applied separately)."""
    rows = []
    for f in sorted((Path(folder) / "runs").glob("*.csv")):
        df = pd.read_csv(f)
        row = {"run": f.stem.replace("_", " "), "n": len(df), "predicted": df["predicted"].mean()}
        row.update({c: df[c].mean() for c in df.columns if c.startswith(("shift", "change", "off-target"))})
        row.update({f"rate {o}": (df["outcome"] == o).mean() for o in "NUMBX"})
        rows.append(row)
    return pd.DataFrame(rows)
