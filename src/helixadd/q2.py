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


def make_swap_hooks(model, pos, sources, directions, points=None):
    """Hooks that swap at every point: sources[i] (batch, d) is a''s vector at point i,
    directions[i] (k, d) the directions to swap there, or None for the whole vector.
    points: the points to swap at (default: all of POINTS, as in the main run)."""
    import torch

    hooks = []
    for i in (POINTS if points is None else points):
        src = torch.tensor(sources[i], dtype=torch.float32, device=model.cfg.device)
        D = None if directions is None else torch.tensor(directions[i], dtype=torch.float32, device=model.cfg.device)

        def swap(act, hook, src=src, D=D):
            x = act[:, pos, :].float()
            new = src if D is None else x - ((x - src) @ D.T) @ D
            act[:, pos, :] = new.to(act.dtype)
            return act

        hooks.append((point_name(i), swap))
    return hooks


def make_clamp_hooks(model, group, clean, store):
    """Hold the group's features at "=" at their clean values (as Q1's suppression, with the
    clean value in place of 0). group: {layer: feature ids}; clean: {layer: (batch, k)}."""
    from helixadd.intervene import EQ

    hooks = []
    for layer, feats in group.items():
        W_dec = model.transcoders._get_decoder_vectors(layer, feats).to(model.cfg.device)
        feats = feats.to(model.cfg.device)

        def read_input(x, hook, layer=layer, feats=feats):
            store[layer] = model.transcoders.encode_layer(x[:, EQ, :], layer)[:, feats]
            return x

        def fix_output(out, hook, layer=layer, W_dec=W_dec):
            # (clean - current) x decoder: the feature's output becomes its clean value
            out[:, EQ, :] += ((clean[layer].to(W_dec.dtype) - store[layer]) @ W_dec).to(out.dtype)
            return out

        hooks.append((f"blocks.{layer}.{model.feature_input_hook}", read_input))
        hooks.append((f"blocks.{layer}.{model.feature_output_hook}", fix_output))
    return hooks


def generate(model, prompts, hooks, watch, max_new_tokens=4, return_logits=False):
    """Greedy generation with the hooks; returns the texts and each supernode's activations
    at "=" during the first step ({name: (batch, n_features)} as numpy); with
    return_logits, also the first step's logits (batch, vocab) as a CPU tensor."""
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
                first_logits = logits.float().cpu()
            tokens = torch.cat([tokens, logits.argmax(dim=-1)[:, None]], dim=1)
    texts = [model.tokenizer.decode(row) for row in tokens[:, n_prompt:]]
    if return_logits:
        return texts, first, first_logits
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


def run_one(model, quads, position, dirs, directions, ref, watch, batch=64,
            points=None, source_X=None, clamp=None, save_logits=False):
    """One run: swap on every prompt; one row per prompt with outcomes, shift and change.

    Descriptive options (9.8; the defaults give the main run):
    points: swap only at these points; source_X: per point, (100, d) vectors to
    take the partner's vector from (default: the collected vectors); clamp:
    (group, clean) to hold a supernode at its clean values, clean being its
    (n_prompts, n_features) activations in the clean run.
    """
    import torch

    from helixadd.helix import change, off_target, shift

    rows, acts = [], {}
    pos = POS[position]
    Xs = [d["X"] for d in dirs] if source_X is None else source_X
    for i in range(0, len(quads), batch):
        chunk = quads[i:i + batch]
        partner = [q[2] if position == "a" else q[3] for q in chunk]
        sources = [Xs[p][partner] for p in range(len(POINTS))]
        hooks = make_swap_hooks(model, pos, sources, directions, points)
        if clamp is not None:
            group, clean = clamp
            values, start = {}, 0
            for layer in sorted(group):                      # same order as make_read_hooks
                k = len(group[layer])
                values[layer] = torch.tensor(clean[i:i + batch, start:start + k], device=model.cfg.device)
                start += k
            hooks += make_clamp_hooks(model, group, values, {})
        out = generate(model, [make_prompt(a, b) for a, b, _, _ in chunk], hooks, watch, return_logits=save_logits)
        texts, s = out[0], out[1]
        for name, v in s.items():
            acts.setdefault(name, []).append(v)
        for j, (a, b, a2, b2) in enumerate(chunk):
            pred = parse_answer(texts[j])
            row = {"a": a, "b": b, "a2": a2, "b2": b2, "carry": is_carry(a, b), "raw": texts[j],
                   "pred": pred, "outcome": outcome(pred, a + b), "predicted": pred == a2 + b2}
            if save_logits:
                ids = model.tokenizer.convert_tokens_to_ids([str(a + b), str(a2 + b2)])
                row["logit clean answer"] = float(out[2][j, ids[0]])
                row["logit target answer"] = float(out[2][j, ids[1]])
            rows.append(row)
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


# ---- descriptive runs (preregistration 9.8, plus two additions) ----

DESC_FOLDER = Path("results/q2_descriptive")
FAST = [10, 5, 10 / 3, 2.5, 2]
SLOW = [100, 50]
N_POINT_PROMPTS = 200          # one-point runs: first 200 evaluation prompts (9.8)
N_POINT_RANDOM = 20            # floor per point for the two groups (9.8)


def pair_set(name, position, pairs, seed=0):
    """(a, b, a2, b2) per prompt for the descriptive swaps.

    fast: same tens, units differ by an odd amount other than 5, same carry status
      (so every fast period, 2 and 5 included, sits at a different angle for a');
    slow: same units, tens differ by anything but 5 (so T = 50 differs too);
    carry flip: same tens, other units, carry status flipped (addition).
    """
    rng = np.random.default_rng(seed)
    out = []
    for a, b in pairs:
        x = a if position == "a" else b
        options = []
        for y in range(100):
            if y == x:
                continue
            a2, b2 = (y, b) if position == "a" else (a, y)
            if (a2, b2) in EXCLUDED or (b2, a2) in EXCLUDED:
                continue
            du, dt = y % 10 - x % 10, y // 10 - x // 10
            if name == "fast":
                ok = dt == 0 and du % 2 != 0 and abs(du) != 5 and is_carry(a2, b2) == is_carry(a, b)
            elif name == "slow":
                ok = du == 0 and abs(dt) != 5
            elif name == "carry flip":
                ok = dt == 0 and is_carry(a2, b2) != is_carry(a, b)
            else:
                raise ValueError(name)
            if ok:
                options.append((a2, b2))
        if options:
            a2, b2 = options[rng.integers(len(options))]
            out.append((a, b, a2, b2))
    return out


def sub_helix(data, position, kept):
    """Per point, (100, d): every number's vector rebuilt from the mean and the circles of
    `kept` only (refitted with just those columns, on all 100 numbers)."""
    from helixadd.helix import basis

    cols = [c for T in kept for c in (1 + 2 * Q2_PERIODS.index(T), 2 + 2 * Q2_PERIODS.index(T))]
    out = []
    for i in POINTS:
        X = data[position][i]
        mean = X.mean(axis=0)
        B = basis(np.arange(100), Q2_PERIODS)[:, cols]
        C = np.linalg.lstsq(B, X - mean, rcond=None)[0]
        out.append(mean + B @ C)
    return out


def descriptive_plan():
    """(name, position, pair set, kind of run, detail)."""
    runs = []
    for p in ["a", "b"]:
        # clamp test (9.8): digit swap with the last-digit supernode held at its clean values
        runs.append((f"{p} digit swap clamped", p, "digit", "clamp", None))
        # each period alone, plus the two groups on the same pair sets (9.8)
        for T in Q2_PERIODS:
            runs.append((f"{p} period {T:g} swap", p, "fast" if T in FAST else "slow", "period", T))
        runs.append((f"{p} fast group swap", p, "fast", "group", "digit"))
        runs.append((f"{p} slow group swap", p, "slow", "group", "slow"))
        # carry flip (addition): digit swap that changes the carry
        runs.append((f"{p} digit swap carry flip", p, "carry flip", "group", "digit"))
        # sub-helix patching into a' + b (9.8): whole vector of a from one group or one period
        for kept, ps in [(GROUPS["digit"], "fast"), (GROUPS["slow"], "slow")] + [([T], "fast" if T in FAST else "slow") for T in Q2_PERIODS]:
            label = "digit group" if kept == GROUPS["digit"] else "slow group" if kept == GROUPS["slow"] else f"period {kept[0]:g}"
            runs.append((f"{p} sub-helix {label}", p, ps, "sub-helix", kept))
        # one point at a time (9.8), first N_POINT_PROMPTS prompts
        for g in ["digit", "slow"]:
            for i in POINTS:
                runs.append((f"{p} {g} point {i:02d} swap", p, g, "point", (g, i, None)))
                runs += [(f"{p} {g} point {i:02d} random {r:02d}", p, g, "point", (g, i, r)) for r in range(N_POINT_RANDOM)]
    return runs


def run_descriptive(model, data, pairs, seed=0, folder=DESC_FOLDER):
    """Every run of descriptive_plan(), skipping the ones already saved. Watches the
    comparison supernode too (addition)."""
    import json

    from helixadd.helix import circle_directions, fit

    folder = Path(folder)
    (folder / "runs").mkdir(parents=True, exist_ok=True)
    (folder / "reference").mkdir(parents=True, exist_ok=True)
    (folder / "params.json").write_text(json.dumps({
        "points": POINTS, "groups": GROUPS, "periods": Q2_PERIODS, "seed": seed, "n_prompts": len(pairs),
        "n_point_prompts": N_POINT_PROMPTS, "n_point_random": N_POINT_RANDOM,
        "watched": ["last digit", "magnitude", "comparison"], "prompts": "q1.eval_pairs()"}, indent=2))
    watch = supernode_groups()
    from helixadd.groups import load_labels, supernode
    from helixadd.intervene import by_layer
    comp = supernode(load_labels(), "comparison")
    watch["comparison"] = by_layer(comp["layer"], comp["feature"])
    dirs = {p: point_directions(data, p) for p in ["a", "b"]}
    fits = {p: [fit(np.arange(100), data[p][i], Q2_PERIODS)[1] for i in POINTS] for p in ["a", "b"]}

    def quads_for(pos, ps, kind):
        sub = pairs[:N_POINT_PROMPTS] if kind == "point" else pairs
        if ps in ("digit", "slow"):
            q = swap_pairs(ps, pos, sub, seed)
        else:
            q = pair_set(ps, pos, sub, seed)
        if kind == "sub-helix":                               # run a' + b, put a back
            q = [(a2, b2, a, b) for a, b, a2, b2 in q]
        return q

    def ref_for(name_key, quads):
        f = folder / "reference" / f"{name_key}.npz"
        if f.exists():
            z = np.load(f)
            return {w: {n: z[f"{w}|{n}"] for n in watch} for w in ["clean", "target"]}
        ref = reference(model, quads, watch)
        np.savez_compressed(f, **{f"{w}|{n}": v for w, d in ref.items() for n, v in d.items()})
        return ref

    for name, pos, ps, kind, detail in descriptive_plan():
        f = folder / "runs" / f"{name.replace(' ', '_')}.csv"
        if f.exists():
            continue
        quads = quads_for(pos, ps, kind)
        ref = ref_for(f"{pos}_{ps}_{kind if kind in ('sub-helix', 'point') else 'swap'}", quads)
        d = dirs[pos]
        kw = {}
        if kind == "clamp":
            directions = [x["digit"] for x in d]
            kw["clamp"] = (watch["last digit"], ref["clean"]["last digit"])
        elif kind == "period":
            directions = [circle_directions(C, [detail]) for C in fits[pos]]
        elif kind == "group":
            directions = [x[detail] for x in d]
        elif kind == "sub-helix":
            directions = None
            kw["source_X"] = sub_helix(data, pos, detail)
        else:                                                  # one point at a time
            g, i, r = detail
            kw["points"] = [i]
            if r is None:
                directions = [x[g] for x in d]
            else:
                rng = np.random.default_rng(10_000 + 1000 * (pos == "b") + 100 * (g == "slow") + 20 * i + r)
                directions = [random_directions(x["rest"], len(x[g]), rng) for x in d]
        run_one(model, quads, pos, d, directions, ref, watch, **kw).to_csv(f, index=False)
        print(name, "saved")


# ---- post hoc (decided after the descriptive runs; not part of any preregistered claim) ----

POSTHOC_FOLDER = Path("results/q2_posthoc")
DIGIT_WITH_2 = [10, 5, 10 / 3, 2.5, 2]      # the digit group with T = 2, the k = 50 harmonic of period 10
CORE_FAST = [10, 5, 2]
N_POSTHOC_RANDOM = 20
N_REMOVAL_RANDOM = 5
REMOVALS = {**{f"period {T:g}": [T] for T in Q2_PERIODS}, "digit+2": DIGIT_WITH_2, "slow": [100, 50]}


def posthoc_plan():
    """(name, operand, pair set, kind, detail)."""
    runs = []
    for p in ["a", "b"]:
        runs.append((f"{p} digit+2 swap main pairs", p, "digit", "swap", DIGIT_WITH_2))
        runs.append((f"{p} digit+2 swap odd pairs", p, "fast", "swap", DIGIT_WITH_2))
        runs += [(f"{p} digit+2 random {r:02d} main pairs", p, "digit", "random", r) for r in range(N_POSTHOC_RANDOM)]
        runs.append((f"{p} digit+2 swap carry flip", p, "carry flip", "swap", DIGIT_WITH_2))
        # the grid: fast part x slow additions, on the main digit pairs
        for fname, fast in [("10,5,2", CORE_FAST), ("digit+2", DIGIT_WITH_2)]:
            for sname, slow in [("", []), ("+100", [100]), ("+50", [50]), ("+100+50", [100, 50])]:
                runs.append((f"{p} grid {fname}{sname} main pairs", p, "digit", "swap", fast + slow))
        runs.append((f"{p} ceiling main pairs", p, "digit", "ceiling", None))
        runs.append((f"{p} all circles slow pairs", p, "slow", "swap", Q2_PERIODS))
        runs.append((f"{p} line slow pairs", p, "slow", "line", None))
        # removal: the part set to the average over numbers, on the clean prompts
        runs.append((f"{p} nothing removed", p, "digit", "clean", None))
        runs += [(f"{p} remove {k}", p, "digit", "remove", v) for k, v in REMOVALS.items()]
        runs += [(f"{p} remove random{k} {r:02d}", p, "digit", "remove random", (k, r))
                 for k in (9, 4) for r in range(N_REMOVAL_RANDOM)]
        # comparison with Kantamneni: run a' + b, patch the fit of the clean a built from part of
        # the helix (average + those circles), read the clean answer's logit; on both pair sets the
        # unpatched run (0) and the real vector (100%) give the scale
        for ps in ["fast", "slow"]:
            runs.append((f"{p} kantamneni {ps} unpatched", p, ps, "k unpatched", None))
            runs.append((f"{p} kantamneni {ps} real vector", p, ps, "k real", None))
            runs.append((f"{p} kantamneni {ps} all circles", p, ps, "k fit", Q2_PERIODS))
        parts = [("digit+2", DIGIT_WITH_2, "fast"), ("slow", [100, 50], "slow")]
        parts += [(f"period {T:g}", [T], "slow" if T in (100, 50) else "fast") for T in Q2_PERIODS]
        runs += [(f"{p} kantamneni {ps} {label}", p, ps, "k fit", kept) for label, kept, ps in parts]
    return runs


def run_posthoc(model, data, pairs, seed=0, folder=POSTHOC_FOLDER):
    """Every run of posthoc_plan(), skipping the ones already saved. Logits of the clean and
    the target answer are saved in every run."""
    import json

    from helixadd.groups import load_labels, supernode
    from helixadd.helix import basis, circle_directions, fit
    from helixadd.intervene import by_layer

    folder = Path(folder)
    (folder / "runs").mkdir(parents=True, exist_ok=True)
    (folder / "reference").mkdir(parents=True, exist_ok=True)
    (folder / "params.json").write_text(json.dumps({
        "points": POINTS, "periods": Q2_PERIODS, "seed": seed, "n_prompts": len(pairs),
        "n_random": N_POSTHOC_RANDOM, "n_removal_random": N_REMOVAL_RANDOM,
        "note": "post hoc, after the descriptive runs (preregistration 9.13)"}, indent=2))
    watch = supernode_groups()
    comp = supernode(load_labels(), "comparison")
    watch["comparison"] = by_layer(comp["layer"], comp["feature"])

    cache = {}
    for name, pos, ps, kind, detail in posthoc_plan():
        f = folder / "runs" / f"{name.replace(' ', '_')}.csv"
        if f.exists():
            continue
        if pos not in cache:
            dirs = point_directions(data, pos)
            fits = [fit(np.arange(100), data[pos][i], Q2_PERIODS)[1] for i in POINTS]
            rest13 = []                                   # number space minus every circle and the line
            for i, C in zip(POINTS, fits):
                helix = orthonormal(C)
                rest13.append(orthonormal(dirs[i]["rest"] - dirs[i]["rest"] @ helix.T @ helix))
            means = [np.tile(data[pos][i].mean(axis=0), (100, 1)) for i in POINTS]
            cache[pos] = (dirs, fits, rest13, means)
        dirs, fits, rest13, means = cache[pos]
        quads = swap_pairs(ps, pos, pairs, seed) if ps in ("digit", "slow") else pair_set(ps, pos, pairs, seed)
        reverse = kind.startswith("k ")
        if reverse:                                            # run a' + b, put a back
            quads = [(a2, b2, a, b) for a, b, a2, b2 in quads]
        ref_file = folder / "reference" / f"{pos}_{ps}{'_reversed' if reverse else ''}.npz"
        if ref_file.exists():
            z = np.load(ref_file)
            ref = {w: {n: z[f"{w}|{n}"] for n in watch} for w in ["clean", "target"]}
        else:
            ref = reference(model, quads, watch)
            np.savez_compressed(ref_file, **{f"{w}|{n}": v for w, d in ref.items() for n, v in d.items()})
        kw = {"save_logits": True}
        if kind == "swap":
            directions = [circle_directions(C, detail) for C in fits]
        elif kind == "random":
            rng = np.random.default_rng(50_000 + 1000 * (pos == "b") + detail)
            directions = [random_directions(x, 9, rng) for x in rest13]
        elif kind == "ceiling":
            directions = None
        elif kind == "line":
            directions = [orthonormal(C[:1]) for C in fits]
        elif kind == "clean":
            directions = [np.zeros((0, data[pos].shape[-1]))] * len(POINTS)
        elif kind == "remove":
            directions = [circle_directions(C, detail) for C in fits]
            kw["source_X"] = means
        elif kind == "k unpatched":
            directions = [np.zeros((0, data[pos].shape[-1]))] * len(POINTS)
        elif kind == "k real":
            directions = None
        elif kind == "k fit":
            directions = None
            kw["source_X"] = sub_helix(data, pos, detail)
        else:                                                    # remove random
            k, r = detail
            rng = np.random.default_rng(60_000 + 1000 * (pos == "b") + 100 * k + r)
            directions = [random_directions(x, k, rng) for x in rest13]
            kw["source_X"] = means
        run_one(model, quads, pos, dirs, directions, ref, watch, **kw).to_csv(f, index=False)
        print(name, "saved")


# ---- carry check (post hoc, 2026-11-21) ----

CARRY_FOLDER = Path("results/q2_carry")


def run_carry_check(model, data, pairs, seed=0, folder=CARRY_FOLDER):
    """Carry-flip swaps with growing parts of the helix: fast + T = 2, + T = 50, + T = 100,
    all circles, and the whole vector. If the tens follow the new carry only once the slow
    circles are swapped too, the tens read the carry from the slow circles."""
    import json

    from helixadd.groups import load_labels, supernode
    from helixadd.helix import circle_directions, fit
    from helixadd.intervene import by_layer

    folder = Path(folder)
    (folder / "runs").mkdir(parents=True, exist_ok=True)
    (folder / "params.json").write_text(json.dumps({
        "pairs": "pair_set('carry flip')", "points": POINTS, "seed": seed, "n_prompts": len(pairs),
        "note": "post hoc, 2026-11-21"}, indent=2))
    watch = supernode_groups()
    comp = supernode(load_labels(), "comparison")
    watch["comparison"] = by_layer(comp["layer"], comp["feature"])
    parts = [("fast+2", DIGIT_WITH_2), ("fast+2+50", DIGIT_WITH_2 + [50]), ("fast+2+100", DIGIT_WITH_2 + [100]),
             ("all circles", Q2_PERIODS), ("whole vector", None)]
    for pos in ["a", "b"]:
        dirs = point_directions(data, pos)
        fits = [fit(np.arange(100), data[pos][i], Q2_PERIODS)[1] for i in POINTS]
        quads = pair_set("carry flip", pos, pairs, seed)
        ref = None
        for label, kept in parts:
            f = folder / "runs" / f"{pos}_carry_flip_{label.replace(' ', '_')}.csv"
            if f.exists():
                continue
            if ref is None:
                ref = reference(model, quads, watch)
            directions = None if kept is None else [circle_directions(C, kept) for C in fits]
            run_one(model, quads, pos, dirs, directions, ref, watch, save_logits=True).to_csv(f, index=False)
            print(pos, label, "saved")
