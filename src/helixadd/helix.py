import numpy as np


def spectrum(X):
    """X: (100, d), row n = the activation for the number n.
    Returns the share of the variance at each frequency k = 0..50."""
    X = X - np.mean(X, axis=0)          # gap 1: remove the average vector over the 100 numbers
    F = np.fft.rfft(X, axis=0)          # gap 2: transform along the numbers
    power = np.sum(np.abs(F)**2, axis=1) # gap 3: strength per frequency, summed over the d dimensions
    return power / power.sum()

A_POS, B_POS = 40, 42          # positions of a and b in every prompt (44 tokens)
FIXED_B = 50                   # the b used when collecting a (b comes after a, so it cannot change a's vector)
N_A_PER_B = 20                 # a's averaged over for each b


def number_prompts(seed=0):
    """The prompts for collect_numbers: 100 for a (fixed b), then N_A_PER_B per b."""
    from helixadd.prompts import EXCLUDED, make_prompt

    rng = np.random.default_rng(seed)
    a_pairs = [(a, FIXED_B) for a in range(100)]
    b_pairs = []
    for b in range(100):
        choices = [a for a in range(100) if (a, b) not in EXCLUDED and (b, a) not in EXCLUDED]
        b_pairs += [(int(a), b) for a in rng.choice(choices, N_A_PER_B, replace=False)]
    return a_pairs, b_pairs


def collect_numbers(model, folder="results/number_acts", batch=100, seed=0):
    """Residual stream on the number tokens, at 17 points (embedding, after each layer).

    Returns {"a": (17, 100, d), "b": (17, 100, d)}: row n is the vector for the
    number n. For b, the mean over N_A_PER_B prompts with different a's.
    Saved with its parameters (results/<folder>/data.npz + params.json).
    """
    import importlib.metadata as md

    import torch

    from helixadd.model import MODEL_NAME, tokenize
    from helixadd.prompts import PREFIX, make_prompt
    from helixadd.results import load_arrays_if_exists, save_arrays

    params = {"model": MODEL_NAME, "dtype": "float16", "prefix": PREFIX,
              "points": "blocks.0.hook_resid_pre, then blocks.L.hook_resid_post for L = 0..15",
              "a": f"a = 0..99 with b = {FIXED_B}", "b": f"mean over {N_A_PER_B} random a per b, seed {seed}",
              "positions": [A_POS, B_POS], "transformer_lens": md.version("transformer-lens"),
              "circuit_tracer": md.version("circuit-tracer"), "torch": md.version("torch")}
    data = load_arrays_if_exists(folder, params)
    if data is not None:
        return data

    names = ["blocks.0.hook_resid_pre"] + [f"blocks.{L}.hook_resid_post" for L in range(model.cfg.n_layers)]
    a_pairs, b_pairs = number_prompts(seed)

    def run(pairs, pos):
        """(17, len(pairs), d) vectors at one position, float32 on the CPU."""
        out = []
        for i in range(0, len(pairs), batch):
            chunk = pairs[i:i + batch]
            tokens = tokenize(model, [make_prompt(a, b) for a, b in chunk])
            # sanity check: the token at pos is the number we think it is
            for j, (a, b) in enumerate(chunk):
                assert model.tokenizer.decode(tokens[j, pos]) == str(a if pos == A_POS else b)
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens, names_filter=lambda n: n in names)
            out.append(torch.stack([cache[n][:, pos, :] for n in names]).float().cpu())
            del cache
        return torch.cat(out, dim=1).numpy()

    a_acts = run(a_pairs, A_POS)                                       # (17, 100, d), row = a
    b_all = run(b_pairs, B_POS)                                        # (17, 2000, d), grouped by b
    b_acts = b_all.reshape(len(names), 100, N_A_PER_B, -1).mean(axis=2)   # (17, 100, d), row = b
    save_arrays(folder, params, a=a_acts, b=b_acts)
    return {"a": a_acts, "b": b_acts}


PERIODS = [2, 5, 10, 100]


def basis(numbers, periods=PERIODS):
    """(len(numbers), 1 + 2 * len(periods)): the linear term, then cos and sin for each period."""
    n = np.asarray(numbers, dtype=float)           #the numebrs as a float array
    columns = [n]                                  # gap 1: the linear term
    for T in periods:
        angle = 2 * np.pi * n / T                      # gap 2: the angle for period T
        columns += [np.cos(angle), np.sin(angle)]       # gap 3: its cos and sin
    return np.stack(columns, axis=1)



def fit(numbers, X, periods=PERIODS):
    """Fit X ≈ mean + basis(numbers) @ C. Returns (mean, C)."""
    mean = np.mean(X, axis=0)                                       # gap 4: the average vector over these numbers
    B = basis(numbers, periods)
    C = np.linalg.lstsq(B, X - mean, rcond=None)[0]       # gap 5: what B @ C should match
    return mean, C


def predict(numbers, mean, C, periods=PERIODS):
    """The helix vectors for these numbers."""
    return mean + basis(numbers, periods) @ C                                       # gap 6



def point_name(i):
    """Hook name of point i (0 = embedding, i = after layer i - 1)."""
    return "blocks.0.hook_resid_pre" if i == 0 else f"blocks.{i - 1}.hook_resid_post"


def patch_logits(model, pairs, point, pos, vectors, answers=None):
    """Logit of the correct answer, with the vector at `pos` replaced at `point`.

    pairs: [(a, b), ...]; vectors: (len(pairs), d), the replacement for each prompt,
    or None for the normal run. answers: the answer whose logit is read for each
    prompt (default a + b; the corrupted runs read the clean answer instead).
    """
    import torch

    from helixadd.model import tokenize
    from helixadd.prompts import make_prompt

    tokens = tokenize(model, [make_prompt(a, b) for a, b in pairs])

    def replace(act, hook):
        act[:, pos, :] =  v                      # gap 1: which position, and with what
        return act

    v = None if vectors is None else torch.tensor(vectors, dtype=model.cfg.dtype, device=model.cfg.device)   # gap 2
    hooks = [] if vectors is None else [(point_name(point), replace)]
    with torch.no_grad():
        logits = model.run_with_hooks(tokens, fwd_hooks=hooks)[:, -1, :]
    if answers is None:
        answers = [a + b for a, b in pairs]
    ids = [model.tokenizer.convert_tokens_to_ids(str(x)) for x in answers]
    return logits[torch.arange(len(pairs)), ids].float().cpu().numpy()    # gap 3


METHODS = ["mean", "helix", "line only", "circles only", "pca 9"]
N_TEST = 20                    # numbers held out of the fit; only these are patched
N_PARTNERS = 10                # prompts per held-out number (the other operand drawn at random)


def split(seed=0):
    """(train, test) numbers: the fit sees train only."""
    rng = np.random.default_rng(seed)
    test = np.sort(rng.choice(100, N_TEST, replace=False))
    return np.setdiff1d(np.arange(100), test), test


def replacement(method, X, train, test, periods=PERIODS):
    """(len(test), d) replacement vectors for the test numbers, from a fit on the train numbers.

    mean: the train average (no information about the number). helix: line and
    PERIODS. line only, circles only: the two parts fitted on their own. pca 9:
    the real vector projected on the top 9 principal directions of the train
    numbers (it sees the test vector, so it is a ceiling, not a rival).
    """
    mean = X[train].mean(axis=0)
    if method == "mean":
        return np.tile(mean, (len(test), 1))
    if method == "pca 9":
        V = np.linalg.svd(X[train] - mean, full_matrices=False)[2][:9]
        return mean + (X[test] - mean) @ V.T @ V
    columns = {"helix": slice(None), "line only": slice(0, 1), "circles only": slice(1, None)}[method]
    C = np.linalg.lstsq(basis(train, periods)[:, columns], X[train] - mean, rcond=None)[0]
    return mean + basis(test, periods)[:, columns] @ C


def patching_pairs(position, test, seed=0):
    """Prompts for one position: each test number with N_PARTNERS random other operands."""
    from helixadd.prompts import EXCLUDED

    rng = np.random.default_rng(seed + 1)
    pairs = []
    for x in test:
        for y in rng.choice(100, N_PARTNERS, replace=False):
            pair = (int(x), int(y)) if position == "a" else (int(y), int(x))
            if pair not in EXCLUDED and pair[::-1] not in EXCLUDED:
                pairs.append(pair)
    return pairs


def run_patching(model, data, folder="results/helix_patching", seed=0, batch=100):
    """Mean correct-answer logit for every position, point and method, plus the clean run.

    Saved as one row per (position, point, method) with its parameters.
    """
    import pandas as pd

    from helixadd.results import load_if_exists, save

    params = {"activations": "results/number_acts", "periods": PERIODS, "methods": METHODS,
              "n_test": N_TEST, "n_partners": N_PARTNERS, "seed": seed,
              "replaced": "the operand's vector at one point, on its own token"}
    df = load_if_exists(folder, params)
    if df is not None:
        return df

    train, test = split(seed)
    rows = []
    for position, pos in [("a", A_POS), ("b", B_POS)]:
        pairs = patching_pairs(position, test, seed)
        row_of = {int(x): i for i, x in enumerate(test)}
        which = [row_of[a if position == "a" else b] for a, b in pairs]       # test row for each prompt

        def logits(point, vectors):
            out = []
            for i in range(0, len(pairs), batch):
                v = None if vectors is None else vectors[which[i:i + batch]]
                out.append(patch_logits(model, pairs[i:i + batch], point, pos, v))
            return np.concatenate(out)

        clean = logits(0, None)
        rows.append({"position": position, "point": -1, "method": "clean", "logit": clean.mean(), "n": len(pairs)})
        for point in range(17):
            X = data[position][point]
            for method in METHODS:
                vecs = replacement(method, X, train, test)
                rows.append({"position": position, "point": point, "method": method,
                             "logit": logits(point, vecs).mean(), "n": len(pairs)})
            print(position, point, "done")
    df = pd.DataFrame(rows)
    save(folder, params, df)
    return df


# ---- the paper's version: patch into a run with a different number ----

CORRUPT_METHODS = ["layer patch", "helix", "circles only", "line only", "pca 9"]


def corrupted_triples(position, test, seed=0):
    """(clean pair, corrupted pair) for one position: the operand at `position` is
    a test number in the clean pair and a different random number in the corrupted one."""
    from helixadd.prompts import EXCLUDED

    rng = np.random.default_rng(seed + 2)
    out = []
    for x in test:
        for y in rng.choice(100, N_PARTNERS, replace=False):
            other = int(rng.choice([v for v in range(100) if v != x]))
            clean = (int(x), int(y)) if position == "a" else (int(y), int(x))
            corrupt = (other, int(y)) if position == "a" else (int(y), other)
            if not any(p in EXCLUDED or p[::-1] in EXCLUDED for p in (clean, corrupt)):
                out.append((clean, corrupt))
    return out


def run_corrupted(model, data, folder="results/helix_patching_corrupted", seed=0, batch=100):
    """Kantamneni-style patching: run a' + b (or a + b'), put a vector for the clean
    operand on its token at one point, read the clean answer's logit.

    logit difference = patched logit - corrupted logit (both for the clean answer).
    layer patch uses the collected vector of the clean operand (for b: the mean
    over 20 a's, so it is close to, not equal to, the clean run's own vector).
    """
    import pandas as pd

    from helixadd.results import load_if_exists, save

    params = {"activations": "results/number_acts", "periods": PERIODS, "methods": CORRUPT_METHODS,
              "n_test": N_TEST, "n_partners": N_PARTNERS, "seed": seed,
              "setup": "corrupted run a'+b (a+b'), clean operand's vector patched on its token, clean answer's logit"}
    df = load_if_exists(folder, params)
    if df is not None:
        return df

    train, test = split(seed)
    row_of = {int(x): i for i, x in enumerate(test)}
    rows = []
    for position, pos in [("a", A_POS), ("b", B_POS)]:
        triples = corrupted_triples(position, test, seed)
        corrupt = [c for _, c in triples]
        answers = [a + b for (a, b), _ in triples]
        which = [row_of[a if position == "a" else b] for (a, b), _ in triples]

        def logits(point, vectors):
            out = []
            for i in range(0, len(triples), batch):
                v = None if vectors is None else vectors[which[i:i + batch]]
                out.append(patch_logits(model, corrupt[i:i + batch], point, pos, v, answers[i:i + batch]))
            return np.concatenate(out)

        base = logits(0, None)
        rows.append({"position": position, "point": -1, "method": "corrupted", "logit difference": 0.0, "n": len(triples)})
        for point in range(17):
            X = data[position][point]
            for method in CORRUPT_METHODS:
                vecs = X[test] if method == "layer patch" else replacement(method, X, train, test)
                rows.append({"position": position, "point": point, "method": method,
                             "logit difference": float((logits(point, vecs) - base).mean()), "n": len(triples)})
            print(position, point, "done")
    df = pd.DataFrame(rows)
    save(folder, params, df)
    return df

Q2_PERIODS = [100, 50, 10, 5, 10 / 3, 2.5, 2]


# ---- the same patching test with other period sets ----

PERIOD_SETS = {"4 periods": [2, 5, 10, 100], "5 periods": [2, 5, 10, 50, 100], "7 periods": Q2_PERIODS}


def run_corrupted_periods(model, data, folder="results/helix_patching_periods", seed=0, batch=100, points=range(12)):
    """run_corrupted with the helix fitted on each set in PERIOD_SETS (same split and prompts).

    Only points 0-11 (embedding to after layer 10): later the operand's vector barely matters.
    """
    import pandas as pd

    from helixadd.results import load_if_exists, save

    params = {"activations": "results/number_acts", "period_sets": PERIOD_SETS, "n_test": N_TEST,
              "n_partners": N_PARTNERS, "seed": seed, "points": list(points),
              "setup": "as helix_patching_corrupted: corrupted run a'+b (a+b'), helix vector of the clean operand "
                       "patched on its token, clean answer's logit; fit on 80 numbers, the 20 held out are patched"}
    df = load_if_exists(folder, params)
    if df is not None:
        return df

    train, test = split(seed)
    row_of = {int(x): i for i, x in enumerate(test)}
    rows = []
    for position, pos in [("a", A_POS), ("b", B_POS)]:
        triples = corrupted_triples(position, test, seed)
        corrupt = [c for _, c in triples]
        answers = [a + b for (a, b), _ in triples]
        which = [row_of[a if position == "a" else b] for (a, b), _ in triples]

        def logits(point, vectors):
            out = []
            for i in range(0, len(triples), batch):
                v = None if vectors is None else vectors[which[i:i + batch]]
                out.append(patch_logits(model, corrupt[i:i + batch], point, pos, v, answers[i:i + batch]))
            return np.concatenate(out)

        base = logits(0, None)
        for point in points:
            X = data[position][point]
            vectors = {"layer patch": X[test]}
            for name, periods in PERIOD_SETS.items():
                vectors[name] = replacement("helix", X, train, test, periods)
            for method, vecs in vectors.items():
                rows.append({"position": position, "point": point, "method": method,
                             "logit difference": float((logits(point, vecs) - base).mean()), "n": len(triples)})
            print(position, point, "done")
    df = pd.DataFrame(rows)
    save(folder, params, df)
    return df


def circle_directions(C, kept, periods=Q2_PERIODS):
    """Orthonormal directions (k, d) of the periods in `kept`, from a fit's C."""
    rows = []
    for T in kept:
        j = periods.index(T)
        rows += [1 + 2*j,2 + 2*j]                  # gap 1: the cos and sin rows of C for period j
    U, S, Vt = np.linalg.svd(C[rows], full_matrices=False)
    return Vt[S > 1e-6 * S[0]]                # orthonormal; drops the zero row (sin of T = 2)


def replace_along(x, source, D):
    """x with its part along the directions D taken from `source`
    (for the swap: the vector of a' at the same point)."""
    return x - np.dot(np.dot(x - source, D.T), D)


def shift(s_swap, s_clean, s_target):
    """Per prompt: how far s_swap moved from s_clean towards s_target (0 = not, 1 = all the way).
    Each argument: (n_prompts, n_features)."""
    moved = s_swap - s_clean                        # gap 1: the move caused by the swap
    wanted = s_target - s_clean                     # gap 2: the move to the target
    return (moved * wanted).sum(axis=1) / (wanted * wanted).sum(axis=1)    # gap 3: |wanted|^2, per prompt


def change(s_swap, s_clean):
    """Per prompt: |s_swap - s_clean| / |s_clean| (as in 7.2)."""
    return np.linalg.norm(s_swap - s_clean, axis=1) / np.linalg.norm(s_clean, axis=1)


def off_target(s_swap, s_clean, s_target):
    """Per prompt: the movement not towards the target, |moved - shift * wanted| / |s_clean|.

    Movement that matches the real a' + b run does not count. If the target equals
    the clean run (shift undefined), all movement counts.
    """
    moved = s_swap - s_clean
    wanted = s_target - s_clean
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.nan_to_num(shift(s_swap, s_clean, s_target), nan=0.0, posinf=0.0, neginf=0.0)
    return np.linalg.norm(moved - k[:, None] * wanted, axis=1) / np.linalg.norm(s_clean, axis=1)
