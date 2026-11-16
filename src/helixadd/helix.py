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


def patch_logits(model, pairs, point, pos, vectors):
    """Logit of the correct answer, with the vector at `pos` replaced at `point`.

    pairs: [(a, b), ...]; vectors: (len(pairs), d), the replacement for each prompt,
    or None for the normal run.
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
    ids = [model.tokenizer.convert_tokens_to_ids(str(a + b)) for a, b in pairs]
    return logits[torch.arange(len(pairs)), ids].float().cpu().numpy()    # gap 3


METHODS = ["mean", "helix", "line only", "circles only", "pca 9"]
N_TEST = 20                    # numbers held out of the fit; only these are patched
N_PARTNERS = 10                # prompts per held-out number (the other operand drawn at random)


def split(seed=0):
    """(train, test) numbers: the fit sees train only."""
    rng = np.random.default_rng(seed)
    test = np.sort(rng.choice(100, N_TEST, replace=False))
    return np.setdiff1d(np.arange(100), test), test


def replacement(method, X, train, test):
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
    C = np.linalg.lstsq(basis(train)[:, columns], X[train] - mean, rcond=None)[0]
    return mean + basis(test)[:, columns] @ C


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
