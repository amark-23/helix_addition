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
