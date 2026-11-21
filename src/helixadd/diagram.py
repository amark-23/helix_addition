"""Path diagram for one prompt: kept graph nodes grouped into boxes, edges summed between boxes."""
import numpy as np
import pandas as pd

A_POS, B_POS, EQ = 40, 42, 43


def box_of(node, categories):
    """The box name for one row of nodes.csv."""
    if node["kind"] == "logit":
        return "answer"
    if node["kind"] == "error":
        return "error"
    if node["kind"] == "token":
        if node["pos"] == A_POS:
            return "a token"
        if node["pos"] == B_POS:
            return "b token"
        return "other tokens"
    # from here on, the node is a feature
    if node["pos"] == A_POS:
        return "features on a"
    if node["pos"] == B_POS:
        return "features on b"
    if node["pos"] == EQ:
        return categories.get((int(node["layer"]), int(node["feature"])), "rare (unscored)")
    return "other positions"


def box_edges(folder, categories):
    """Summed edge weight between every pair of boxes, for one saved graph."""
    nodes = pd.read_csv(f"{folder}/nodes.csv").set_index("index")
    edges = pd.read_csv(f"{folder}/edges.csv")

    nodes["box"] = nodes.apply(lambda row: box_of(row, categories), axis=1)
    edges["source box"] = nodes.loc[edges["source"], "box"].values
    edges["target box"] = nodes.loc[edges["target"], "box"].values

    edges = edges[edges["source box"] != edges["target box"]]   # drop arrows from a box to itself
    return edges.groupby(["source box", "target box"])["weight"].sum().reset_index()


# ---- finer boxes, as in Anthropic's diagrams ----

# boxes at "=": by the variable a feature tracks best (final labels), whatever its family
EQ_BOX_OF_VARIABLE = {
    "units (a, b)": "units of a, b", "min mod 10": "smaller ends in", "max mod 10": "larger ends in",
    "sum mod 10": "sum ends in", "tens (a, b)": "tens of a, b", "tens separate": "tens of a, b",
    "sum // 10": "sum in the", "min // 10": "smaller in the", "min range": "smaller in the",
    "max // 10": "larger in the", "max range": "larger in the", "a < b": "a < b", "sum mod 100": "sum mod 100",
}
EQ_BOXES = ["units of a, b", "smaller ends in", "larger ends in", "sum ends in", "tens of a, b", "sum in the",
            "smaller in the", "larger in the", "a < b", "sum mod 100"]
OPERAND_VARIABLES = ["size", "tens", "units", "parity", "mod 5"]


def eq_box_label(box, a, b):
    """The box's name with its values on the prompt a + b, e.g. "sum ends in 5"."""
    lo, hi, s = min(a, b), max(a, b), a + b
    return {"units of a, b": f"units of a, b: ({a % 10}, {b % 10})", "smaller ends in": f"smaller ends in {lo % 10}",
            "larger ends in": f"larger ends in {hi % 10}", "sum ends in": f"sum ends in {s % 10}",
            "tens of a, b": f"tens of a, b: ({a // 10}, {b // 10})", "sum in the": f"sum in the {10 * (s // 10)}s",
            "smaller in the": f"smaller in the {10 * (lo // 10)}s", "larger in the": f"larger in the {10 * (hi // 10)}s",
            "a < b": "a < b" if a < b else "a > b", "sum mod 100": f"sum = {s % 100} (mod 100)"}[box]


def score_operand_features(acts, min_r2=0.6, lead=0.15, nested=0.05):
    """Label operand-token features by the property of the number they track.

    acts: (100, k), row n = the feature's activation for the number n.
    Variables: size (R^2 of a straight line in n), tens, units, parity, mod 5
    (R^2 of group means). A coarser variable wins if it is within `nested` of a
    finer one that contains it (parity and mod 5 inside units; size inside tens).
    Returns a DataFrame with the R^2 of every variable, the best one and the label.
    """
    from helixadd.scoring import r2_groups

    n = np.arange(100)
    groups = {"tens": n // 10, "units": n % 10, "parity": n % 2, "mod 5": n % 5}
    rows = []
    for j in range(acts.shape[1]):
        x = acts[:, j].astype(float)
        if x.std() == 0:
            rows.append({v: 0.0 for v in OPERAND_VARIABLES} | {"best": "none", "label": "unclear"})
            continue
        r = {"size": float(np.corrcoef(x, n)[0, 1] ** 2)}
        r.update({k: float(r2_groups(x, g)) for k, g in groups.items()})
        best = max(r, key=r.get)
        for coarse, fine in [("parity", "units"), ("mod 5", "units"), ("size", "tens")]:
            if best == fine and r[coarse] >= r[fine] - nested:
                best = coarse
        others = [v for k, v in r.items() if k != best and not {k, best} <= {"units", "parity", "mod 5"}
                  and not {k, best} == {"size", "tens"}]
        ok = r[best] >= min_r2 and (not others or r[best] - max(others) >= lead)
        rows.append(r | {"best": best, "label": best if ok else "unclear"})
    return pd.DataFrame(rows)


def operand_feature_acts(model, feats, position, batch=50):
    """(100, len(feats)) activations of the given features on the operand's token.

    feats: list of (layer, feature). Row n is the number n: for a, the prompt n + 50;
    for b, the mean over the 20 prompts of helix.number_prompts().
    """
    import torch

    from helixadd.helix import A_POS, B_POS, N_A_PER_B, number_prompts
    from helixadd.model import tokenize
    from helixadd.prompts import make_prompt

    pos = A_POS if position == "a" else B_POS
    a_pairs, b_pairs = number_prompts()
    pairs = a_pairs if position == "a" else b_pairs
    by_layer = {}
    for j, (layer, f) in enumerate(feats):
        by_layer.setdefault(int(layer), []).append((j, int(f)))
    out = np.zeros((len(pairs), len(feats)), dtype=np.float32)
    for i in range(0, len(pairs), batch):
        store = {}
        hooks = []
        for layer, items in by_layer.items():
            idx = torch.tensor([f for _, f in items], device=model.cfg.device)

            def read(x, hook, layer=layer, idx=idx):
                store[layer] = model.transcoders.encode_layer(x[:, pos, :], layer)[:, idx].float().cpu().numpy()
                return x

            hooks.append((f"blocks.{layer}.{model.feature_input_hook}", read))
        tokens = tokenize(model, [make_prompt(a, b) for a, b in pairs[i:i + batch]])
        with torch.no_grad(), model.hooks(fwd_hooks=hooks):
            model(tokens)
        for layer, items in by_layer.items():
            for k, (j, _) in enumerate(items):
                out[i:i + batch, j] = store[layer][:, k]
    if position == "b":
        out = out.reshape(100, N_A_PER_B, len(feats)).mean(axis=1)
    return out


def encoder_circle_shares(model, feats, data, position):
    """For each operand feature: the share of its input direction that lies along the
    fast circles (T = 10, 5, 3.3, 2.5, 2) and along the slow circles (T = 100, 50).

    The input direction is the encoder row, times the norm's weights if the model has them
    (the transcoder reads the normalized residual stream). The circles are those fitted at
    the points before and after the feature's layer; the two shares are averaged. A random
    direction would give about k / d (0.004 for fast, 0.002 for slow).
    """
    import torch

    from helixadd.helix import Q2_PERIODS, circle_directions, fit
    from helixadd.q2 import DIGIT_WITH_2

    n_points = data[position].shape[0]
    circles = {}
    for p in range(n_points):
        C = fit(np.arange(100), data[position][p], Q2_PERIODS)[1]
        circles[p] = (circle_directions(C, DIGIT_WITH_2), circle_directions(C, [100, 50]))
    rows = []
    for layer in sorted({int(l) for l, _ in feats}):
        idx = [int(f) for l, f in feats if int(l) == layer]
        with torch.no_grad():
            W = model.transcoders[layer].W_enc[idx].float()
            ln = model.blocks[layer].ln2
            if hasattr(ln, "w"):
                W = W * ln.w.float()
            W = W.cpu().numpy()
        W = W / np.linalg.norm(W, axis=1, keepdims=True)
        for f, w in zip(idx, W):
            shares = []
            for p in (layer, min(layer + 1, n_points - 1)):
                fast, slow = circles[p]
                shares.append((float(((fast @ w) ** 2).sum()), float(((slow @ w) ** 2).sum())))
            rows.append({"layer": layer, "feature": f, "fast share": np.mean([s[0] for s in shares]),
                         "slow share": np.mean([s[1] for s in shares])})
    return pd.DataFrame(rows)
