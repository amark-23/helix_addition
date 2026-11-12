"""Preregistration 7.3: attribution graphs, summarized per prompt.

For each graph (circuit-tracer `attribute`), two descriptive measures:
- influence share: how much of the influence on the answer flows through each
  group of nodes (the five supernode groups at "=", mostly active and unclear
  features at "=", features at other positions, error nodes, token embeddings);
- edges between the two primary groups: for each pair of groups (source,
  target), the average share of a target feature's direct input that comes
  from the source group.
Exploratory (not preregistered): the unclear share split further by the exact
kinds of scripts/exact_kinds.py (unclear features that fire mostly on one exact
operand value, sum or difference). These columns are parts of "influence
unclear", so they are not counted in the shares that sum to 1.
Node order in a circuit-tracer graph: [features, errors (layer x position),
token embeddings, logits]; rows are targets, columns are sources.
"""
import pandas as pd
import torch

EQ = 43
GROUPS = ["last digit", "magnitude", "mod 100", "comparison", "mixed family"]
CATEGORIES = GROUPS + ["mostly active", "unclear", "rare (unscored)", "other positions", "error", "embedding"]
PRIMARY = ["last digit", "magnitude"]
EXACT_KINDS = ["exact operand value", "exact sum", "exact difference"]


def feature_categories(labels):
    """{(layer, feature): category} for every scored feature at "="."""
    from helixadd.groups import supernode

    cat = {}
    for _, r in labels.iterrows():
        cat[(int(r["layer"]), int(r["feature"]))] = "mostly active" if r["role"] == "mostly active" else "unclear"
    for name in GROUPS:
        family = "mixed" if name == "mixed family" else name
        for _, r in supernode(labels, family).iterrows():
            cat[(int(r["layer"]), int(r["feature"]))] = name
    return cat


def exact_kinds(path="results/exact_kinds/data.csv"):
    """{(layer, feature): exact kind} for the unclear features in scripts/exact_kinds.py."""
    df = pd.read_csv(path)
    return {(int(r["layer"]), int(r["feature"])): r["kind"] for _, r in df.iterrows()}


def summarize(graph, categories, exact=None):
    """One row of descriptive numbers for one graph. `exact`: optional output of exact_kinds()."""
    from circuit_tracer.graph import compute_graph_scores, compute_node_influence, normalize_matrix

    A = graph.adjacency_matrix.float()
    n_feat = len(graph.selected_features)
    n_err = graph.cfg.n_layers * graph.n_pos
    n_tok = graph.n_pos
    n_logit = len(graph.logit_targets)

    feats = graph.active_features[graph.selected_features].cpu()          # (n_feat, 3): layer, pos, feature
    node_cat, node_exact = [], []
    for layer, pos, feature in feats.tolist():
        if pos != EQ:
            node_cat.append("other positions")
        else:
            node_cat.append(categories.get((layer, feature), "rare (unscored)"))
        is_unclear = node_cat[-1] == "unclear" and exact is not None
        node_exact.append(exact.get((layer, feature), "none") if is_unclear else "none")
    node_cat += ["error"] * n_err + ["embedding"] * n_tok

    logit_weights = torch.zeros(A.shape[0], device=A.device)
    logit_weights[-n_logit:] = graph.logit_probabilities.to(A.device).float()
    influence = compute_node_influence(A, logit_weights)[: n_feat + n_err + n_tok].cpu()
    share = pd.Series(influence.numpy(), index=node_cat).groupby(level=0).sum() / float(influence.sum())

    replacement, completeness = compute_graph_scores(graph)
    # the same columns for every graph (so rows can be appended to one CSV)
    row = {"replacement_score": replacement, "completeness_score": completeness, "n_feature_nodes": n_feat}
    row.update({f"influence {c}": float(share.get(c, 0.0)) for c in CATEGORIES})
    if exact is not None:   # exploratory, parts of "influence unclear"
        part = pd.Series(influence[:n_feat].numpy(), index=node_exact).groupby(level=0).sum() / float(influence.sum())
        row.update({f"unclear part: {k}": float(part.get(k, 0.0)) for k in EXACT_KINDS})

    N = normalize_matrix(A).cpu()
    idx = {g: [i for i, c in enumerate(node_cat[:n_feat]) if c == g] for g in PRIMARY}
    for src in PRIMARY:
        for tgt in PRIMARY:
            ok = idx[src] and idx[tgt]
            row[f"edges {src} -> {tgt}"] = float(N[idx[tgt]][:, idx[src]].sum(dim=1).mean()) if ok else float("nan")
    row.update({f"n {g}": len(idx[g]) for g in PRIMARY})
    return row
