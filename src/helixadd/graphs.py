"""Preregistration 7.3: attribution graphs, summarized per prompt.

For each graph (circuit-tracer `attribute`), two descriptive measures:
- influence share: how much of the influence on the answer flows through each
  group of nodes (the five supernode groups at "=", mostly active and unclear
  features at "=", features at other positions, error nodes, token embeddings);
- edges between the two primary groups: for each pair of groups (source,
  target), the average share of a target feature's direct input that comes
  from the source group.
Node order in a circuit-tracer graph: [features, errors (layer x position),
token embeddings, logits]; rows are targets, columns are sources.
"""
import pandas as pd
import torch

EQ = 43
GROUPS = ["last digit", "magnitude", "mod 100", "comparison", "mixed family"]
CATEGORIES = GROUPS + ["mostly active", "unclear", "rare (unscored)", "other positions", "error", "embedding"]
PRIMARY = ["last digit", "magnitude"]


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


def summarize(graph, categories):
    """One row of descriptive numbers for one graph."""
    from circuit_tracer.graph import compute_graph_scores, compute_node_influence, normalize_matrix

    A = graph.adjacency_matrix.float()
    n_feat = len(graph.selected_features)
    n_err = graph.cfg.n_layers * graph.n_pos
    n_tok = graph.n_pos
    n_logit = len(graph.logit_targets)

    feats = graph.active_features[graph.selected_features].cpu()          # (n_feat, 3): layer, pos, feature
    node_cat = []
    for layer, pos, feature in feats.tolist():
        if pos != EQ:
            node_cat.append("other positions")
        else:
            node_cat.append(categories.get((layer, feature), "rare (unscored)"))
    node_cat += ["error"] * n_err + ["embedding"] * n_tok

    logit_weights = torch.zeros(A.shape[0], device=A.device)
    logit_weights[-n_logit:] = graph.logit_probabilities.to(A.device).float()
    influence = compute_node_influence(A, logit_weights)[: n_feat + n_err + n_tok].cpu()
    share = pd.Series(influence.numpy(), index=node_cat).groupby(level=0).sum() / float(influence.sum())

    replacement, completeness = compute_graph_scores(graph)
    # the same columns for every graph (so rows can be appended to one CSV)
    row = {"replacement_score": replacement, "completeness_score": completeness, "n_feature_nodes": n_feat}
    row.update({f"influence {c}": float(share.get(c, 0.0)) for c in CATEGORIES})

    N = normalize_matrix(A).cpu()
    idx = {g: [i for i, c in enumerate(node_cat[:n_feat]) if c == g] for g in PRIMARY}
    for src in PRIMARY:
        for tgt in PRIMARY:
            ok = idx[src] and idx[tgt]
            row[f"edges {src} -> {tgt}"] = float(N[idx[tgt]][:, idx[src]].sum(dim=1).mean()) if ok else float("nan")
    row.update({f"n {g}": len(idx[g]) for g in PRIMARY})
    return row
