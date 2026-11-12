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
    import torch
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


def save_pruned(graph, folder, node_threshold=0.8, edge_threshold=0.98):
    """Prune one graph (circuit-tracer's prune_graph) and save what is left as two small CSVs.

    nodes.csv: index, kind (feature / error / token / logit), layer, pos,
    feature, token, activation, influence (cumulative, as in the viewer).
    edges.csv: target, source, weight (raw adjacency entry; rows are targets).
    The full adjacency matrix (about 3 GB) is not saved.
    """
    from pathlib import Path

    from circuit_tracer.graph import prune_graph

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    node_mask, edge_mask, cumulative = (t.cpu() for t in prune_graph(graph, node_threshold, edge_threshold))

    n_feat = len(graph.selected_features)
    n_err = graph.cfg.n_layers * graph.n_pos
    n_tok = graph.n_pos
    rows = []
    for i in node_mask.nonzero().flatten().tolist():
        r = {"index": i, "kind": None, "layer": -1, "pos": -1, "feature": -1, "token": -1,
             "activation": float("nan"), "influence": float(cumulative[i])}
        if i < n_feat:
            sel = int(graph.selected_features[i])
            layer, pos, feature = graph.active_features[sel].tolist()
            r.update(kind="feature", layer=layer, pos=pos, feature=feature,
                     activation=float(graph.activation_values[sel]))
        elif i < n_feat + n_err:
            layer, pos = divmod(i - n_feat, graph.n_pos)
            r.update(kind="error", layer=layer, pos=pos)
        elif i < n_feat + n_err + n_tok:
            pos = i - n_feat - n_err
            r.update(kind="token", pos=pos, token=int(graph.input_tokens[pos]))
        else:
            r.update(kind="logit", token=int(graph.logit_targets[i - n_feat - n_err - n_tok].vocab_idx),
                     activation=float(graph.logit_probabilities[i - n_feat - n_err - n_tok]))
        rows.append(r)
    pd.DataFrame(rows).to_csv(folder / "nodes.csv", index=False)

    tgt, src = edge_mask.nonzero(as_tuple=True)
    weight = graph.adjacency_matrix.cpu()[tgt, src].float()
    pd.DataFrame({"target": tgt.numpy(), "source": src.numpy(), "weight": weight.numpy()}).to_csv(
        folder / "edges.csv", index=False)
    return len(rows), len(tgt)
