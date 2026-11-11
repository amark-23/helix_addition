"""The feature groups that Q1 switches off (preregistration 4.1, 4.2, 6, 8.2).

A group is a DataFrame with one row per feature (columns layer, feature).
intervene.by_layer(group["layer"], group["feature"]) turns it into the form
the hooks need.
"""
import numpy as np
import pandas as pd

LABELS = "results/feature_labels/data.csv"


def load_labels(path=LABELS):
    return pd.read_csv(path)


def supernode(labels, family):
    """All clearly labeled features of one family, whatever their role
    (mostly active and unclear features are never in a supernode)."""
    clear = ~labels["role"].isin(["mostly active", "unclear"])
    return labels[clear & (labels["family"] == family)][["layer", "feature"]].reset_index(drop=True)


def random_group(labels, group, seed):
    """A random group matched on layer to `group`: the same number of features
    in every layer, drawn from the non-dense features, never from `group` itself."""
    rng = np.random.default_rng(seed)
    pool = labels[labels["role"] != "mostly active"][["layer", "feature"]]
    in_group = pool.merge(group, how="left", indicator=True)["_merge"] == "both"
    pool = pool[~in_group.to_numpy()]

    parts = []
    for layer, n in group["layer"].value_counts().sort_index().items():
        candidates = pool[pool["layer"] == layer]
        assert n <= len(candidates), f"layer {layer}: group has {n} features, pool only {len(candidates)}"
        # GAP: pick n different rows of `candidates` at random, using rng, and append them to parts
        sampled = candidates.sample(n=n, random_state=rng)
        parts.append(sampled)
    return pd.concat(parts).reset_index(drop=True)
