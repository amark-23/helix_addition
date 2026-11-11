import pandas as pd

from helixadd.groups import load_labels, random_group, supernode


def test_supernode_excludes_dense_and_unclear():
    labels = load_labels()
    g = supernode(labels, "last digit").merge(labels, on=["layer", "feature"])
    assert len(g) > 0
    assert not g["role"].isin(["mostly active", "unclear"]).any()
    assert (g["family"] == "last digit").all()


def test_random_group_matches_layers_and_avoids_the_group():
    labels = load_labels()
    group = supernode(labels, "last digit")
    r = random_group(labels, group, seed=0)
    assert r["layer"].value_counts().sort_index().equals(group["layer"].value_counts().sort_index())
    assert len(r.merge(group)) == 0                                  # none of the group's own features
    assert not r.duplicated().any()                                  # no feature twice
    dense = labels[labels["role"] == "mostly active"][["layer", "feature"]]
    assert len(r.merge(dense)) == 0                                  # no mostly active features


def test_random_group_is_reproducible():
    labels = load_labels()
    group = supernode(labels, "magnitude")
    assert random_group(labels, group, seed=7).equals(random_group(labels, group, seed=7))
    assert not random_group(labels, group, seed=7).equals(random_group(labels, group, seed=8))
