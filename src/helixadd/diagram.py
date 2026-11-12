"""Path diagram for one prompt: kept graph nodes grouped into boxes, edges summed between boxes."""
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
