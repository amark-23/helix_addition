import json
from pathlib import Path

import pandas as pd


def load_if_exists(folder, params):
    folder = Path(folder)
    if not (folder / "params.json").exists():
        return None
    saved = json.loads((folder / "params.json").read_text())
    if saved != params:
        raise ValueError(f"{folder} holds a result with different parameters:\n{saved}")
    return pd.read_csv(folder / "data.csv")


def save(folder, params, df):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    df.to_csv(folder / "data.csv", index=False)
    (folder / "params.json").write_text(json.dumps(params, indent=2))