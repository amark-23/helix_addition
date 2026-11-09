import torch
import importlib.metadata as md
import numpy as np
from helixadd.model import MODEL_NAME, TRANSCODER_SET
from helixadd.prompts import PREFIX, all_pairs, make_prompt
from helixadd.results import load_arrays_if_exists, save_arrays

@torch.no_grad()
def eq_activations(model,prompt):
    # run get_activations on the prompt
    _, acts = model.get_activations(prompt) #run prompt, keep activations
    eq = acts[:, -1, :]                     #keep the "=" token activations
    layers, features = torch.nonzero(eq, as_tuple=True)
    values = eq[layers, features]
    return layers.cpu(), features.cpu(), values.float().cpu() #move to cpu from gpu 


def collect(model, folder, n=None):
    params = {
        "model": MODEL_NAME, "transcoders": TRANSCODER_SET, "dtype": "float16",
        "prefix": PREFIX, "pairs": "all_pairs()" if n is None else f"all_pairs()[:{n}]",
        "position": "last",
        "circuit_tracer": md.version("circuit-tracer"),
        "transformers": md.version("transformers"),
        "torch": md.version("torch"),
    }
    data = load_arrays_if_exists(folder, params)
    if data is not None:
        return data

    pairs = all_pairs() if n is None else all_pairs()[:n]
    all_prompt, all_layer, all_feature, all_value = [], [], [], []
    for i, (a, b) in enumerate(pairs):
        layers, features, values = eq_activations(model, make_prompt(a, b))
        all_layer.append(layers)
        all_feature.append(features)
        all_value.append(values)
        all_prompt.append(torch.full((len(layers),), i))
        if i % 1000 == 0:
            print(i, "of", len(pairs))

    data = {
        "prompt": torch.cat(all_prompt).numpy(),
        "layer": torch.cat(all_layer).numpy(),
        "feature": torch.cat(all_feature).numpy(),
        "value": torch.cat(all_value).numpy(),
        "a": np.array([a for a, b in pairs]),
        "b": np.array([b for a, b in pairs]),
    }
    save_arrays(folder, params, **data)
    return data