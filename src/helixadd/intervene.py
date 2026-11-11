"""Switch off a group of transcoder features at the "=" position, many prompts at once.

Same operation as circuit-tracer's feature_intervention (value 0, live
attention, the real MLPs keep running), written for batches and for one
position: at each layer with features in the group, read the features'
activations from the MLP input, then add (0 - activation) x decoder vector to
the MLP output. Checked against circuit-tracer with check_against_library (needs the GPU).
"""
import torch

EQ = 43   # position of the question's "=" in every prompt (44 tokens; -1 stops being "=" once tokens are generated)


def by_layer(layers, features):
    """A group as {layer: tensor of feature ids}, from two equal-length lists."""
    group = {}
    for layer, feature in zip(layers, features):
        group.setdefault(int(layer), []).append(int(feature))
    return {layer: torch.tensor(feats) for layer, feats in group.items()}


def make_hooks(model, group, store):
    """Hooks that switch off the group's features at position EQ.

    group: {layer: feature ids}. store: a dict the hooks fill with each layer's
    activations of the group's features (before switching off), {layer: (batch, k)}.
    """
    hooks = []
    for layer, feats in group.items():
        W_dec = model.transcoders._get_decoder_vectors(layer, feats).to(model.cfg.device)   # (k, d_model)
        feats = feats.to(model.cfg.device)

        def read_input(x, hook, layer=layer, feats=feats):
            # x: the MLP input, (batch, positions, d_model)
            acts = model.transcoders.encode_layer(x[:, EQ, :], layer)   # (batch, all features of the layer)
            store[layer] = acts[:, feats]                                # (batch, k)
            return x

        def fix_output(out, hook, layer=layer, W_dec=W_dec):
            # out: the MLP output, (batch, positions, d_model)
            # at position EQ, add (0 - activation) x decoder vector, summed over the group's features:
            # (batch, k) @ (k, d_model) -> (batch, d_model)
            out[:, EQ, :] += (0 - store[layer]) @ W_dec
            return out

        hooks.append((f"blocks.{layer}.{model.feature_input_hook}", read_input))
        hooks.append((f"blocks.{layer}.{model.feature_output_hook}", fix_output))
    return hooks


@torch.no_grad()
def suppress_generate(model, prompts, group, max_new_tokens=4):
    """Greedy generation with the group switched off at "=", for a batch of prompts.

    Returns the generated text of every prompt and the logits at the first
    generated position (batch, vocab). An empty group gives the clean run.
    """
    from helixadd.model import tokenize

    store = {}
    tokens = tokenize(model, prompts)                 # (batch, 44)
    n_prompt = tokens.shape[1]
    first_logits = None
    with model.hooks(fwd_hooks=make_hooks(model, group, store)):
        for step in range(max_new_tokens):
            logits = model(tokens)[:, -1, :]          # the prediction for the next token, (batch, vocab)
            if step == 0:
                first_logits = logits.float().cpu()
            next_token = logits.argmax(dim=-1)                          # greedy: highest score, (batch,)
            tokens = torch.cat([tokens, next_token[:, None]], dim=1)    # append: (batch, n + 1)
    texts = [model.tokenizer.decode(row) for row in tokens[:, n_prompt:]]
    return texts, first_logits


def check_against_library(model, pairs, groups, batch=16):
    """Run the same suppressions with circuit-tracer (one prompt at a time) and with
    suppress_generate (batched), and compare. Prints agreement only, no answers.

    pairs: list of (a, b). groups: {name: group as returned by by_layer}.
    Returns a DataFrame with one row per (group, prompt).
    """
    import pandas as pd

    from helixadd.prompts import make_prompt
    from helixadd.readout import parse_answer

    rows = []
    for name, group in groups.items():
        interventions = [(layer, EQ, int(f), 0.0) for layer, feats in group.items() for f in feats]
        for i in range(0, len(pairs), batch):
            chunk = pairs[i:i + batch]
            prompts = [make_prompt(a, b) for a, b in chunk]
            ours, ours_logits = suppress_generate(model, prompts, group)
            for j, ((a, b), p) in enumerate(zip(chunk, prompts)):
                text = model.feature_intervention_generate(
                    p, interventions, freeze_attention=False, max_new_tokens=4, do_sample=False,
                    return_activations=False, verbose=False)[0]
                lib_logits = model.feature_intervention(p, interventions, freeze_attention=False,
                                                        return_activations=False)[0][0, -1].float().cpu()
                answer_id = model.tokenizer.convert_tokens_to_ids(str(a + b))
                rows.append({"group": name, "a": a, "b": b,
                             # the library returns prompt + continuation (possibly with BOS text first)
                             "same_answer": parse_answer(ours[j]) == parse_answer(text[text.index(p) + len(p):]),
                             "same_top_token": int(ours_logits[j].argmax()) == int(lib_logits.argmax()),
                             "logit_diff": abs(float(ours_logits[j, answer_id] - lib_logits[answer_id]))})
    df = pd.DataFrame(rows)
    print(df.groupby("group").agg(same_answer=("same_answer", "mean"), same_top_token=("same_top_token", "mean"),
                                  max_logit_diff=("logit_diff", "max")).round(3).to_string())
    return df
