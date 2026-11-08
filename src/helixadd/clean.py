import importlib.metadata as md

import pandas as pd

from helixadd.model import MODEL_NAME, TRANSCODER_SET, answer_logits, generate
from helixadd.prompts import PREFIX, all_pairs, is_carry, make_prompt
from helixadd.readout import parse_answer
from helixadd.results import load_if_exists, save

#write down setting of the run:
#model,transcoders,float16,prefix, tokens to generate, batch size, lib versions
#check for existing result
#otherwise go through all pairs in batches(64), build 64 prompts, 4 tokens each
#get each prompts logit for correct answer
# for each pair record a,b,carry,rawtext, parsed answer, correct?, logit as a table
def run_clean(model, folder, few_shot=True, batch=64):
    params = {
        "model": MODEL_NAME, "transcoders": TRANSCODER_SET, "dtype": "float16",
        "prefix": PREFIX if few_shot else "", "max_new_tokens": 4,
        "pairs": "all_pairs()", "batch": batch,
        "circuit_tracer": md.version("circuit-tracer"),
        "transformers": md.version("transformers"),
        "torch": md.version("torch"),
    }
    df = load_if_exists(folder, params)
    if df is not None:
        return df

    pairs = all_pairs()
    rows = []
    for i in range(0, len(pairs), batch):
        chunk = pairs[i:i + batch]
        prompts = [make_prompt(a, b, few_shot=few_shot) for a, b in chunk]
        texts = generate(model, prompts)
        logits = answer_logits(model, prompts, [a + b for a, b in chunk])
        for (a, b), text, logit in zip(chunk, texts, logits):
            pred = parse_answer(text)
            rows.append({"a": a, "b": b, "carry": is_carry(a, b), "raw": text,
                         "pred": pred, "correct": pred == a + b,
                         "logit_correct": logit})
    df = pd.DataFrame(rows)
    save(folder, params, df)
    return df