import torch
from circuit_tracer import ReplacementModel

MODEL_NAME = "meta-llama/Llama-3.2-1B"
TRANSCODER_SET = "llama" 

def load_model(dtype=torch.float16):
    return ReplacementModel.from_pretrained(MODEL_NAME, TRANSCODER_SET, dtype=dtype)

def tokenize(model, prompts):
    return model.tokenizer(prompts, return_tensors="pt").input_ids.to(model.cfg.device)

@torch.no_grad()
def generate(model,prompts, max_new_tokens=4):
    tokens = tokenize(model, prompts)
    out = model.generate(tokens, max_new_tokens=max_new_tokens, do_sample=False, stop_at_eos=False, verbose = False)
    new_tokens = out[:, tokens.shape[1]:]
    return [model.tokenizer.decode(row) for row in new_tokens]


@torch.no_grad()
def answer_logits(model, prompts, answers):
    tokens = tokenize(model, prompts)
    logits = model(tokens)[:, -1, :]
    ids = [model.tokenizer.convert_tokens_to_ids(str(x)) for x in answers]
    return logits[torch.arange(len(ids)), ids].float().cpu().tolist()