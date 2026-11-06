import sys
sys.path.insert(0, "src")

from transformers import AutoTokenizer
from helixadd.prompts import make_prompt

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

for prompt in [make_prompt(36, 59, few_shot=False),
               make_prompt(3, 5, few_shot=False),
               make_prompt(36, 59)]:
    ids = tok(prompt).input_ids
    print(len(ids), tok.convert_ids_to_tokens(ids))
    print()