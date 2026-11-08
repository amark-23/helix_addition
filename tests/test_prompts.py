import pytest
from transformers import AutoTokenizer
from helixadd.prompts import make_prompt, all_pairs

#load llama tokenizer and hand it to tests
@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

#tokenize calc: 36 + 59 = and check it splits into the tokens we expect
def test_zero_shot_tokens(tok):
    ids = tok(make_prompt(36, 59, few_shot=False)).input_ids
    tokens = tok.convert_ids_to_tokens(ids)
    assert tokens == ["<|begin_of_text|>", "calc", ":", "Ġ", "36", "+", "59", "="]

#goes through every number in (0,99) and checks each one is a single token
def test_operands_single_token(tok):
    for n in range(100):
        ids = tok(str(n), add_special_tokens=False).input_ids
        tokens = tok.convert_ids_to_tokens(ids)
        assert len(tokens) == 1, f"Number {n} is not a single token: {tokens}"

#build full fewshot prompt (9992 pairs) and check that each prompt is 44 tokens
def test_few_shot_length(tok):
    for a, b in all_pairs():
        ids = tok(make_prompt(a, b, few_shot=True)).input_ids
        assert len(ids) == 44, f"Prompt for {a} + {b} is not 44 tokens: {len(ids)}"

#check that the very last token in the prompt is always "=", for a few prompts
def test_last_token_is_equals(tok):
    for a, b in [(36, 59), (3, 5), (99, 99)]:
        ids = tok(make_prompt(a, b, few_shot=True)).input_ids
        tokens = tok.convert_ids_to_tokens(ids)
        assert tokens[-1] == "=", f"Last token for {a} + {b} is not '=': {tokens[-1]}"
