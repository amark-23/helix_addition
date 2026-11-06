# Progress

## 2026-11-06

Done:
- preregistration.md: Q1 design mostly fixed (prompts, labels, suppression,
  readout, decision rules). Q2 not started.
- src/helixadd/prompts.py: PREFIX, make_prompt, is_carry, all_pairs,
  sample_pairs. Checked by hand (9992 pairs, 5/10 carry).
- scripts/explore_tokens.py: tokenization verified (results in
  preregistration section 2).
- Local setup: .venv with transformers, HF login, .gitignore.

Next:
1. Turn the tokenization facts into a test (tests/test_prompts.py, pytest).
2. Remaining Q1 decisions: label thresholds, number of random groups,
   internal-test measure, accuracy threshold, graph subset size.
3. model.py: load the model, measure clean accuracy on all prompts.
