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

## 2026-11-08

Done:
- tests/test_prompts.py (tokenization facts), pyproject.toml (package installable).
- src/helixadd/readout.py + tests/test_readout.py (lenient rule, raw text saved).
- src/helixadd/model.py: load_model (fp16), tokenize, generate (greedy, 4 tokens).

Notes:
- Kaggle offers 2 T4s (16 GB each, separate). The model must fit on one.
  The second GPU can run a parallel copy if something is slow (likely graphs).

Clean accuracy (Kaggle T4, fp16, circuit-tracer v0.5.3):
- Few-shot: 99.75%, 25 wrong. All 25 are exactly +10 (units right, upper +1),
  true sums 127-143. Two prefix answers (138, 141) are in that range.
- Zero-shot: 99.77%, 23 wrong. Only 1 pair wrong in both runs (74+58).
  Errors cluster by operand (b = 64, b = 70), not by sum range.
- Hypothesis (unchecked): the prefix decides which prompts fail.
  To check: pred and diff for the zero-shot wrong rows.

Next:
1. Check zero-shot diffs (locally, from the saved CSV).
2. Remaining Q1 decisions: label thresholds, number of random groups,
   internal-test measure, graph subset size.
3. Then: feature labels (operand plots).
