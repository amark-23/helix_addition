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

## 2026-11-09

Template comparison (scripts/compare_templates.py):
- Few-shot: all 25 errors are +10.
- Zero-shot: 15 of 23 are +-10 (both directions); the other 8 are copying an
  operand (24+7 -> 24), dropping the hundreds (95+94 -> 89), and a few odd
  ones. 5 zero-shot outputs start with a space (handled by parse_answer).
- Conclusion: the prefix does not change how many prompts fail, it changes the
  kind of error. Few-shot errors are clean magnitude errors, which supports it
  as the main template.

Design change: look at Llama's features before fixing the label groups
(preregistration section 3). Next: collect activations at "=" for all prompts.

Feature activations at "=" (notebooks/02_feature_activations.ipynb):
- activations.eq_activations (one prompt) and activations.collect (all
  prompts, sparse, saved as data.npz + params.json).
- Full run done: 9,992 prompts, 7,546,419 entries, about 755 active features
  per prompt (828 for 36+59; layers 12-14 have the most).
- data.npz stays out of git (.gitignore). Copies: Kaggle notebook output
  (attach with Add Input) and a local download in results/eq_activations/.
  Only params.json is committed.
- Notebook workflow: no editable install on Kaggle (sys.path), update() pulls
  and reloads modules without a restart. Import modules, not functions.

Next:
1. Count distinct features that fire at "=" and how often (laptop).
2. Operand plots for a sample of frequent features: which kinds exist?
3. Then fix the label groups and rule (preregistration section 3).

## 2026-11-10

Feature labels (src/helixadd/labels.py, scripts/score_features.py):
- 5,043 features fire at "=" on >= 20 prompts (of 10,839 that fire at all).
- Each is scored (R^2) on a library of variables, on its values and on its
  on/off pattern; nested and interaction rules; two labels: role (Anthropic's
  add function / sum / lookup table, plus comparison) and family (last digit /
  magnitude / mod 100 / comparison); mostly_active is a separate flag.
- Thresholds favor precision (best R^2 >= 0.6, lead >= 0.15). 634 features
  get a clear role; 274 mixed; 4,135 unclear.
- The unclear pile (figures/unclear_gallery.png) holds clear shapes just under
  the bar, two-condition features, dense textures and sparse specks; no clear
  misclassifications seen.
- Results: results/feature_scores/ (older versions deleted).
- Blind check prepared (scripts/blind_check.py, notes/blind_check_guide.md),
  not yet done.

- Dense features (active on > 50% of prompts) now get the role "mostly active";
  add function dropped to 19 features, almost all "b is a single digit".
- Codebook for the blind check: notes/taxonomy.md and figures/taxonomy/
  (examples outside the blind set; 11 non-textbook candidates removed by hand,
  most of them lookup tables). Practice round: figures/taxonomy/practice.png,
  results/practice/.

- No Anthropic-style add function features ("add 9": a or b ends in 9, full
  lines) at "=": the method detects a synthetic one (additive R^2 0.95), no
  real feature has it. Hypothesis: max/min mod 10 features (nested squares)
  are Llama's version, the same lines cut off at a = b. To check later.
- Taxonomy reference as a PDF: figures/taxonomy/taxonomy.pdf.

Next:
1. Read notes/taxonomy.md, do the practice round
   (python scripts/blind_compare.py practice), then the blind round
   (python scripts/blind_compare.py).
2. Write the label method into preregistration section 3; settle the
   remaining Q1 entries (random groups, internal-test measure, graph subset,
   strict + relaxed threshold check). Freeze Q1.
3. Suppression code (interventions) for Q1.

## 2026-11-11

Blind check, round 1 (75 features, results/blind_check/):
- Role agreement 77% (58/75). Per automatic label: comparison 15/15,
  add function 14/15, sum 14/15, lookup table 8/15, mostly active 7/15.
- Family agreement 81% (55/68, mostly active left out). last digit 15/15,
  mod 100 11/12, magnitude 22/31.
- Lookup via units (a, b): 4/4 agree. Lookup via tens (a, b): 4/11.
  tens (a, b) is the best variable in 12 of the 17 role disagreements.
- Mostly active disagreements are features firing on 50-75% of prompts with a
  clear shape (a < b at 50.4%, max mod 10 at 62%, sum >= 100). Features firing
  on >= 96% of prompts were all agreed on.

Fixes after round 1 (scripts/score_features.py):
- mostly active: share 0.5 -> 0.9.
- tens (a, b) must beat every other variable by 0.2 (DELTA_FLEX).
- New labels in results/feature_scores/; round-1 labels kept in
  results/feature_scores_round1/ (they are what round 1 measured).
- Round 2 prepared (scripts/blind_check_2.py): 30 new features, 10 lookup,
  10 mostly active, 10 whose role changed. Taxonomy PDF and notes updated to
  the 90% rule.

Blind check, round 2 (30 new features, results/blind_check_2/):
- Role agreement 73% (22/30); sample weighted toward the hard categories.
- Features whose role changed because of the fixes: 7/10 agree now; their
  round-1 labels would have agreed on 1/10. The fixes helped.
- mostly active (90% rule): 8/10. comparison 5/5, sum 2/2.
- Lookup via tens (a, b) still weak: 7/13 (rounds 1+2: 11/24). The
  disagreements are mostly comparison shapes restricted to a range
  ("equality for numbers > 50", "for numbers < 50"). Lookup via
  units (a, b): 2/2 (rounds 1+2: 6/6).
- Magnitude family 5/11.
