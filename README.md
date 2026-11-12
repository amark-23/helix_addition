# helixaddition

Does Llama-3.2-1B add through parallel pathways (a last-digit and a magnitude
pathway, as in Claude 3.5 Haiku), and which part of the operand representation
does each pathway read? Transcoder features, attribution graphs and
interventions; Kaggle T4 for the model runs.

The design, all decisions and the decision rules: `preregistration.md`.

## Layout

```
src/helixadd/        the package
  prompts.py         prompt template, operand pairs, carry
  model.py           load the model (circuit-tracer), generate, answer logits
  readout.py         parse the model's answer
  clean.py           clean-run accuracy
  activations.py     feature activations at "=" (sparse)
  scoring.py         R^2 of group means for many features at once
  labels.py          variable library, label rules, score_features
  plots.py           operand plots
  results.py         save / load results with their parameters
tests/               pytest
notebooks/           Kaggle runners (model on GPU)
scripts/             steps run on the saved results
  compare_templates.py   clean-run errors, few-shot vs zero-shot
  score_features.py      score and label every feature at "="
  final_labels.py        automatic labels + manual review -> final labels
  label_examples.py      figure: a sample of each label
  relaxed_labels.py      looser labels for the robustness variant
  q1_analysis.py         Q1 suppression results (decision rules)
  q1_graphs_analysis.py  Q1 attribution graph summary
  exact_kinds.py         exploratory: exact-value unclear features
  checks/                blind checks, manual review, taxonomy PDF
results/             saved results, each with params.json (data.npz files are not in git)
figures/             label examples, taxonomy/taxonomy.pdf, exploratory plots
notes/               progress log, Q1 results, open leads, conventions, taxonomy, review criteria
```

## Order

1. `notebooks/` on Kaggle: clean accuracy, then activations at "=".
2. `scripts/score_features.py`, the checks in `scripts/checks/`, then
   `scripts/final_labels.py`.
3. `notebooks/03_q1_runs.ipynb` (suppression) and `notebooks/04_graphs.ipynb`
   (attribution graphs) on Kaggle, then `scripts/q1_analysis.py` and
   `scripts/q1_graphs_analysis.py`. Results: `notes/q1_results.md`.

Tests: `pytest` (the tokenization tests need the Llama tokenizer and a
Hugging Face login).
