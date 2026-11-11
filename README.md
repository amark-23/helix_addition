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
  checks/                blind checks, manual review, taxonomy PDF
results/             saved results, each with params.json (data.npz files are not in git)
figures/             label_examples.png, taxonomy/taxonomy.pdf
notes/               progress log, conventions, taxonomy, review criteria
```

## Order

1. `notebooks/` on Kaggle: clean accuracy, then activations at "=".
2. `scripts/score_features.py`, the checks in `scripts/checks/`, then
   `scripts/final_labels.py`.
3. Suppression (Q1), after the preregistration is frozen.

Tests: `pytest` (the tokenization tests need the Llama tokenizer and a
Hugging Face login).
