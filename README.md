# helixaddition

How does Llama-3.2-1B add two-digit numbers? This project asks two questions,
using the public transcoders for the model (circuit-tracer):

- **Q1.** Do its transcoder features form separate pathways for the last digit
  and for the size of the answer, as described for Claude 3.5 Haiku?
- **Q2.** Which part of the operands' representation does each pathway use?
  Numbers are represented on a helix (Kantamneni and Tegmark, 2025); the
  question is which periods of that helix each pathway depends on.

Short answers (details, figures and limits in
[`results/README.md`](results/README.md)):

- Q1: two groups of features, labeled "last digit" and "magnitude", affect
  different digits of the answer. Switching off the first changes the units
  digit on 48% of no-carry prompts and never the tens; switching off the
  second changes the tens on 26% of prompts and never the units. For random
  groups of the same size, the 95th percentile is at most 6% for either
  digit.
- Q2: the last-digit features follow the fast circles of the helix
  (periods 10, 5 and 2), the magnitude features the slow circles (periods 100
  and 50). Swapping those circles between two numbers moves the magnitude
  group 0.96 of the way to the other number, and the last-digit group 0.99
  once period 2 is included (a post-hoc addition). The carry reaches the
  tens through the period-50 circle.

These results hold for the labeled features of one model, on one prompt
format.

## Setup

```
pip install -e .
pip install git+https://github.com/safety-research/circuit-tracer@v0.5.3
```

The model runs need a GPU with about 14 GB (a Kaggle T4 was used) and a
Hugging Face token with access to `meta-llama/Llama-3.2-1B`. 

## Layout

```
src/helixadd/          the package
  prompts.py           prompt template, operand pairs, carry
  model.py             load the model and transcoders, generate
  readout.py           parse the answer, outcome categories
  clean.py             clean-run accuracy
  activations.py       feature activations at "="
  scoring.py, labels.py  label every feature by the variable it tracks
  groups.py            supernodes and layer-matched random groups
  intervene.py         switch off feature groups (batched)
  q1.py, analysis.py   Q1 runs and decision rules
  graphs.py, diagram.py  attribution graph summaries, path diagram boxes
  helix.py             number vectors, spectrum, helix fit, patching
  q2.py                Q2 swaps, removal, controls
  plots.py, results.py operand plots; save results with their parameters
notebooks/             GPU runs (Kaggle), numbered in the order they were run
scripts/               analysis and figures, run on the saved results
  checks/              blind checks and manual review pages
results/               every result with the parameters that produced it
figures/               figures used in results/README.md
```

## Reproducing

| Step | Notebook (GPU) | Scripts |
|---|---|---|
| Prompts, clean accuracy | 01 | `compare_templates.py` |
| Feature activations at "=", labels | 02 | `score_features.py`, `checks/`, `final_labels.py`, `relaxed_labels.py`, `label_examples.py` |
| Q1 suppression | 03 | `q1_analysis.py`, `q1_figures.py` |
| Q1 attribution graphs | 04, 05, 12 | `q1_graphs_analysis.py`, `path_diagram.py`, `path_diagram_fine.py`, `exact_kinds.py` |
| Q2 number representations | 06, 07 | `number_spectrum.py`, `helix_patching_figure.py` |
| Q2 swaps | 08, 09, 10, 11 | `q2_analysis.py`, `q2_descriptive_analysis.py`, `q2_figures.py` |

Every notebook clones this repository and saves its results under
`results/<step>/` with a `params.json`. Runners skip results that already
exist. Two large files are not in git:
`results/eq_activations/data.npz` (rebuilt by notebook 02) and
`results/path_diagram/viewer/` (rebuilt by notebook 05).
