# Progress

## Status (2026-11-12)

Q1 is complete: `notes/q1_results.md`. Open leads: `notes/leads.md`.
Next: design Q2 (which helix components each pathway reads).

## Log

**2026-11-06.** Q1 design drafted in `preregistration.md`. `prompts.py`
(prefix, make_prompt, is_carry, pairs); tokenization checked by hand
(9,992 pairs, half with carry).

**2026-11-08.** Tests for tokenization and readout. `model.py` (fp16,
greedy, 4 tokens). Clean accuracy on a T4: few-shot 99.75% (all 25 errors
are exactly +10), zero-shot 99.77%.

**2026-11-09.** Template comparison: the few-shot prefix does not change how
many prompts fail, only the kind of error (clean +10 magnitude errors), so it
stays the main template. Feature activations at "=" for all prompts
(about 755 active features per prompt, most in L12-L14); `data.npz` is kept
out of git.

**2026-11-10.** Feature labels (`scoring.py`, `labels.py`): 5,043 features
fire on at least 20 prompts; each is scored by R^2 against a library of
variables, with precision-first thresholds. No Anthropic-style periodic add
function ("add 9") at "="; max/min mod 10 features look like Llama's version.
Taxonomy and codebook for the blind check (`notes/taxonomy.md`,
`figures/taxonomy/taxonomy.pdf`).

**2026-11-11.** Blind checks: round 1 agreement 77% on role, round 2 73% on a
harder sample. Fixes: mostly active at 90%, tens (a, b) must lead by 0.2.
Two manual reviews (`notes/manual_review.md`): tens lookups 53 keep / 92 drop,
add functions 15 keep / 26 drop. Final labels in `results/feature_labels/`.
Q1 preregistration frozen. Batched suppression (`intervene.py`, checked
against circuit-tracer: deviation 8.1), random groups (`groups.py`), relaxed
labels (deviation 8.3), runner (`q1.py`, 610 runs).

**2026-11-12.** All 610 suppression runs and 200 attribution graphs done.
Results and their reading: `notes/q1_results.md`. Exact-value unclear features
(`scripts/exact_kinds.py`) and other leads: `notes/leads.md`.

## Practical notes

- Kaggle T4: about 14.6 GB usable per GPU; model plus transcoders take about
  12.3 GB. Two GPUs do not add memory; a second one can only run a parallel
  copy.
- circuit-tracer `attribute` needs `batch_size=64` on a T4 (512 runs out of
  memory) and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` set before
  torch loads.
- Long runs: Save Version, Save & Run All. All runners are resumable.
- Kaggle imports: sys.path, not an editable install; import modules, not
  functions, so `update()` can reload them.
