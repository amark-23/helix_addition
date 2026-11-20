# Progress

## Status (2026-11-12)

Q1 is complete: `notes/q1_results.md`. Open leads: `notes/leads.md`.
Q2 frozen 18-11-26 (preregistration section 9); main run done 19-11-26
(`results/q2_runs/`, `scripts/q2_analysis.py`, `results/q2_analysis/`).
Next: the descriptive runs of 9.8 (see the 2026-11-19 entry).

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

**2026-11-16.** Q2 step 1. Residual stream on the number tokens at 17 points
(`helix.collect_numbers`, `notebooks/06_number_acts.ipynb`,
`results/number_acts/`); for b, the mean over 20 a's. Spectrum
(`helix.spectrum`, `scripts/number_spectrum.py`, `figures/number_spectrum.png`,
`figures/number_spectrum_L10.png`), straight line removed first: the same
periods as in Kantamneni's models stand out (T = 2, 5, 10, 100, plus 50).
The line and T = 100 are the largest single source of difference between
numbers at layers 3-5; T = 10 and T = 5 grow to a similar size at layers 9-12;
T = 2 is present from layer 0 on. Shares of variance only, not use. Slow
non-line structure (k = 3-6) remains: suspect the 0-9 vs 10-99 token split.

Helix fit and patching (`helix.basis`, `fit`, `predict`, `patch_logits`,
`run_patching`, `run_corrupted`; `notebooks/07_helix_patching.ipynb`;
`results/helix_patching/`, `results/helix_patching_corrupted/`). Fit on 80
numbers, patched only on prompts with the other 20.
- After layer 10 the operand token no longer matters: replacing it changes
  nothing. Only the embedding to after L10 can be tested.
- Replacement (0 = average vector, 1 = clean): helix 0.50-0.64, circles only
  slightly higher, line only below 0, PCA 9 (sees the test vector) similar.
- Paper's setup (run a' + b, patch the clean a; share of the full-vector
  patch): helix 0.69-0.74, circles only 0.70-0.76, line only 0.15-0.38,
  PCA 9 0.61-0.89. Kantamneni's GPT-J table gives about 0.86 for the helix
  (computed from their Table 1). Same jump from the embedding (0.69) to after
  L0 (0.74) as their layer 0 vs layer 1.
- About 80% of the top-9 PCA variance lies inside the 8 helix directions
  (random directions: 0.5%).

**2026-11-18.** Q2 design drafted (preregistration section 9): swap a circle
group (digit: T = 10, 5, 3.3, 2.5; slow: T = 100, 50) from a' into a + b, at
every point from the embedding to after L10, and measure how far each Q1
supernode moves towards the real a' + b run. Floor: random number directions;
ceiling: whole-vector swap. Circle directions are fitted per point: they
rotate across layers (embedding vs after L0 share about 20%, later
neighbours 50-80%). Digit and slow directions barely overlap (max squared
cosine under 0.09, except 0.107 after L6, where slow is orthogonalized).

**2026-11-19.** Q2 main run (998 prompts, 100 random sets per swap).
- Primary passes for a and b. Slow swap -> magnitude shift 0.96/0.95 (random
  p95 0.04-0.05), exact a' + b answer on 95-96%. Digit swap -> last-digit
  shift 0.64/0.63 (p95 0.32), exact answer on 40% (p95 16%).
- Secondary (off-target change of the other supernode vs random p95):
  a digit 0.163 vs 0.160, a slow 0.127 vs 0.121, b digit 0.155 vs 0.177,
  b slow 0.148 vs 0.128. Claims (9.7): a "both supernodes depend on both
  groups"; b "the slow circles feed both supernodes". Excess over random is
  0.01-0.03, against own-supernode effects of 0.37 and 0.92 above random.
- In plain terms: magnitude reads the slow circles almost entirely; last digit
  reads the digit circles for about 2/3, the rest comes from outside the helix.
- Q1's last digit -> magnitude link is not tested here: the digit swap keeps
  the carry status fixed by design.
Next, descriptive (9.8): the clamp test (required, digit swap moved magnitude
on a), each period alone, one point at a time, sub-helix patching. Additions:
watch the comparison supernode too; a digit swap that flips the carry.
To delete by hand: results/q2_results.zip (already unpacked).

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
