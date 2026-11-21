# Preregistration

Status: Q1 frozen 11-11-26. Q2 frozen 18-11-26 (section 9).
Q1 freeze: the commit that sets this line to "Q1 frozen on <date>".
Any change to Q1 after the freeze goes in "Deviations" at the bottom, with the reason.

Learning notes are in `notes/`. This file holds decisions only.

---

## 1. Questions

### Q1: Does Llama-3.2-1B add through parallel pathways?

Using the public transcoders for Llama-3.2-1B, we ask whether its addition
features form separate pathways like those of Haiku: a last-digit pathway and
at least one magnitude pathway, each of which changes a different part of the
answer when it is suppressed.

- Decision rules: section 7.

### Q2: Is each pathway fed by its own part of the helix?

Design: section 9. Split into:
- Q2a: which part of the operand representation each pathway reads;
- Q2b: whether that part is a Fourier code specifically.

Current wording: if the pathways exist, we ask whether the last-digit pathway
depends on the periodic components of the operand representation (T = 2, 5, 10
and their harmonics) and the magnitude pathway on its slow components (the
linear term, T = 100 and above), and which specific periods each pathway reads.

---

## 2. Model and prompts

- Model: Llama-3.2-1B.
- Transcoders: the public per-layer transcoders (one per layer, 16 total),
  `mntss/transcoder-Llama-3.2-1B` (circuit-tracer short name "llama").
- Library: circuit-tracer v0.5.3, installed from the GitHub tag, TransformerLens
  backend. All runs (clean and intervened) go through the same ReplacementModel.
- dtype: float16 (float32 does not fit on one 16 GB T4).
- Decoding: greedy (`do_sample=False`), exactly 4 new tokens.
### Main template (few-shot)

A fixed prefix of 4 solved examples, identical in every prompt, then the
question. As a Python string:

```python
PREFIX = ("calc: 31+45=76\n"
          "calc: 17+29=46\n"
          "calc: 52+86=138\n"
          "calc: 63+78=141\n")
prompt = f"{PREFIX}calc: {a}+{b}="
```

The examples cover one prompt of each kind:

|                    | no carry   | carry      |
|--------------------|------------|------------|
| **2-digit answer** | 31+45=76   | 17+29=46   |
| **3-digit answer** | 52+86=138  | 63+78=141  |

No units digit and no tens digit repeats across the 8 example operands.

Note: the submitted proposal says 3 examples. Changed to 4 before the freeze
so the prefix shows 3-digit answers, which are 49.5% of all prompts.

### Robustness template (zero-shot)

`calc: {a}+{b}=`, the format used for Haiku. Q1 is rerun on it as a check
that the prefix does not drive the results.

### Operands

- a, b in 0..99, all 10,000 pairs (single-digit operands included).
- Excluded: the 4 example pairs and their swaps (31+45, 45+31, 17+29, 29+17,
  52+86, 86+52, 63+78, 78+63). 8 prompts.
- Prompts with a single-digit operand (about 19%) are tagged. Main results are
  reported with and without them.
- Carry: (a mod 10) + (b mod 10) >= 10. 45% of pairs. Every result is reported
  separately for carry and no-carry prompts.
- Subsets are sampled half carry, half no-carry, with `sample_pairs`:
  - suppression runs: 1,000 prompts (`sample_pairs(1000, seed=0)`), minus the
    ones the model answers wrongly in the clean run;
  - attribution graphs: 200 of those 1,000 (100 carry, 100 no-carry).

### Tokenization (verified 2026-11-06, `tests/test_prompts.py`)

- Zero-shot: `[<|begin_of_text|>, calc, :, Ġ, 36, +, 59, =]`, 8 tokens.
  `Ġ` is the space, as its own token.
- Every operand 0..99 is one token, single digits included. The 3-digit
  example answers (138, 141) are one token too.
- Few-shot: 44 tokens for every pair (BOS + 4 example lines of 9 tokens,
  `Ċ` = newline, + 7 for the question).
- Fixed positions: operand a at -4, operand b at -2, the question's "=" at -1.
- The few-shot prompt has 5 "=" tokens. Code always uses position -1 and never
  searches for "=".

### Accuracy and wrong answers

- Accuracy is measured on all prompts before any intervention.
- Template decision (2026-11-08): few-shot kept as main. Clean accuracy is
  99.75% few-shot and 99.77% zero-shot (results/clean_accuracy*/), so the
  prefix is not needed for accuracy; it is kept because it fixes the output
  format (answer, newline, "calc:").
- Prompts the model gets wrong in the clean run are excluded before any
  intervention, separately per template. The count and the excluded pairs are
  reported.

---

## 3. Feature labels (Q1)

Done before the freeze and before any suppression (2026-11-09 to 2026-11-11).
Looking at features is allowed before the freeze; outcomes are not.

### 3.1 Which features

- Every transcoder feature's activation at the question's "=" (position -1),
  on all 9,992 prompts (`activations.collect`, results/eq_activations/).
  10,839 features fire there at least once.
- Scored: the 5,043 that fire on at least 20 prompts. 20 keeps exact-sum
  features (every sum from 19 to 179 has at least 20 prompts) and is enough
  points to see a pattern. Features below it are counted, not labeled.

### 3.2 Score

For a candidate variable v(a, b), R^2 = 1 - SS_within / SS_tot, where each
prompt is predicted by the average activation of its group under v
(eta squared). Computed twice per feature and variable, on the activation
values and on the on/off pattern (1 where the feature fires); the higher
counts. Code: `helixadd.labels` (r2_all, range_r2, r2_additive), tested
against the plain definition (tests/test_labels.py).

### 3.3 Variable library

Each variable has a role (Anthropic's kinds, from the operand-plot shape) and
a family (the kind of condition).

| variable | role | family |
|---|---|---|
| a mod 10, b mod 10 | add function | last digit |
| a // 10, b // 10, a range, b range | add function | magnitude |
| units separate (g1(a mod 10) + g2(b mod 10)) | add function | last digit |
| tens separate (g1(a // 10) + g2(b // 10)) | add function | magnitude |
| sum mod 10 | sum | last digit |
| sum // 10, sum range | sum | magnitude |
| sum mod 100 | sum | mod 100 |
| units (a, b) jointly, carry | lookup table | last digit |
| tens (a, b) jointly | lookup table | magnitude |
| min // 10, max // 10, min range, max range, diff range, a < b, a = b | comparison | comparison |
| max mod 10, min mod 10 | comparison | last digit |

Ranges: every interval [lo, lo + L) with L in {5, 10, 20, 40}, scored as a
2-group variable; the best one counts.

### 3.4 Rules

- Nested: a variable that contains another (knowing it tells you the other;
  checked on the grid by `contains`, listed in NESTED and tested) counts only
  if it beats what it contains by 0.05.
- Interaction: a joint lookup (units (a, b), tens (a, b)) counts only if it
  beats its additive model by 0.1.
- tens (a, b) must also beat every other variable by 0.2.
- Label: the role and family of the best remaining variable, if its R^2 is at
  least 0.6 and it leads every other role (family) by at least 0.15.
  Otherwise "unclear" (R^2 too low) or "mixed" (no clear lead).
- Mostly active: features firing on more than 90% of prompts get the role
  "mostly active" (Anthropic's frequency rule), whatever their shape.
- Thresholds favor precision: a wrong label in a supernode is worse than a
  missing one.

### 3.5 Checks by hand

- Blind check 1 (75 features, labeled by hand without seeing the
  automatic labels): role agreement 77%; comparison 15/15, add function
  14/15, sum 14/15, lookup 8/15, mostly active 7/15 (then at a 50% rule).
- Two fixes from it: mostly active 50% -> 90%; the 0.2 rule for tens (a, b).
- Blind check 2 (30 new features, weighted toward the hard cases): role
  agreement 73%; of the 10 features the fixes changed, 7 agree (1 with the
  old labels). Lookups via tens (a, b) still 11/24 over both rounds.
- Manual review: every lookup found through tens (a, b) (145) was reviewed by
  hand against written criteria (notes/manual_review.md; pages drawn by
  scripts/checks/review_sheet.py):
  53 kept, 92 dropped. Dropped features become "unclear".
- Manual review 2: every add function feature (41) was reviewed by hand, since
  the 19 that fire on 50-90% of prompts were never blind-checked: 15 kept, 26
  dropped (all 19 dense ones dropped, mostly for an a = b structure).

### 3.6 Final labels

results/feature_labels/ (`scripts/final_labels.py`): automatic labels, with
the tens (a, b) lookups and the add function features replaced by the manual
reviews. 145 comparison, 233 sum, 81 lookup table (53 of them hand-confirmed
magnitude lookups), 15 add function (all hand-confirmed), 315 mostly active,
104 mixed, 4,150 unclear.

### 3.7 Known limits

- Precision is favored over coverage: many real features are "unclear"
  (clear shapes just under the bar, two conditions at once).
- Llama at "=" has no clean Anthropic-style "add 9" features; its add
  function features are mostly "b is a single digit".
- The magnitude lookups and the add function features depend on a manual
  step; Q1 is also run without the hand-confirmed lookups (section 4.2).

---

## 4. Supernodes and suppression

### 4.1 Supernodes

A supernode is all features of one family (final labels, results/feature_labels/),
whatever their role: a pathway carries one kind of information through every
role (add function, lookup table, sum). A feature with a clear family but a
"mixed" role belongs to its family. Features at "=" only.

| group | features | made of | use |
|---|---|---|---|
| last digit | 108 | 56 comparison (max/min mod 10), 28 lookup, 14 sum, 10 mixed role | primary test |
| magnitude | 98 | 53 lookup (hand-confirmed), 27 sum, 15 add function (hand-confirmed), 3 mixed role | primary test |
| mod 100 | 151 | 151 sum | secondary, reported |
| comparison | 89 | 89 comparison | secondary, reported |
| mixed family | 132 | 91 mixed role, 41 sum with an unclear family | secondary, reported |

Not in any supernode: mostly active (315; their role comes from frequency,
and switching off 315 always-on features would mostly break the model) and
unclear (4,150).

Coverage per prompt (active features at "="): last digit median 35 (min 12),
magnitude median 18 (min 8).

### 4.2 Robustness variants (same tests, reported next to the main ones)

- Last digit without the 56 max/min mod 10 features (52 left).
- Magnitude without the 53 hand-confirmed lookups (45 left).
- Relaxed groups: labels with R^2 >= 0.5 instead of 0.6 and every tens (a, b)
  lookup counted without manual review. If the effects grow with the groups,
  the unlabeled features are likely more of the same.

### 4.3 Predictions for the secondary groups

- mod 100: these features know the last two digits of the sum, so suppressing
  them should change both digits: mostly category B (not U or M alone).
- mixed family: carries more than one kind of information; expected B or a
  spread over U, M and B. No pass/fail.

### 4.4 Suppression method

- Mechanism (circuit-tracer `feature_intervention`): at each layer, the
  transcoder reads the current MLP input and computes the feature's
  activation a. The code adds (v - a) x the feature's decoder vector to the
  MLP output. The real MLP keeps running.
- Value: v = 0 (feature off). Transcoder features are 0 on most tokens, so
  this is a state the model sees often.
- Position: "=" only, where the labels were assigned.
- Layers: every layer where the supernode has features, all at once. Each
  feature is clamped at its own layer on the already-changed input, so it
  cannot switch back on.
- Attention: live (`freeze_attention=False`). The default is True, and in
  generation the freeze applies only to the first token, so the default would
  mix frozen and live steps.
- The suppression stays in place on the prompt positions while the answer
  tokens are generated (see section 5). To verify in code.
- Random groups (section 6) use exactly the same method.

---

## 5. Readout and outcome categories

Answer readout:
- Greedy generation, at most 4 new tokens. The raw generated text is saved
  for every prompt, so the parsing rule can be changed without a rerun.
- Primary rule (lenient): skip whitespace at the start; if the next character
  is a digit, read digits up to the first non-digit, and that is the answer.
  Otherwise (or empty output): no answer, category X.
  Examples: "95" -> 95, " 95" -> 95, "95+3" -> 95, "9 5" -> 9,
  "abc95" -> X, "" -> X.
- Sensitivity check (strict): the number must also be followed by a newline
  or the end of the output, otherwise X. "95+3" -> X, "95\ncalc" -> 95.
  Results are reported under both rules if they differ.

Outcome categories, comparing the readout to the correct answer:
- units = answer mod 10
- upper = floor(answer / 10)   (95 -> 9, 126 -> 12)

|                    | upper same     | upper changed   |
|--------------------|----------------|-----------------|
| **units same**     | N: no change   | M: upper only   |
| **units changed**  | U: units only  | B: both         |

- X: output does not start with a number.

Every outcome is in exactly one of N, U, M, B, X, so the 5 rates sum to 100%.
Rates are reported separately for carry and no-carry prompts.

Secondary measure:
- Drop in the logit of the correct answer token at the first generated
  position (before minus after suppression). Catches effects too weak to
  change the top token.

---

## 6. Controls

- Random feature groups of the same size as each supernode, drawn from the
  scored features that are not mostly active (5,043 minus 315), matched on
  layer: each random group has as many features in each layer as the
  supernode it is compared with. Drawing from all features would mostly pick
  features that rarely fire and make the comparison too easy.
- 100 random groups per primary supernode (last digit, magnitude); 100 per
  robustness variant size as well. Same 1,000 prompts, same method.
- Compute rule, fixed now: if a timing run (no supernode involved) shows the
  plan does not fit the available GPU time, the number of random groups is
  cut to 50 before the number of prompts is reduced.
- Uncertainty: bootstrap over prompts (1,000 resamples), 95% intervals for
  every rate.
- The X rate under random groups is the baseline for "the model was broken
  rather than a pathway removed".

---

## 7. Decision rules

"Random groups" below means the random feature groups of section 6, of the
same size as the supernode being tested. "p95" means the 95th percentile of a
measure across those groups.

### 7.1 Primary: output double dissociation (Q1)

Last-digit supernode:
- On no-carry prompts, suppressing it gives U above the p95 of U under random
  groups, and M no higher than the p95 of M under random groups.
- No-carry only, because on carry prompts losing the units digits can also
  lose the carry, so a change in the upper part is expected there.

Magnitude supernode:
- On all prompts (carry and no-carry), suppressing it gives M above the p95 of
  M under random groups, and U no higher than the p95 of U under random groups.
- All prompts, because the units digit depends only on the two units digits,
  so the magnitude pathway should never change it.

The primary rule passes only if both supernodes pass.

### 7.2 Secondary: internal separation

- Suppress one supernode and measure the activations of the other supernode's
  features at "=".
- Passes if the change is no larger than the p95 of the change caused by
  random groups, in both directions.
- Measure of "change": for each prompt, |S_after - S_before| / S_before,
  where S is the summed activation of the other group's features at "=";
  averaged over prompts. The same measure is computed for every random group
  (of the suppressed group's size) to give the p95.

### 7.3 Descriptive: graph structure (no pass/fail)

- On the 200 attribution graphs: edge weight between the two supernodes,
  compared to edge weight within each.
- Share of the influence on the answer that flows through each group (and
  through mixed, unclear, mostly active and error nodes). Reported next to
  every suppression result as the coverage of the labels.
- Evidence from the replacement model, not a test.

### 7.4 Prediction about the carry

- Suppressing the last-digit supernode gives a higher B rate on carry prompts
  than on no-carry prompts.
- If B is just as high on no-carry prompts, suspect labeling (mixed features
  in the supernode) rather than the carry.

### 7.5 What we can claim

| Primary | Secondary | Claim |
|---------|-----------|-------|
| pass    | pass      | separate pathways |
| pass    | fail      | the supernodes affect different digits, and interact internally |
| fail    | any       | no evidence of separate pathways in the labeled features |

Scope: the five groups hold the clearly labeled features only (about 43% of
the activation at "="; mostly active about a third, unclear the rest). So:
- a pass is meaningful despite partial coverage (each group is compared with
  random groups of the same size), but the claim is about these features, not
  about Llama's complete pathways;
- a fail is ambiguous: no separate pathways, or the labeled features are too
  small a part of them. It is not read as "Llama has no pathways".
The share of graph influence through each group (7.3) is reported next to
every suppression result, so a reader can judge the coverage.

### 7.6 Exploratory (not part of the decision)

- Sufficiency: keep only one pathway and check whether its digit survives.
  Run only if time allows, and labeled exploratory.

---

## 8. Deviations

### 8.1 Suppression implementation (2026-11-11, before any suppression run)

Section 4.4 names circuit-tracer's `feature_intervention` as the mechanism.
It handles one prompt at a time (about 0.5 s per prompt on a T4), which makes
the planned runs take 40-80 GPU hours. The same operation is implemented
batched in `helixadd.intervene` (`suppress_generate`): at each layer with
features in the group, the features' activations at position 43 are computed
with the library's own encoder (`transcoders.encode_layer`), and
(0 - activation) x decoder vector is added to the MLP output there; live
attention; greedy generation of 4 tokens, rerunning the full sequence at
every step (the library uses a KV cache).

Check against the library (`check_against_library`, 32 prompts from
`sample_pairs(32, seed=123)`, not the main sample; agreement only, no
answers shown): same parsed answer on 32/32 prompts for the clean run, 32/32
with the last-digit group off, 31/32 with the magnitude group off; largest
difference in the correct answer's logit 0.031. The one disagreement is a
tie: the top two tokens differ by 0.016 in both implementations, below
float16 precision, so their order depends on the order of additions.

Nothing else in the design changes.

### 8.2 Random groups: which features they may contain (2026-11-11, before any suppression run)

Clarifies section 6, which did not say this. A random group compared with a
supernode never contains that supernode's own features (otherwise it is
partly the group itself). It may contain features of the other groups (e.g.
magnitude features in a random group for last digit): the null is "any
non-dense features with the same layer profile", which a random draw could
hit by chance.

Measured speed of the batched implementation: 46 ms per prompt on a T4
(64-prompt batches), about 8 GPU hours for the full plan, so the compute rule
in section 6 does not apply: 100 random groups per size, 1,000 prompts.


### 8.3 Relaxed labels: exact definition (2026-11-11, before any suppression run)

Clarifies section 4.2. The relaxed labels are the automatic labels with the
R^2 bar at 0.5 instead of 0.6, every other rule unchanged, and no manual
review applied (so every tens (a, b) lookup and every add function feature
counts). Code: scripts/relaxed_labels.py, results/feature_labels_relaxed/.
Every strict supernode feature is also in the relaxed one. Sizes: last digit
145 (strict 108), magnitude 326 (strict 98; most of the difference is the
unreviewed tens (a, b) lookups and add function features).

---

## 9. Q2

Q2a: which parts of the operand helix each Q1 supernode depends on.

### 9.1 What step 1 found (exploratory, before this section)

Done before writing this section, on the residual stream at the two operand
tokens (positions 40 and 42; for b the mean over 20 a's), at 17 points
(embedding, after each layer). Code: `helixadd.helix`, notebooks 06 and 07,
`notes/progress.md` (2026-11-16).

- Spectrum (straight line removed): clear peaks at T = 100, 50, 10, 5, 3.3,
  2.5 and 2. Nothing else stands out.
- After layer 10 the operand token no longer matters: replacing its vector
  with the average leaves the answer's logit unchanged.
- A helix (line + T = 2, 5, 10, 100), fitted on 80 numbers and patched in for
  the other 20 (a' + b runs, as in Kantamneni and Tegmark), recovers about
  70-74% of the full vector's effect; circles alone the same; PCA with 9
  dimensions 61-89%.

These results set the periods and the points below.

### 9.2 Circles

- Periods: T = 100, 50, 10, 5, 3.3, 2.5, 2 (the spectrum peaks of 9.1).
- At each point and for each operand: fit the line and all 7 periods on all
  100 numbers (`helix.fit`). The directions of period T are its cos and sin
  rows of C, orthonormalized (`helix.circle_directions`). For T = 2 only the
  cos row is non-zero, so that circle is 1 direction.
- Groups:
  - digit circles: T = 10, 5, 3.3, 2.5 (8 directions);
  - slow circles: T = 100, 50 (4 directions).
- Overlap check, before the freeze: the angles between the digit and slow
  directions at each point. If any squared cosine is above 0.1, the slow
  directions are made orthogonal to the digit directions before use, and
  this is recorded here.
  Done 2026-11-18 (operand a): largest squared cosine 0.017-0.087 at every
  point except after layer 6 (0.107). So at that point the slow directions
  are made orthogonal to the digit directions; everywhere else they are used
  as fitted. The same check is run for b before the freeze.
- The directions are fitted per point because the helix rotates across
  layers: between the embedding and after layer 0 the circles share about
  20% of their directions, between later neighbouring points 50-80%, and by
  layer 3 almost none with the embedding.

### 9.3 Intervention: swap a circle

- Clean prompt a + b. On a's token, the part of the vector along a circle's
  directions D is set to that of another number a':
  x <- x - ((x - x') D^T) D, where x' is the collected vector of a' at the
  same point (`results/number_acts/`). Everything outside D stays as it was.
- Digit swap (digit circles): a' has the same tens digit as a, a different
  units digit, and a' + b has the same carry status as a + b. Predicted
  answer: a' + b (units change, tens stay).
- Slow swap (slow circles): a' has the same units digit as a and a different
  tens digit. Predicted answer: a' + b (tens change, units stay).
- a' is drawn once per prompt (seed 0) among the numbers that qualify.
- Points: every point from the embedding to after layer 10 at once (12
  points), so later layers cannot rebuild the circle.
- The same for b (b' on b's token, b's directions and vectors), as a separate
  run. Results are reported per operand.

### 9.4 Prompts

The 998 evaluation prompts of Q1. Each also needs the real a' + b run (and
a + b'), which gives the target activations in 9.5.

### 9.5 Measures

For a Q1 supernode S (last digit, magnitude) and one prompt, s is the vector
of S's feature activations at "=" (first generated position):
s_clean in a + b, s_target in the real a' + b, s_swap after the swap.

- Shift (towards the target): (s_swap - s_clean) . (s_target - s_clean)
  / |s_target - s_clean|^2. 0 = unchanged, 1 = like the real a' + b run.
  Used for the supernode whose digit the swap should change: last digit
  for the digit swap, magnitude for the slow swap.
- Off-target change: |(s_swap - s_clean) - shift x (s_target - s_clean)|
  / |s_clean|: the movement that is not towards the real a' + b run. Used
  for the other supernode (magnitude for the digit swap, last digit for the
  slow swap). Movement that matches the real target is allowed: e.g. the
  last-digit supernode holds max/min mod 10 features, whose values change
  with the tens. If the target equals the clean run, all movement counts.
- Change: |s_swap - s_clean| / |s_clean|, as in 7.2. Reported, not used in
  a rule.
- Both averaged over prompts.

Context (no decision): outcome categories of section 5 and the share of
prompts that give exactly the predicted answer a' + b.

### 9.6 Reference points

- Ceiling: swap the whole vector (x <- x') at the same points. Shows how far
  a swap on the operand token can move things at all; information about a
  already copied to later positions before layer 10 is not swapped.
- Floor: swap along random directions of the same number (8 for digit, 4
  for slow), drawn at each point from the space the 100 number vectors span,
  after removing the digit and slow directions from it. 100 random sets per
  size. Drawing from the whole residual stream would mostly hit directions
  that carry nothing about numbers and make the comparison too easy (the
  same reason as section 6).
- "p95": the 95th percentile of a measure across the random sets.
- Uncertainty: bootstrap over prompts, as in section 6.

### 9.7 Decision rules

Primary: each supernode depends on its circles (per operand).
- Digit swap: shift of the last-digit supernode above the p95.
- Slow swap: shift of the magnitude supernode above the p95.
- Passes if both pass.

Secondary: separation (per operand).
- Digit swap: off-target change of the magnitude supernode no higher than
  the p95.
- Slow swap: off-target change of the last-digit supernode no higher than
  the p95.

| Primary | Digit swap moves magnitude | Slow swap moves last digit | Claim |
|---------|----------------------------|----------------------------|-------|
| pass    | no                         | no                         | each supernode depends on its own circles |
| pass    | yes                        | no                         | the digit circles feed both supernodes; the slow circles only magnitude |
| pass    | no                         | yes                        | the slow circles feed both supernodes; the digit circles only last digit |
| pass    | yes                        | yes                        | both supernodes depend on both groups; no split along the helix |
| fail    | any                        | any                        | no evidence that the supernodes depend on these circles |

Expected: the digit swap may move the magnitude supernode (Q1: the last-digit
group feeds the magnitude group). This test cannot tell whether that is
direct or through the last-digit supernode (see 9.8).

Scope, as in 7.5: the claim is about the labeled supernodes. A fail is
ambiguous: the model may use the same information in directions the helix
fit misses (about a quarter of the full vector's effect in 9.1).

### 9.8 Descriptive (no pass/fail)

- Each of the 7 periods swapped alone (a' chosen to differ in that period's
  position): shift and change of both supernodes, and outcomes.
- The 2 groups, one point at a time (12 points): where each supernode depends
  on each group. Floor for the 2 groups only, 20 random sets per point, on
  200 of the prompts.
- Sub-helix patching (Kantamneni-style): into a' + b, the whole vector of a
  replaced by the fit of a built from one group or one period only; outcome
  and shift of both supernodes.
- If the digit swap moves the magnitude supernode: the same swap with the
  last-digit supernode held at its clean activations. If magnitude still
  moves, the dependence is direct.

### 9.9 Q2b: is it a Fourier code?

Not tested separately. Reported from step 1: how close the helix comes to the
full vector and to PCA in patching, and the overlap between the helix
directions and the top PCA directions (about 80%; random directions 0.5%).

### 9.10 Before the freeze

- Overlap check (9.2).
- Pilot, declared and not counted: the digit and slow swaps, the ceiling and
  5 random sets, on 50 prompts not reused in the main run if it changes the
  design (otherwise the main run uses all 998). Purpose: check the code and
  that the ceiling moves the supernodes at all.
- Timing run; compute rule of section 6 if the plan does not fit.

Pilot 1 (2026-11-18, the last 50 evaluation prompts, 5 random sets): the code
runs, the ceiling gives the predicted answer on 98-100% of prompts, 58 s for
28 runs (main run about 4.5 GPU hours). It showed one flaw in the design: the
secondary rule used the plain change of the other supernode, which counts
movement towards the real a' + b run as a failure of separation. Replaced by
the off-target change (9.5, 9.7). No other change; the pilot numbers are not
results. A second pilot checks the new measure before the freeze.

### 9.11 Compute (estimate)

Per operand: 2 swaps + ceiling + 200 random sets, 998 prompts each, plus the
real a' + b runs: about 5 GPU hours for both operands at the Q1 speed.
Descriptive runs: about 2 more hours.

### 9.12 Additions to the descriptive runs (2026-11-20, after the main run)

Descriptive only, decided after seeing the main result, so not part of any
claim in 9.7:
- The comparison supernode is watched in every descriptive run (shift,
  change, off-target).
- A digit swap that flips the carry status (same tens, other units): tests
  whether the last-digit -> magnitude link of Q1 carries the carry, which the
  main digit swap holds fixed by design.
- Pair sets for "each period alone": fast periods use a' with the same tens
  and a units difference that is odd and not 5 (so T = 2, 5, 2.5 all differ);
  slow periods use a' with the same units and a tens difference other than 5
  (so T = 50 differs). The two groups are rerun on the same pair sets so the
  periods can be compared with them.
- Sub-helix vectors are fitted on all 100 numbers (circles only, no line).

### 9.13 Post hoc runs (2026-11-20, after the descriptive runs)

Not part of any claim in 9.7. In the descriptive runs, T = 2 swapped alone
moved the last-digit supernode as far as the whole digit group (shift 0.69
vs 0.64 in the main run). T = 2 is the k = 50 harmonic of period 10, like
3.3 and 2.5, and should have been in the digit group (9.2); it was listed
separately as parity. Sub-helix patching (9.8) turned out uninformative:
vectors made of part of the helix do not look like numbers, and the model
breaks on both digits. Extra runs (`helixadd.q2.run_posthoc`, notebook 10),
all with the logits of the clean and the target answer:
- digit group + T = 2 (9 directions) swapped on the main digit pairs and the
  odd-difference pairs; floor: 20 random sets of 9 directions on the main
  pairs, drawn outside every circle and the line;
- the same group in the carry-flip swap;
- a grid on the main digit pairs: fast part (T = 10, 5, 2, or digit group +
  T = 2) x slow additions (none, T = 100, T = 50, both); the last cell is
  all circles, compared with the whole-vector ceiling;
- all circles and the line alone, swapped on the slow pairs;
- removal: each period, digit + T = 2, and slow set to the average over
  numbers on the clean prompts; floor: 5 random removals of 9 and of 4
  directions.
- comparison with Kantamneni and Tegmark: in a' + b runs, a's vector is
  replaced by the fit of the clean a built from part of the helix (the
  average plus those circles: digit + T = 2, slow, each period, all
  circles), with the clean answer's logit; the unpatched run and the real
  vector give the 0 and 100% points. The exact-answer version of 9.8 was
  uninformative; this graded version is their measure.
- Note (2026-11-21): the sub-helix fits use all 100 numbers, so the patched
  numbers are in the fit (in-sample); the step-1 helix patching (9.1) was
  out-of-sample. The two are not directly comparable.
- Note (2026-11-21): the main run's floor (9.6) drew random directions from
  the number space minus the digit and slow groups. That space still held
  the T = 2 direction and the line, so random sets could partly swap T = 2,
  which moves the last-digit supernode (floor median 0.27 for 8 directions).
  The post-hoc floor excludes every circle and the line (median 0.01 for 9
  directions). The preregistered decisions stand as computed; the main
  run's digit floor was conservative.
