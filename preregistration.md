# Preregistration

Status: draft, not frozen.
Freeze date: (set when complete)
Any change after the freeze goes in "Deviations" at the bottom, with the reason.

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

Draft, not yet worked through. To be split into:
- Q2a: which part of the operand embedding each pathway reads;
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
- Subsets (e.g. for graphs) are sampled half carry, half no-carry. Size: (open)

### Tokenization (verified 2026-11-06, `scripts/explore_tokens.py`)

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

Order (decided 2026-11-09): the groups and the label rule are NOT fixed yet.
1. Collect the activations of all features at "=" on all prompts.
2. Look at operand plots for a sample of features (and possibly cluster them)
   to see which kinds of features Llama has. Anthropic's categories (last
   digit, magnitude at different precisions, mod 100, lookup, add function)
   are things to check for, not assumptions.
3. Then fix the groups and the label rule below, and freeze.
Looking at features before the freeze is allowed. No suppression runs before
the freeze.

Draft groups (to be revised after step 2):
- Last digit: depends only on the units digits of a and b.
- Magnitude: depends smoothly on size (possibly split by precision).
- Mod 100: depends on the exact last two digits of the sum.
- Other: everything else, incl. near-constant features. Reported, not in a
  supernode.

Planned coverage measures (descriptive, no pass/fail): share of graph
influence through each group; suppressing "other" as its own group.

The rest of this section is an earlier draft and will be rewritten:

- Features are taken at the "=" token (the last position, where the answer is
  predicted).
- Operand plot: for each feature, its activation at "=" on all 10,000 prompts
  a + b with a, b in 0..99.
- For each candidate variable, fit the activations and record R².
  "Fit" means: group the prompts by the variable's value and predict each
  activation by its group average. R² is the fraction of the activation's
  variance explained by those averages.
  Candidate variables:

  |          | last-digit family | magnitude family   |
  |----------|-------------------|--------------------|
  | operand a | a mod 10         | floor(a / 10)      |
  | operand b | b mod 10         | floor(b / 10)      |
  | sum       | (a + b) mod 10   | floor((a + b) / 10)|

  Magnitude is measured with the tens digit, never with the whole number:
  the whole number contains the last digit, so it would score as high as
  a mod 10 on every last-digit feature. On the full 100 x 100 grid, a mod 10
  and floor(a / 10) are independent.
  All variables have 10 groups except floor((a + b) / 10), which has 20
  (0 to 19). With 10,000 prompts, the extra groups add about 0.001 to R² by
  chance, which is negligible.
  A feature's score for a family is its highest R² among that family's 3
  variables.
- Label rule: the feature gets the family with the highest R² if
  - R² >= R_min, and
  - it beats the best R² of the other family by at least D.
  Otherwise the label is "mixed".
  R_min = (open), D = (open)
- Mixed features go in no supernode. Their count is reported.
- Visual check: a random sample of each label is inspected by eye.
  Sample size: (open)

---

## 4. Supernodes and suppression

- Supernode: all features at "=" with the same label.
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
  features at "=". Number of groups: (open)
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
- Exact measure of "change": (open)

### 7.3 Descriptive: graph structure (no pass/fail)

- Edge weight between the two supernodes, compared to edge weight within each.
- Reported as evidence from the replacement model, not as a test.

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
| fail    | any       | no evidence of separate pathways |

### 7.6 Exploratory (not part of the decision)

- Sufficiency: keep only one pathway and check whether its digit survives.
  Run only if time allows, and labeled exploratory.

---

## 8. Deviations

(none)
