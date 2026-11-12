# Exploratory leads

Found after the Q1 freeze. None of this is part of a preregistered test.

## Features the label library cannot describe

The library has no exact-value variables, so these get R^2 near 0 and the
label "unclear". `scripts/exact_kinds.py` finds them by where each unclear
feature fires most (`results/exact_kinds/`, `figures/unclear_kinds.png`):
855 exact operand value, 116 exact sum, 20 exact difference. In the 200
graphs they carry 0.07% of the influence together.

- Exact operand value, as a union of a horizontal and a vertical line
  (Anthropic's add-function form with exact values): "a or b is 10"
  (L12 #81936), 0 (L8 #32661, L11 #64992), 1 (L7 #103572), 2 (L12 #57761),
  4 (L11 #76127). L7 #13477: b = 78, b = 86, a = 63, a = 52. The claim "no
  Anthropic-style add function at '='" holds only for periodic ones ("add 9").
- Exact sum, layer 15: "the answer is 100" (L15 #57869, #75421), also 90,
  105, 84. Single diagonal lines, partly lit.
- a = b exactly (layers 6-10, "doubling"), some only for large numbers
  (L9 #109796, #439).

Idea: ranges of width 1 as variables. Causal check of L11 #64992 ("+0").

## For Q2

- Period-2 (parity) features at "=": L14 #76111 and #103718 "sum is odd",
  L14 #3767 and L12 #112333 "a and b both even", L14 #24399 "both odd".
  Relevant for T = 2.
- "max(a, b) about X" L-shapes (L12 #11335 and similar): the magnitude
  counterpart of "max ends in d". Both sit where Haiku has add-function features.
- Max/min mod 10 features (nested squares) as Llama's version of "add d":
  the same lines cut off at a = b. They carry most of the last-digit effect.

## Graphs

- Path-based coverage: remove a group's nodes, measure the lost
  input-to-answer influence. Run on about 20 prompts.
- Edge asymmetry vs layer order: compare edges only between features in the
  same layer ranges.
