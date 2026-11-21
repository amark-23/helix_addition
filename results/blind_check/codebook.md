# Feature taxonomy: how to read an operand plot

Every plot is one feature at the "=" token. Vertical axis: a (first operand),
horizontal axis: b (second operand), both 0 to 99. Color: how strongly the
feature fires on the prompt a+b. White: it does not fire.

Examples of every category: `figures/taxonomy/taxonomy.pdf`. None of them
is in a blind check.

## Step 1: the role (what shape is it?)

Ask these in order and stop at the first yes.

1. **Is almost the whole plot filled (more than 90%)?** -> `mostly active`.
   Shape does not matter here (Anthropic's rule: active on most prompts).
   A plot filled on about half, e.g. one side of the a = b diagonal, is not
   mostly active: label its shape.
2. **Diagonal lines or bands from top-left to bottom-right?** -> `sum`.
   Along such a line a + b is constant, so the feature depends on the sum.
3. **Full-length vertical and/or horizontal lines or bars?** -> `add function`.
   A condition on one operand, or a union of conditions on each (Anthropic's
   "add 9": a or b ends in 9, lines every 10 in both directions). At "=" in
   Llama we found none of that kind; the ones we found are "b is a single
   digit" (one thick bar at the left edge).
4. **L-shapes, corners, nested squares, or the a = b diagonal (bottom-left to
   top-right)?** -> `comparison`. These depend on min(a, b), max(a, b) or a - b.
5. **A regular grid of dots or blobs, or a few separate blobs?** -> `lookup table`.
   It depends on a and b together, not through their sum.
6. **None of these, or several mixed, or a shape only in part of the plot?** -> `unclear`.

## Step 2: the family (what condition?)

- **last digit**: the pattern repeats every 10 (lines every 10, a grid with
  spacing 10, nested squares every 10).
- **magnitude**: one or a few smooth regions (a band, a corner, blobs, bars at
  the edge).
- **mod 100**: diagonal lines exactly 100 apart (e.g. sum 55 and sum 155), or a
  single thin diagonal line.
- **comparison**: L-shapes, corners, the a = b diagonal, regions above or
  below that diagonal.
- **unclear**: none of these, or for a mostly active feature when it is not possible to
  tell what its value tracks.

## Confusions to watch for

- **L-shape vs blobs.** An L that bends at the a = b diagonal is `comparison`
  (min or max). Blobs that sit anywhere are `lookup table`.
- **Sum lines every 10 vs 100 apart.** Count the gap between lines: 10 ->
  `last digit`, 100 -> `mod 100`.
- **Full lines vs nested squares.** Lines every 10 that run the full length ->
  `add function`. Lines that turn a corner at the a = b diagonal ->
  `comparison` / `last digit` (max or min ends in d).
- **Grid vs nested squares.** A grid has dots in rows and columns. Nested
  squares are lines that turn a corner at the a = b diagonal.
- **Mostly active with a shape.** If almost all of it is filled, it is
  `mostly active` even if a diagonal is visible inside it. About half filled:
  label the shape.
- Lookup tables were the hardest category to find clean examples for (11
  candidates were removed from the examples as not textbook). Be strict there.

## Procedure for a blind check

1. Read this note and `figures/taxonomy/taxonomy.pdf`.
2. Prepare a round: `python scripts/checks/blind_check.py NAME`.
3. Label `figures/blind_check_NAME/page_*.png` in
   `results/blind_check_NAME/sheet.csv`, without opening `key.csv`.
4. Compare: `python scripts/checks/blind_compare.py results/blind_check_NAME`.
5. When unsure: `unclear`, with the reason in `note`.

Rounds done so far: results/blind_check (75 features) and
results/blind_check_2 (30 features); see preregistration section 3.5.
