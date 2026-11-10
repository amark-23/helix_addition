# Feature taxonomy: how to read an operand plot

Every plot is one feature at the "=" token. Vertical axis: a (first operand),
horizontal axis: b (second operand), both 0 to 99. Color: how strongly the
feature fires on the prompt a+b. White: it does not fire.

Examples of every category: `figures/taxonomy/page_1.png` and `page_2.png`.
None of them is in the blind set.

## Step 1: the role (what shape is it?)

Ask these in order and stop at the first yes.

1. **Is a large part of the plot filled (more than half)?** -> `mostly active`.
   Shape does not matter here (Anthropic's rule: active on most prompts).
2. **Diagonal lines or bands from top-left to bottom-right?** -> `sum`.
   Along such a line a + b is constant, so the feature depends on the sum.
3. **Only vertical bars, or only horizontal bars?** -> `add function`.
   It depends on one operand alone. (In Llama these are mostly "b is a single digit".)
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
- **unclear**: none of these, or for a mostly active feature when you cannot
  tell what its value tracks.

## Confusions to watch for

- **L-shape vs blobs.** An L that bends at the a = b diagonal is `comparison`
  (min or max). Blobs that sit anywhere are `lookup table`.
- **Sum lines every 10 vs 100 apart.** Count the gap between lines: 10 ->
  `last digit`, 100 -> `mod 100`.
- **Grid vs nested squares.** A grid has dots in rows and columns. Nested
  squares are lines that turn a corner at the a = b diagonal.
- **Mostly active with a shape.** If more than half is filled, it is
  `mostly active` even if you see a diagonal inside it.
- Lookup tables were the hardest category to find clean examples for (11
  candidates were removed from the examples as not textbook). Be strict there.

## Procedure

1. Read this note and look at both example pages.
2. Practice: label `figures/taxonomy/practice.png` (letters A to T) in
   `results/practice/sheet.csv`, then run `python scripts/blind_compare.py practice`.
   Look at your disagreements. Here learning is allowed.
3. Blind round: label `figures/blind_check/page_1.png` to `page_4.png` in
   `results/blind_check/sheet.csv`. Do not open `results/blind_check/key.csv`.
   Then run `python scripts/blind_compare.py`.
4. When unsure: `unclear`, and write why in `note`.
