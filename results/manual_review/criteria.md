# Manual review: magnitude lookups (tens (a, b))

Why: in the blind checks, lookup tables labeled through tens (a, b) agreed
with manual inspection on 11 of 24. They no longer count automatically; each
one is kept only if it is confirmed by hand.

This review is not blind (the feature IDs are shown). It is curation, done
before any suppression has run, so it cannot be influenced by Q1's results.

## For each plot

Pages: `python scripts/checks/review_sheet.py` draws `figures/review/page_*.png`.

Write `keep` or `drop` in the `decision` column of
`results/manual_review/tens_lookups.csv` (one row per review number).

**keep** only if ALL of these hold:
- One or a few separate, compact blobs.
- Each blob sits at a specific a AND a specific b ("a about X and b about Y").
- No L-shape or corner bending at the a = b diagonal (that is comparison).
- No diagonal band from top-left to bottom-right (that is sum).
- No full-length bar along one axis (that is add function).
- Not most of the plot filled.

**drop** in every other case, including when unsure.

`note` (optional): what it looks like instead, e.g. "comparison: a = b for
large numbers", "sum band", "blob plus L-shape". Notes are not used for
labels; they show what the dropped features are.

When done, `scripts/final_labels.py` builds the final labels from the
automatic labels plus these decisions.


---

# Manual review 2: add function features

Why: after the mostly-active bar moved to 90%, 19 features that fire on
50-90% of prompts were labeled add function (mostly through
"tens separate"). None of the add function features was blind-checked after
that change, and the label examples showed dense textures among them. All 41
are reviewed by hand.

Pages: `python scripts/checks/review_sheet.py add` draws
`figures/review_add/page_*.png`. Decisions: `results/manual_review/add_functions.csv`.

**keep** only if ALL of these hold:
- The pattern is made of vertical and/or horizontal bars or lines: a
  condition on b (vertical), on a (horizontal), or a union of the two.
- The bars run along most of the plot (they do not stop at the a = b diagonal).
- No a = b diagonal structure, no triangle on one side of it (comparison).
- No band from top-left to bottom-right (sum).
- No blobs at specific (a, b) locations (lookup table).
- Not a dense texture without clear bars.

**drop** in every other case, including when unsure.
