# Manual review: magnitude lookups (tens (a, b))

Why: in the blind checks, lookup tables labeled through tens (a, b) agreed
with manual inspection on 11 of 24. They no longer count automatically; each
one is kept only if you confirm it.

This review is not blind (you see the feature IDs). It is curation, done
before any suppression has run, so it cannot be influenced by Q1's results.

## For each plot (figures/review/page_*.png)

Write `keep` or `drop` in the `decision` column of
`results/manual_review/tens_lookups.csv` (one row per review number).

**keep** only if ALL of these hold:
- One or a few separate, compact blobs.
- Each blob sits at a specific a AND a specific b ("a about X and b about Y").
- No L-shape or corner bending at the a = b diagonal (that is comparison).
- No diagonal band from top-left to bottom-right (that is sum).
- No full-length bar along one axis (that is add function).
- Not most of the plot filled.

**drop** in every other case, including when you are unsure.

`note` (optional): what it looks like instead, e.g. "comparison: a = b for
large numbers", "sum band", "blob plus L-shape". Notes are not used for
labels; they show what the dropped features are.

When done, tell Claude: the final labels are then built from the automatic
labels plus your decisions.
