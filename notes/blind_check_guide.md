# Blind check: how to label

Open `figures/blind_check/page_*.png`. Each panel is one feature at "=":
a on the vertical axis, b on the horizontal axis, white = not active.
Fill in `results/blind_check/sheet.csv` (one row per panel number).
Do not open `results/blind_check/key.csv` until you are done.

## my_role (Anthropic's kinds, by the shape of the plot)
- `add function`: horizontal and/or vertical bars (depends on a, on b, or on each separately)
- `sum`: diagonal lines or bands from top-left to bottom-right (depends on a + b)
- `lookup table`: a grid or blobs (depends on a and b together, not through the sum)
- `comparison`: L-shapes, corners, or the a = b diagonal (depends on min, max, a - b)
- `mostly active`: on almost everywhere (the whole plot is filled)
- `unclear`: you can't tell, or it fits none of these

## my_family (what kind of condition)
- `last digit`: repeats every 10 (units digits)
- `magnitude`: one or a few smooth regions (size)
- `mod 100`: diagonal lines 100 apart, or thin exact-sum lines
- `comparison`: about which operand is larger / their difference
- `unclear` (for a mostly active feature: write what its value seems to track, or unclear)

`note`: optional, anything odd.
