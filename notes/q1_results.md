# Q1 results

Question: do Llama-3.2-1B's transcoder features form separate last-digit and
magnitude pathways, like Haiku's?

Answer: partly. The two supernodes affect different digits of the answer, but
they are not independent: the last-digit supernode feeds the magnitude
supernode. Preregistered claim (section 7.5, primary pass, internal fail):
**"the supernodes affect different digits, and interact internally"**.

Code: `scripts/q1_analysis.py` (suppression), `scripts/q1_graphs_analysis.py`
(graphs). Numbers: `results/q1_analysis/`, `results/q1_graphs/`.

## Suppression (7.1, 7.2, 7.4)

998 evaluation prompts per run, 100 layer-matched random groups per real group.
No X outcomes anywhere; the lenient and strict readouts give the same decisions.

| Suppressed | Prompts | Intended digit broken | Random p95 | Other digit broken | Random p95 |
|---|---|---|---|---|---|
| last digit (108 features) | no-carry | U 48.0% [43.6, 52.4] | 0.2% | M 0.0% | 5.6% |
| magnitude (98 features) | all | M 25.9% [23.4, 28.6] | 2.0% | U 0.0% | 3.6% |

- 7.1 primary: pass. Each supernode breaks its own digit far above random
  groups, and never the other digit.
- 7.2 internal: fail by a hair, in one direction only. Last digit off changes
  the magnitude features' summed activation by 12.05% (random p95 11.98%,
  median 4.4%). Magnitude off changes the last-digit features by 3.4% (p95
  5.7%, passes).
- 7.4 carry: with last digit off, both digits break (B) on 26.6% of carry
  prompts and 14.5% of no-carry prompts; difference 95% CI [6.9, 17.3]
  percentage points. Supported, but 14.5% on no-carry prompts is not small.

Robustness variants:
- Last digit without the max/min mod 10 features: U 9.4% (passes, much
  weaker). Most of the effect comes from those features.
- Magnitude without the 53 hand-kept tens lookups: M 0.2% vs p95 1.9%
  (fails). The magnitude effect depends on the manually reviewed features.
- Relaxed labels (R^2 0.5, no manual review): pass. U 47.8%, M 45.8%.

Secondary groups:
- mod 100: almost no effect (N 99.2%). The prediction "mostly B" is not
  supported.
- comparison: M 40.6%. mixed family: M 36.3%. Both break the tens digit.

## Attribution graphs (7.3, descriptive)

200 graphs (100 carry, 100 no-carry), circuit-tracer `attribute`,
batch size 64. About 65 s per graph on a T4.

- Replacement score 0.67, completeness 0.93 (means; CIs within 0.003).
- Influence share per group (means): other positions 37.4%, embeddings 18.5%,
  mostly active 14.6%, error 9.0%, unclear 7.5%, last digit 3.5%, mixed
  family 3.4%, magnitude 2.1%, mod 100 2.0%, comparison 2.0%. The 5 groups
  together: 13%.
- Exact-value unclear features (exploratory split): 0.07% together.

How to read the shares: a node's influence counts every path from it to the
answer, so a path embedding -> operand feature -> "=" feature -> answer
credits all 3 nodes. Earlier nodes collect credit for everything downstream;
the shares are not a split of the answer into parts. "13%" is the groups'
share of influence credit, not the fraction of the answer they compute.

At "=" alone, scored features hold about 35% (groups 13, mostly active 14.6,
unclear 7.5); the groups are about 37% of that, in line with the 43%
activation share in preregistration 7.5. Other positions are the input stage
(features describing each operand), outside the groups by design.

Edges between the two primary groups, per source feature (share of a target
feature's direct input, divided by the source group's size; last digit has
about 35 nodes per graph, magnitude about 18):

| Source -> target | Per source feature |
|---|---|
| last digit -> last digit | 0.0029 |
| last digit -> magnitude | 0.0010 |
| magnitude -> magnitude | 0.0009 |
| magnitude -> last digit | 0.0002 |

Magnitude features take about as much input per feature from last-digit
features as from each other; last-digit features take almost none from
magnitude. The same direction as the 7.2 fail. A likely reason: the tens
digit needs the carry from the units.

Caveat: edges only run from earlier to later layers. Magnitude features sit
later (42 of 98 in L14; last digit is spread over L11-L14, 11 in L15). Layer
order can explain part of the asymmetry, but not all: the 23 magnitude
features in L12-L13 could feed last-digit features in L14-L15 and barely do.

## Path diagram (illustration)

`figures/path_diagram.png` (`scripts/path_diagram.py`, `helixadd/diagram.py`,
`notebooks/05_path_diagram.ipynb`): 36+59=95 (carry) and 36+53=89 (no carry),
each from one pruned attribution graph (nodes with 80% of the influence,
about 2,300). Nodes are grouped into boxes; an arrow is the share of the
target box's positive direct input from the source box.

- Both operands feed last digit and comparison directly (13-29%).
- Comparison is the main input to magnitude (44% in both prompts).
- Magnitude (41%, 33%) and comparison (35%, 37%) supply most of the input to
  the sum mod 100 features; last digit adds 9% and 5%.
- Switched off on these prompts: last digit 95 -> 91, 89 -> 85; magnitude
  95 -> 85, 89 -> 79; comparison 115, 59; mod 100 and mixed family no change.
- The answer's direct input is 61% (65%) unlabeled: for 36+59, mostly active
  37%, unclear 14%, error 10%.
- The carry does not show: last digit -> magnitude appears only in the
  no-carry prompt (6%).
- Sum mod 100 sits before the answer but switching it off changes nothing.

One prompt each: an illustration of the results above, not evidence.

## Limits

- The claim is about the labeled features, not Llama's complete pathways.
- The magnitude result rests on the hand-kept lookups (variant above).
- Influence share is a poor coverage measure (see above). A better one: the
  share of the input-to-answer influence lost when a group's nodes are removed
  from the graph. Not computed.
