1. Questions and Hypotheses
Q1: Does Llama-3.2-1B add through parallel path
ways?
Using the public transcoders for Llama-3.2-1B, we ask
whether its addition features form separate pathways like
those of Haiku: a last-digit pathway and at least one mag
nitude pathway, each of which changes a different part of the
answer when it is suppressed.

transcoder->imitates MLP to explain speciific behavior
we replace the MLPs with their transcoders to get the replacement model
on that prompt it shoudl produce the same output.
we can see the effect of one feature to the next one with a fixed number.

this set of number is the attribution graph.
nodes->token embeddings, features, output logit
edges->linear effects

we prune this.

one graph  = one prompt.

In “36+59=”, the model predicts “95” from the residual stream at the last position, the “=”. Whatever produces the answer has to arrive there.

Features on the operand tokens (“36”, “59”) mostly describe that number. Example: “this number ends in 6”.
Attention copies that information over to “=”.
Features on “=” combine both operands. That’s where you find things like “the sum ends in 5” or “the sum is around 90”.



pathways are made from transcoder features.They make an attribution graph. The related features are made into supernodes,and we can check the graph by intervening on them.

we specifically look at the addition features of the operands, looking at their activation plots.

Q2:  Is each pathway fed by its own part of the helix?
If the pathways exist, we ask whether the last-digit pathway
depends on the periodic components of the operand represen
tation (T = 2,5,10 and their harmonics) and the magnitude
pathway on its slow components (the linear term, T = 100
and above), and which specific periods each pathway reads