# Research notes

Scratch log for ideas that are not results yet. Honest by default: if an
idea is untested, say so here.

## 2026-10-17: Kolmogorov compression (ff-tracking)

Vinay's connection, from an Instagram reel: the ff-tracking demo stores the
*program that makes the video* instead of the pixels. Kolmogorov compression
in practice. The reel itself is a single 6-second MIT-licensed art piece
(a virtual camera filming a terminal with an AI agent typing), not a tool;
no install, no use for the pipeline.

The LLM analog: *weight generation* instead of weight storage. A compact
generator plus task-specific parameters, rather than the full weight
matrices. That is the LoRA / hypernetworks neighborhood: the generator is
shared and cheap, the code it takes as input is small, and the weights it
emits are the large expensive thing you never ship.

Why it belongs in this repo: the task-aware thesis says spend the bit budget
where the usage profile is. Weight generation is the extreme form of that
bet: instead of deciding how to spend bits on existing weights, learn the
mapping from a small code to the weights, and spend everything on the code.
Signet (sign-only weight deltas) sits in the same neighborhood: keep the
*structure that matters* (direction of every hedged correction), drop the
rest.

Testable, not yet tested: train a tiny generator (e.g. one MLP per layer)
that emits the layer weights from a compact latent, compare against the
same-bit uniform and task-weighted baselines from the battery. Expected to
be hard without training; the honest version of this idea probably needs
QAT on the generator. Not in the battery until someone runs it.
