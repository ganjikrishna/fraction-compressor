# Research notes

Scratch log for ideas that are not results yet. Honest by default: if an
idea is untested, say so here.

## 2026-10-20: Gemini design review (Google AI, via Vinay)

External review of this repo, two rounds. Two conclusions change the
v0.4 design enough to log:

1. Compute-in-place beats rebuild at batch 1. For a cold matrix stored
   as binarized low-rank factors at ~0.55 BPW (rank ~563 for a
   2048x2048 matrix), multiplying through the factors reads ~0.29 MB
   per token vs 8.4 MB for a materialized fp16 W: 29x to 58x less
   memory bandwidth, and fewer FLOPs too (2r < d). Rebuilding W only
   pays off for large-batch prefill. So the v0.4 runtime should
   multiply (X U) V^T directly on the factors and never materialize W.
   Our 145 ms rebuild number is the cost of doing it the slow way.
2. Cheapest end-to-end Signet swap test: a system-instruction hot-swap.
   Three generation runs (base; base + full instruct delta; base +
   sign(delta)) on a strict JSON-format system prompt, 10 prompts,
   scored by json.loads parseability rate and format-token alignment
   against the full-delta run. If the sign run breaks, log LayerNorm
   input mean/variance per step to test the activation-drift theory,
   and try per-layer rescaling on attention projections only.

Also from round 1: look-ahead streaming does not help under CPU
saturation (both sides agreed). Signet deltas as a hot-swappable cache
is the product shape to test once the swap test passes.

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
