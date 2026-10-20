# fraction-compressor

**The problem:** Most LLM compression (GPTQ, AWQ, GGUF, Samsung's
LittleBit) tries to keep everything the model knows, just in fewer bits.
That is the hardest version of the problem. Quality drops fast below
about 4 bits per weight, and sub-1-bit compression only survives with
quantization-aware training.

**What this project does:** It tests the opposite idea. A personal,
single-purpose model does not need its full general capacity. So this
project profiles what you actually ask the model to do, spends the bit
budget there, and lets the unused capacity go. All code is pure numpy,
training-free, and runs on a small CPU machine with no GPU.

## Background

This started from an Instagram post about Samsung Research's LittleBit
(NeurIPS 2025, arXiv:2506.13771). LittleBit uses low-rank latent
factorization, binarized factors, and multi-scale compensation to reach
0.1 bits per weight. Reported results: Llama2-13B from 26.06 GB (FP16)
to 0.84 GB, Llama2-70B to 1.98 GB, up to 11.6x inference speedup
potential. The paper is clear about the cost: extreme compression loses
accuracy, and the published numbers use quantization-aware training.

We audited the SamsungLabs/LittleBit code directly (cloned Oct 11,
2026). Its license is CC BY-NC 4.0 (noncommercial), so no LittleBit
code is used here. The methods below are clean-room implementations
written from the published papers. See NOTICE.md for full attribution.

## How it works

- **v0.1 baselines:** int8 and int4 group quantization, low-rank SVD in
  fp16, and a simple training-free binary path (SVD, binarize the
  factors, store row / latent / column scales).
- **v0.2 methods:** Dual-SVID rank-1 scales, a Joint-ITQ rotation, and
  a residual path, all clean-room from the LittleBit and LittleBit-2
  papers.
- **v0.3 resident fraction:** a usage profiler (profile3.py) records
  real activations while the model runs your prompts. The hottest
  matrices stay resident at high precision. The cold rest is compressed
  hard and streams from disk.
- **Measurement:** relative reconstruction error and cosine similarity
  on the largest real weight matrices of SmolLM2-135M, at matched
  effective bits-per-weight. "Effective" counts every scale we store.

## Results v0.1

16 largest SmolLM2-135M matrices (41.6M params). Lower error / higher
cosine is better. Full detail in RESULTS.md.

| Method | Target BPW | Effective BPW | Rel. error | Cosine |
|---|---|---|---|---|
| FP16 (original) | 16 | 16.00 | 0.000 | 1.000 |
| int8 | - | 8.12 | 0.006 | 1.000 |
| int4 | - | 4.12 | 0.106 | 0.994 |
| svd_fp16 | 0.8 | 0.77 | 0.919 | 0.374 |
| svd_binary | 0.8 | 0.80 | 0.826 | 0.562 |
| svd_fp16 | 0.55 | 0.53 | 0.934 | 0.332 |
| svd_binary | 0.55 | 0.55 | 0.862 | 0.506 |
| svd_fp16 | 0.3 | 0.27 | 0.953 | 0.266 |
| svd_binary | 0.3 | 0.30 | 0.910 | 0.413 |

Task-aware proxy at 0.55 BPW (binary path): uniform rank rel. error
0.862 vs energy-weighted rank 0.856. A directional signal, not a
same-budget proof.

**Takeaway:** the training-free binary path looks bad next to int4.
That is a finding, not a failure. It shows how much of LittleBit's
result is the factorization idea, and how much is the training on top
of it.

## Results v0.2

Same matrices and budgets as v0.1. Training-free throughout. Full
detail in RESULTS2.md.

| Method | Target BPW | Effective BPW | Rel. error | Cosine |
|---|---|---|---|---|
| v0.1 svd_binary (mean scales) | 0.8 | 0.80 | 0.826 | 0.562 |
| v0.2 dual-svid scales | 0.8 | 0.80 | 0.826 | 0.563 |
| v0.2 dual-svid + joint-itq | 0.8 | 0.80 | 0.730 | 0.684 |
| v0.2 dual-svid + itq + residual | 0.8 | 0.80 | 0.727 | 0.685 |
| v0.1 svd_binary (mean scales) | 0.55 | 0.55 | 0.862 | 0.506 |
| v0.2 dual-svid scales | 0.55 | 0.55 | 0.860 | 0.508 |
| v0.2 dual-svid + joint-itq | 0.55 | 0.55 | 0.794 | 0.608 |
| v0.2 dual-svid + itq + residual | 0.55 | 0.55 | 0.800 | 0.595 |
| v0.1 svd_binary (mean scales) | 0.3 | 0.30 | 0.910 | 0.413 |
| v0.2 dual-svid scales | 0.3 | 0.30 | 0.908 | 0.417 |
| v0.2 dual-svid + joint-itq | 0.3 | 0.30 | 0.872 | 0.486 |
| v0.2 dual-svid + itq + residual | 0.3 | 0.30 | 0.883 | 0.460 |

**Takeaway:** of LittleBit's three ideas, only the ITQ rotation pays
off without training. Dual-SVID scales alone give no gain (0.826 to
0.826 at 0.8 BPW). The residual path adds nothing at matched budget.
Sub-1-bit needs recovery training; that is what v0.5 tests.

## Results v0.3

Same matrices and budgets. The profiler found a real usage signal: the
hottest matrix (layer 11 down_proj) sees 6x the input activation of the
quietest one. Hot set: top 4 of 16 matrices, kept resident as fp16 SVD
at 3.0 BPW. Cold set: the other 12, compressed with the v0.2 binary ITQ
path. Full detail in RESULTS3.md.

| Method | Target BPW | Effective BPW | Rel. error | Cosine | Stored MB | Resident MB |
|---|---|---|---|---|---|---|
| v0.3-a uniform v0.1 binary @ 0.8 | 0.8 | 0.80 | 0.826 | 0.562 | 4.15 | - |
| v0.3-b uniform v0.2 itq @ 0.8 | 0.8 | 0.80 | 0.730 | 0.684 | 4.16 | - |
| v0.3-c usage-weighted binary @ 0.8 | 0.8 | 0.77 | 0.743 | 0.665 | 4.00 | - |
| v0.3-d hot/cold split @ 0.8 | 0.8 | 0.80 | 0.787 | 0.616 | 4.14 | 1.318 |
| v0.3-a uniform v0.1 binary @ 0.55 | 0.55 | 0.55 | 0.862 | 0.506 | 2.86 | - |
| v0.3-b uniform v0.2 itq @ 0.55 | 0.55 | 0.55 | 0.794 | 0.608 | 2.85 | - |
| v0.3-c usage-weighted binary @ 0.55 | 0.55 | 0.54 | 0.799 | 0.594 | 2.80 | - |
| v0.3-d hot/cold split @ 0.55 | 0.55 | 0.55 | 0.849 | 0.524 | 2.85 | 1.318 |
| v0.3-a uniform v0.1 binary @ 0.3 | 0.3 | 0.30 | 0.910 | 0.413 | 1.55 | - |
| v0.3-b uniform v0.2 itq @ 0.3 | 0.3 | 0.30 | 0.872 | 0.486 | 1.55 | - |
| v0.3-c usage-weighted binary @ 0.3 | 0.3 | 0.30 | 0.870 | 0.479 | 1.56 | - |
| v0.3-d hot/cold split @ 0.3 | 0.3 | 0.30 | 0.929 | 0.289 | 1.54 | 1.318 |

Measured disk wall: reading a cold matrix from disk takes 3.83 ms, but
rebuilding it takes about 145 ms. That rebuild cost is the measured
reason this streaming approach drops to about 0.02 to 0.1 tokens per
second at 744B scale.

**Takeaway:** the hot/cold split does not beat uniform v0.2 on weight
error at any budget (0.787 vs 0.730 at 0.8 BPW). What it buys is
resident memory: 1.318 MB resident instead of 83 MB for the full fp16
model (63x less), at a small fidelity cost. For dense models, hot/cold
placement is a memory strategy, not a compression strategy. The next
test is end-to-end task accuracy (v0.4), where the hot fraction might
matter.

## Signet test

**The question:** fine-tuning turns a base model into an instruct
model. The difference is a weight delta. How much of that delta's
effect survives if we keep only its sign (1 bit per weight) and apply
it to the base model?

**Setup:** five arms (base, base + full delta, base + sign of delta,
base + DARE-style 90% drop-and-rescale, real instruct model) on
Qwen2.5-0.5B. Scoring uses 24 hand-written logprob probe items across
knowledge, instruction-following, and math sets (signet_results.json).

| Probe set | Base to instruct gap | Closed by sign alone | Closed by DARE-90 |
|---|---|---|---|
| Knowledge | 1.016 nats | 57.8% | Negative (worse than base) |
| Instruction | 6.525 nats | 44.8% | Negative |
| Math | Negative (instruct scored below base) | n/a | n/a |

**Takeaway:** the direction of the fine-tuning change carries roughly
half of its measurable effect on these probes. Which way each weight
moved matters more than how far. The DARE-90 arm destroyed the transfer
on this test. Limits: 24 items, logprob probes only, one small model.
A signal worth a bigger test, not a benchmark.

Note on the SmolLM arms: the first battery runs on SmolLM-135M and
SmolLM2-360M were void, because a launcher bug selected the instruct
file as its own base. The launcher is fixed. Corrected reruns on
SmolLM-135M and SmolLM2-360M are pending, and their results will be
added here.

## How this differs from general compression

| | General compression (LittleBit, GGUF...) | fraction-compressor |
|---|---|---|
| Goal | Keep all capability, fewer bits | Keep used capability, drop the rest |
| Bit budget spent | Uniform / saliency-based | By your measured usage profile |
| Expected general benchmark | Best possible at that size | Worse. On purpose. |
| Expected score on your tasks | Degrades with everyone else's | The number we optimize and publish |
| Fits | Any user | One user / one job, perfectly |

If the task-aware version cannot beat uniform compression on the
profiled tasks at the same size, the idea is wrong and the results
will say so. That is the test.

## Roadmap

- [x] v0.1 Weight-level compressor + baselines + real results table
- [x] v0.2 Clean-room Dual-SVID + Joint-ITQ + residual, benchmarked (RESULTS2.md)
- [x] v0.3 Resident-fraction split: usage profiler (profile3.py, profile3.json),
      hot/cold placement at matched budget, measured disk wall (RESULTS3.md).
      Honest result: real usage signal, but the split loses to uniform on mean
      weight error; it buys 63x less resident memory, not fidelity
- [ ] Signet at scale: rerun the fixed SmolLM arms, extend the sign test to
      larger probe sets and a second model family (hackathon build focus,
      Oct 26 to Nov 4, 2026)
- [ ] v0.4 End-to-end: export a compressed model that actually runs
      (llama.cpp / Ollama on a constrained box, e.g. 11 GB RAM, iGPU),
      report tokens/sec, RAM, perplexity, and task accuracy
      (general benchmark vs your-task benchmark, side by side)
- [ ] v0.5 QAT-lite recovery: short fine-tune of only the scale factors
      on your-task data, measure how much accuracy it buys back per bit

## Run it

```bash
python3 -m venv .venv && .venv/bin/pip install numpy
# any safetensors model works; this one is ~258 MB
curl -L -o data/smollm2-135m.safetensors \
  https://huggingface.co/HuggingFaceTB/SmolLM2-135M/resolve/main/model.safetensors
.venv/bin/python benchmark.py --model data/smollm2-135m.safetensors --max-matrices 16
```

## License

MIT. See NOTICE.md for attribution, model licenses, and the clean-room
statement. Inspired by LittleBit (Samsung Research); no LittleBit code
is used.
