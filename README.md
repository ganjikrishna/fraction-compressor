# fraction-compressor

**You use a fraction of your LLM. Keep that fraction. Compress the rest away.**

Most LLM compression (GPTQ, AWQ, GGUF, Samsung's LittleBit) tries to preserve
*everything* the model knows, just in fewer bits. That is the hardest version
of the problem, and it is why quality falls off a cliff below ~4 bits per
weight for general benchmarks, and why sub-1-bit needs quantization-aware
training to survive at all.

This project tests the opposite bet:

> A personal / single-purpose model does not need its full general capacity.
> Profile what you *actually* ask it to do, spend your bit budget there,
> and let the unused capacity go.

## What the Instagram post that started this actually claims

Samsung Research's LittleBit (NeurIPS 2025, arXiv:2506.13771):
low-rank latent factorization + binarized factors + multi-scale
compensation, down to 0.1 bits per weight. Reported: Llama2-13B
26.06 GB (FP16) -> 0.84 GB, Llama2-70B -> 1.98 GB, up to 11.6x inference
speedup potential. Code: SamsungLabs/LittleBit on GitHub.
Caveat in the post itself, and in the paper: extreme compression costs
accuracy, and the published numbers come *with* quantization-aware training.

## What this repo does (v0.1, honest scope)

1. **Baselines anyone can reproduce in minutes, pure numpy, no GPU:**
   int8 and int4 group quantization, low-rank SVD in fp16, and a simplified
   training-free LittleBit-style path (SVD -> binarize factors -> row /
   latent / column scales).
2. **Real measurement, not marketing:** relative reconstruction error and
   cosine similarity on the largest real weight matrices of a real model
   (SmolLM2-135M for v0.1), at matched effective bits-per-weight, where
   "effective" counts every scale we store. See RESULTS.md.
3. **The task-aware test (the actual thesis):** same bit budget, two ways
   to spend it -- uniform rank everywhere vs rank weighted toward the
   matrices / layers your usage profile exercises. v0.1 uses singular-value
   energy concentration as a cheap stand-in for a usage profile.
   v0.2 replaces it with a real profile: activations recorded while the
   model runs *your* prompts.

## What v0.1 already shows

See RESULTS.md for the run on your machine. Expect the training-free binary
path to look bad next to int4. That is a finding, not a failure: it sizes
exactly how much of LittleBit's result is the factorization idea vs the
QAT training on top of it, and it sets the bar the task-aware version has
to clear at the same budget.

## What v0.2 tested (upstream audit + RESULTS2.md)

We audited SamsungLabs/LittleBit directly (cloned Oct 11, 2026):
its license is CC BY-NC 4.0 (noncommercial), so none of its code is used
here; the two methods below are clean-room numpy implementations written
from the published algorithm descriptions. Its training path also needs
PyTorch CUDA + DeepSpeed QAT, which rules out low-end local boxes, and it
ships no official prebuilt checkpoints.

Training-free, same matrices and budgets as v0.1:

| Addition | Result |
|---|---|
| Dual-SVID rank-1 scales alone | No real gain: 0.826 -> 0.826 rel. error at 0.8 BPW |
| + Joint-ITQ rotation (LittleBit-2) | **Real gain: 0.826 -> 0.730 at 0.8 BPW, 0.862 -> 0.794 at 0.55, 0.910 -> 0.872 at 0.3** |
| + Residual path at matched BPW | No gain training-free: 0.727 / 0.800 / 0.883 (same or worse than ITQ alone) |

Takeaway: of LittleBit's three ideas, only the ITQ rotation pays off
without training. Dual-SVID scales and residual compensation appear to
earn their keep only inside QAT, which is consistent with our v0.1
finding that sub-1-bit needs recovery training. Our next lever is
therefore not more initialization tricks; it is task-aware allocation
plus a small recovery step (roadmap v0.4 / v0.5 below, renumbered).

## What v0.3 tested (resident fraction, Colibri's idea clean-room)

Colibri runs giant MoE models in little RAM by placing weights across a
storage/RAM/VRAM hierarchy: the routed (hot) fraction stays resident, the
rest streams from disk. We stole only the placement idea and tested the
honest dense-model version: profile real activations (profile3.py, 8 short
prompts, pure-numpy forward pass), keep the 4 hottest of the 16 matrices
resident as fp16 SVD at 3.0 BPW, compress the cold 12 with the v0.2 binary
ITQ path, all at matched total budget. Plus a measured disk wall: cold blobs
written to disk, page cache evicted, naive memmap on-demand reads timed.

Results (RESULTS3.md): the usage signal is real (hottest matrix sees 6x the
activation of the quietest) but the split does NOT beat uniform v0.2 ITQ on
mean weight error at matched budget (0.787 vs 0.730 at 0.8 BPW; worse at
0.55 and 0.3). Usage-weighted rank allocation also fails to beat uniform.
What the split buys is resident memory: 1.318 MB resident vs 83 MB full
fp16 (63x less), at a fidelity cost of 0.730 -> 0.787. Cold disk reads are
cheap at this scale (3.83 ms) but the reconstruct matmul is not (145 ms),
which is the measured reason Colibri-style streaming collapses to
~0.02-0.1 tok/s at 744B scale. For dense models, hot/cold is a
memory-placement strategy, not a compression strategy. The next honest
test is end-to-end task accuracy, where the hot fraction might matter.

## The Signet test: does the sign of the fine-tuning delta carry the change?

A side experiment with a sharper question. Fine-tuning turns a base model into
an instruct model; the difference is a weight delta. We asked how much of that
delta's effect survives keeping only its sign: 1 bit per weight, applied to the
base model.

Arms: base, base + full delta, base + sign(delta), base + DARE-style 90%
drop-and-rescale, and the real instruct model. Model: Qwen2.5-0.5B. Probes:
24 logprob items across knowledge, instruction-following, and math sets
(signet_results.json).

| Probe set | Base to instruct gap | Closed by sign alone | Closed by DARE-90 |
|---|---|---|---|
| Knowledge | 1.016 nats | 57.8% | Negative (worse than base) |
| Instruction | 6.525 nats | 44.8% | Negative |
| Math | Negative (instruct scored below base) | n/a | n/a |

Takeaway: the direction of the fine-tuning change carries roughly half of its
measurable effect on these probes; which way each weight moved matters more
than how far. Dropping 90% of the delta and rescaling, the standard merge
trick, destroyed the transfer here. Honest limits: 24 items, logprob probes
only, one small model. This is a signal worth a bigger test, not a benchmark.
The SmolLM arms of the same battery were voided by a launcher bug that
selected the instruct file as its own base; the launcher is fixed and the
reruns are pending.

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

## The comparison that matters

| | General compression (LittleBit, GGUF...) | fraction-compressor |
|---|---|---|
| Goal | Keep all capability, fewer bits | Keep *used* capability, drop the rest |
| Bit budget spent | Uniform / saliency-based | By your measured usage profile |
| Expected general benchmark | Best possible at that size | Worse. On purpose. |
| Expected score on your tasks | Degrades with everyone else's | The number we optimize and publish |
| Fits | Any user | One user / one job, perfectly |

If the task-aware version cannot beat uniform compression *on the profiled
tasks at the same size*, the thesis is wrong and the results will say so.
That is the test.

## Run it

```bash
python3 -m venv .venv && .venv/bin/pip install numpy
# any safetensors model works; this one is ~258 MB
curl -L -o data/smollm2-135m.safetensors \
  https://huggingface.co/HuggingFaceTB/SmolLM2-135M/resolve/main/model.safetensors
.venv/bin/python benchmark.py --model data/smollm2-135m.safetensors --max-matrices 16
```

## License

MIT. Inspired by LittleBit (Samsung Research); no LittleBit code is used.
