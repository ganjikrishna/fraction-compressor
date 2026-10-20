# Results v0.3: the resident-fraction split (real run)

Model file: `data/smollm2-135m.safetensors`  
Same 16 largest matrices as v0.1/v0.2 (41.6M params). Training free throughout.

Usage profile: mean absolute input activation per matrix over 8 short prompts (code, chat, math, translation), pure-numpy forward pass, profile3.json. Hot = top 4 of 16 matrices by usage (8.5% of params in scope):

- `model.layers.11.mlp.down_proj.weight` usage 0.3141
- `model.layers.10.mlp.down_proj.weight` usage 0.1108
- `model.layers.11.mlp.gate_proj.weight` usage 0.1005
- `model.layers.11.mlp.up_proj.weight` usage 0.1005

Hot codec: fp16 SVD at 3.0 BPW (resident). Cold codec: v0.2 dual-svid + ITQ binary at the solved remaining budget (the on-disk analog). Effective BPW is total stored bits / total params, exactly matched across methods.

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

## The disk wall (measured, not theory)

Cold blobs for the 0.55-target split (12 matrices, 1.53 MB) written to disk, page cache evicted with posix_fadvise(DONTNEED), then read back through numpy memmap:

- mean cold read per matrix: **3.83 ms** (33.3 MB/s effective)
- mean warm read per matrix: 0.064 ms (page-cached, for contrast)
- mean reconstruct from blob: 145.4 ms (unpack + matmul, CPU)
- blob round-trip check rel. error: 0.000381 (sanity, not a result)

Verdict on the Colibri analog: the placement idea works and the cold read is cheap at this model scale, but the wall is real and proportional: every cold matrix costs a disk read plus a full reconstruct matmul before it can be used. On a 744B model with 370 GB of cold weights that same proportional cost is why independent measurements land at ~0.02-0.1 tokens/sec.

## Honest verdict

The usage profile is real: the hottest matrix (layer 11 down_proj) sees 6x the
input activation of the quietest one. But on the repo's standard metric (mean
weight error at matched total budget), the stolen idea did NOT earn its place
as a fidelity win:

- Uniform v0.2 ITQ (b) beats the hot/cold split (d) at every budget:
  0.730 vs 0.787 at 0.8, 0.794 vs 0.849 at 0.55, 0.872 vs 0.929 at 0.3.
- Usage-weighted allocation (c) also fails to beat uniform: 0.743 vs 0.730 at
  0.8 (note: (c) under-spent its budget by ~4% at 0.8 due to rank discretization,
  so the gap is partly budget, not just allocation), tie at 0.3 (0.870 vs 0.872).
- Why it loses: the fp16 hot fraction is only 8.5% of params. Making those 8.5%
  near-perfect cannot compensate for starving the other 91.5%. Mean error is
  dominated by the cold majority, and the cold majority does best with a uniform
  budget. At 0.3 BPW the hot fraction eats 85% of the budget and cold sits at
  the rank-1 floor.

What the split DOES buy: resident memory. (d) keeps 1.318 MB resident vs 83 MB
for the full fp16 model (63x less) and vs 4.1 MB if the whole uniform (b)
compressed model were kept resident (3.2x less), at a fidelity cost of
0.730 -> 0.787 at 0.8 BPW. That is the honest version of Colibri's claim:
placement buys resident-memory reduction, not fidelity at fixed budget.

Consequence for the thesis: for dense models, "keep the hot fraction" is a
memory-placement strategy, not a compression strategy. The next honest test is
not finer splits of the same weight-error metric but the end-to-end metric the
project has always promised: perplexity / task accuracy on the profiled tasks,
where the hot fraction's fidelity might actually matter. That is the renumbered
v0.4.
