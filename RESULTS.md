# Results (real run, weight reconstruction)

Model file: `data/smollm2-135m.safetensors`  
Matrices tested: 16 largest 2D weights (41.6M params)  
Metric: mean relative Frobenius error and cosine similarity between original and reconstructed weights. Lower error / higher cosine is better.

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

Task-aware proxy at 0.55 BPW (binary path): uniform rank rel. error 0.862 vs energy-weighted rank 0.856. Note: per-matrix weighting changes each matrix's BPW, so this is a directional signal, not a same-budget proof. Same-budget allocation across the whole model is v0.2.
