# Results v0.2 (real run, weight reconstruction)

Same 16 largest SmolLM2-135M matrices and BPW targets as RESULTS.md (v0.1). Training free throughout: no QAT, no fine tuning.

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

Residual rows are the corrected rerun at matched effective BPW (the first residual run had a rank bug that under-spent the budget; fixed in compress2.py).
