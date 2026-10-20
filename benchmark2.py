"""
v0.2 benchmark: does LittleBit's Dual-SVID scale design, LittleBit-2's
Joint-ITQ rotation, and the residual path actually help a TRAINING-FREE
compressor? Same model, same matrices, same effective BPW targets as v0.1.

Usage: .venv/bin/python benchmark2.py --model data/smollm2-135m.safetensors --max-matrices 16
"""
import argparse, json, time
import numpy as np
from benchmark import load_safetensors
from compress import svd_binary, rel_error, cosine
from compress2 import svd_binary_v2

TARGETS = [0.8, 0.55, 0.3]
VARIANTS = [
    ("v0.1 svd_binary (mean scales)", lambda W, t: svd_binary(W, t)),
    ("v0.2 dual-svid scales", lambda W, t: svd_binary_v2(W, t, use_itq=False)),
    ("v0.2 dual-svid + joint-itq", lambda W, t: svd_binary_v2(W, t, use_itq=True)),
    ("v0.2 dual-svid + itq + residual", lambda W, t: svd_binary_v2(W, t, use_itq=True, residual=True)),
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-matrices", type=int, default=16)
    args = ap.parse_args()

    sd = load_safetensors(args.model)
    mats = [(k, v) for k, v in sd.items() if v.ndim == 2 and v.size >= 200_000]
    mats.sort(key=lambda kv: -kv[1].size)
    mats = mats[: args.max_matrices]
    print(f"{len(mats)} matrices in scope")

    rows = []
    for target in TARGETS:
        for name, fn in VARIANTS:
            errs, coss, bpws = [], [], []
            t0 = time.time()
            for _, W in mats:
                rec, bpw, *rest = fn(W.astype(np.float32), target)
                errs.append(rel_error(W, rec)); coss.append(cosine(W, rec)); bpws.append(bpw)
            row = dict(method=name, target_bpw=target, eff_bpw=float(np.mean(bpws)),
                       rel_err=float(np.mean(errs)), cosine=float(np.mean(coss)),
                       seconds=round(time.time() - t0, 1))
            rows.append(row); print(row)

    with open("results2.json", "w") as f:
        json.dump(dict(model=args.model, rows=rows), f, indent=2)
    with open("RESULTS2.md", "w") as f:
        f.write("# Results v0.2 (real run, weight reconstruction)\n\n")
        f.write(f"Model file: `{args.model}`  \nSame {len(mats)} largest matrices and BPW targets as RESULTS.md (v0.1). "
                "Training free throughout: no QAT, no fine tuning.\n\n")
        f.write("| Method | Target BPW | Effective BPW | Rel. error | Cosine |\n|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['method']} | {r['target_bpw']} | {r['eff_bpw']:.2f} | {r['rel_err']:.3f} | {r['cosine']:.3f} |\n")
    print("wrote results2.json and RESULTS2.md")

if __name__ == "__main__":
    main()
