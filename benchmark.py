"""
Run fraction-compressor on real model weights and produce RESULTS.md

Usage:
  python3 benchmark.py --model data/smollm2-135m.safetensors --max-matrices 24

Honest scope: this measures WEIGHT reconstruction (relative error, cosine)
at a given effective bits-per-weight. It does not measure perplexity or
task accuracy -- that needs a full inference runtime (next step, llama.cpp /
transformers on Flash). Weight error is the cheapest real signal and the
one the LittleBit paper's method has to beat first.
"""
import argparse, json, struct, time
import numpy as np
from compress import quantize_uniform, svd_fp16, svd_binary, rel_error, cosine

def load_safetensors(path):
    """Minimal safetensors reader (numpy only). Handles F32/F16/BF16/I64 etc.
    BF16 -> FP32 is a 16-bit left shift, so no torch is needed."""
    with open(path, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hlen))
        base = 8 + hlen
        out = {}
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            start, end = meta["data_offsets"]
            f.seek(base + start)
            raw = f.read(end - start)
            dt = meta["dtype"]
            if dt == "BF16":
                u = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
                arr = u.view(np.float32)
            elif dt == "F16":
                arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
            elif dt == "F32":
                arr = np.frombuffer(raw, dtype=np.float32)
            elif dt == "F64":
                arr = np.frombuffer(raw, dtype=np.float64).astype(np.float32)
            else:
                continue
            out[name] = arr.reshape(meta["shape"]).copy()
        return out

TARGETS = [0.8, 0.55, 0.3]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-matrices", type=int, default=24)
    args = ap.parse_args()

    sd = load_safetensors(args.model)
    mats = [(k, v) for k, v in sd.items() if v.ndim == 2 and v.size >= 200_000]
    mats.sort(key=lambda kv: -kv[1].size)
    mats = mats[: args.max_matrices]
    total_params = sum(v.size for _, v in mats)
    print(f"{len(mats)} matrices, {total_params/1e6:.1f}M params in scope")

    rows = []
    # baselines that do not depend on target bpw
    for name, bits in [("int8", 8), ("int4", 4)]:
        errs, coss, bpws = [], [], []
        for _, W in mats:
            rec, bpw = quantize_uniform(W.astype(np.float32), bits)
            errs.append(rel_error(W, rec)); coss.append(cosine(W, rec)); bpws.append(bpw)
        rows.append(dict(method=name, target_bpw=None, eff_bpw=float(np.mean(bpws)),
                         rel_err=float(np.mean(errs)), cosine=float(np.mean(coss))))
        print(rows[-1])

    for target in TARGETS:
        for name, fn in [("svd_fp16", svd_fp16), ("svd_binary", svd_binary)]:
            errs, coss, bpws = [], [], []
            t0 = time.time()
            for _, W in mats:
                rec, bpw, r = fn(W.astype(np.float32), target)
                errs.append(rel_error(W, rec)); coss.append(cosine(W, rec)); bpws.append(bpw)
            rows.append(dict(method=name, target_bpw=target, eff_bpw=float(np.mean(bpws)),
                             rel_err=float(np.mean(errs)), cosine=float(np.mean(coss)),
                             seconds=round(time.time() - t0, 1)))
            print(rows[-1])

    # Task-aware test: same total budget, non-uniform rank.
    # Uniform = same r everywhere. Energy-weighted = rank proportional to
    # how concentrated a matrix's singular values are (a cheap proxy for
    # "this matrix carries more of what the model actually uses").
    # We test it on the binary path at 0.55 BPW.
    from compress import svd_factor, effective_bpw
    target = 0.55
    uni_err, wtd_err = [], []
    for _, W in mats:
        W = W.astype(np.float32); m, n = W.shape
        rec_u, _, _ = svd_binary(W, target)
        uni_err.append(rel_error(W, rec_u))
        # energy concentration: participation ratio of singular values
        s = np.linalg.svd(W, compute_uv=False)
        energy = (s ** 2) / (s ** 2).sum()
        conc = float(1.0 / (energy ** 2).sum() / len(s))  # 0..1, high = concentrated
        adj = target * (0.6 + 0.8 * conc)  # spend more where concentrated, less elsewhere
        rec_w, _, _ = svd_binary(W, adj)
        wtd_err.append(rel_error(W, rec_w))
    task_aware = dict(target_bpw=target,
                      uniform_rel_err=float(np.mean(uni_err)),
                      energy_weighted_rel_err=float(np.mean(wtd_err)))
    print("task-aware proxy:", task_aware)

    out = dict(model=args.model, matrices=len(mats), params_in_scope=int(total_params),
               rows=rows, task_aware_proxy=task_aware)
    with open("results.json", "w") as f:
        json.dump(out, f, indent=2)

    with open("RESULTS.md", "w") as f:
        f.write("# Results (real run, weight reconstruction)\n\n")
        f.write(f"Model file: `{args.model}`  \nMatrices tested: {len(mats)} largest 2D weights "
                f"({total_params/1e6:.1f}M params)  \nMetric: mean relative Frobenius error and cosine "
                "similarity between original and reconstructed weights. Lower error / higher cosine is better.\n\n")
        f.write("| Method | Target BPW | Effective BPW | Rel. error | Cosine |\n|---|---|---|---|---|\n")
        f.write("| FP16 (original) | 16 | 16.00 | 0.000 | 1.000 |\n")
        for r in rows:
            f.write(f"| {r['method']} | {r['target_bpw'] or '-'} | {r['eff_bpw']:.2f} | "
                    f"{r['rel_err']:.3f} | {r['cosine']:.3f} |\n")
        f.write(f"\nTask-aware proxy at {target} BPW (binary path): uniform rank rel. error "
                f"{task_aware['uniform_rel_err']:.3f} vs energy-weighted rank "
                f"{task_aware['energy_weighted_rel_err']:.3f}. "
                "Note: per-matrix weighting changes each matrix's BPW, so this is a "
                "directional signal, not a same-budget proof. Same-budget allocation "
                "across the whole model is v0.2.\n")
    print("wrote results.json and RESULTS.md")

if __name__ == "__main__":
    main()
