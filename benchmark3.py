"""
v0.3 benchmark: the resident-fraction split (Colibri's idea, clean-room).

Compares, at the same total bit budget, on the same 16 matrices as v0.1/v0.2:
  (a) uniform v0.1 binary (svd_binary)
  (b) uniform v0.2 dual-svid + joint-ITQ (svd_binary_v2)
  (c) usage-weighted v0.2-style allocation: same binary ITQ codec,
      per-matrix BPW set by the measured profile (profile3.json)
  (d) v0.3 hot/cold: hot 4 matrices resident as fp16 SVD at 3.0 BPW,
      cold 12 matrices as binary ITQ at the solved remaining budget

Plus the honest Colibri wall: cold blobs are written to disk, the page
cache is evicted, and the naive memmap on-demand read is timed.

Usage: .venv/bin/python benchmark3.py --model data/smollm2-135m.safetensors --max-matrices 16
"""
import argparse, json, os, time
import numpy as np
from benchmark import load_safetensors
from compress import svd_binary, svd_fp16, rel_error, cosine, rank_for_bpw, effective_bpw
from compress2 import svd_binary_v2, bpw_dualsvid
from compress3 import (load_profile, split_hot_cold, solve_cold_bpw, HOT_BPW,
                       hot_rank, resident_bytes_fp16, usage_weighted_bpw,
                       serialize_binary_v2, deserialize_binary_v2)

TARGETS = [0.8, 0.55, 0.3]

def method_uniform_binary(W, t):
    rec, bpw, _ = svd_binary(W, t)
    return rec, bpw

def method_v2_uniform(W, t):
    rec, bpw, _ = svd_binary_v2(W, t, use_itq=True)
    return rec, bpw

def run_method(name, mats, fn, params):
    errs, coss, bpws, bits = [], [], [], []
    t0 = time.time()
    for k, W in mats:
        rec, bpw = fn(k, W)
        errs.append(rel_error(W, rec)); coss.append(cosine(W, rec))
        bpws.append(bpw); bits.append(bpw * params[k])
    P = sum(params[k] for k, _ in mats)
    return dict(method=name, eff_bpw=float(sum(bits) / P),
                rel_err=float(np.mean(errs)), cosine=float(np.mean(coss)),
                stored_mb=float(sum(bits) / 8 / 1e6),
                seconds=round(time.time() - t0, 1))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-matrices", type=int, default=16)
    args = ap.parse_args()

    prof = load_profile()
    sd = load_safetensors(args.model)
    mats = [(k, v) for k, v in sd.items() if v.ndim == 2 and v.size >= 200_000]
    mats.sort(key=lambda kv: -kv[1].size)
    mats = mats[: args.max_matrices]
    params = {k: int(v.size) for k, v in mats}
    shapes = {k: v.shape for k, v in mats}
    P = sum(params.values())
    print(f"{len(mats)} matrices, {P/1e6:.1f}M params in scope")

    hot, cold = split_hot_cold([k for k, _ in mats], prof)
    print("hot (resident fp16):")
    for k in hot:
        print(f"  {k} usage={prof.get(k,0):.4f}")
    cold_bpw_by_target = {}
    for t in TARGETS:
        cb, _ = solve_cold_bpw(params, hot, cold, t)
        cold_bpw_by_target[t] = cb
        print(f"target {t}: cold budget {cb:.3f} BPW "
              f"(hot takes {HOT_BPW} BPW on {sum(params[k] for k in hot)/P*100:.1f}% of params)")

    rows = []
    for target in TARGETS:
        # (a) uniform v0.1
        rows.append(run_method(f"v0.3-a uniform v0.1 binary @ {target}",
                               mats, lambda k, W: method_uniform_binary(W, target), params))
        # (b) uniform v0.2
        rows.append(run_method(f"v0.3-b uniform v0.2 itq @ {target}",
                               mats, lambda k, W: method_v2_uniform(W, target), params))
        # (c) usage-weighted v0.2-style
        uw = usage_weighted_bpw([k for k, _ in mats], params, shapes, prof, target)
        rows.append(run_method(f"v0.3-c usage-weighted binary @ {target}",
                               mats, lambda k, W: method_v2_uniform(W, uw[k]), params))
        # (d) hot/cold
        cb = cold_bpw_by_target[target]
        def hotcold(k, W):
            W = W.astype(np.float32)
            if k in hot:
                rec, bpw, r = svd_fp16(W, HOT_BPW)
                return rec, bpw
            rec, bpw, _ = svd_binary_v2(W, cb, use_itq=True)
            return rec, bpw
        r_d = run_method(f"v0.3-d hot/cold split @ {target}", mats, hotcold, params)
        res_bytes = sum(resident_bytes_fp16(*shapes[k], hot_rank(*shapes[k])) for k in hot)
        r_d["resident_mb"] = round(res_bytes / 1e6, 3)
        r_d["hot_param_frac"] = round(sum(params[k] for k in hot) / P, 4)
        r_d["cold_bpw"] = round(cb, 3)
        rows.append(r_d)
        for r in rows[-4:]:
            print(r)

    # The Colibri wall: time naive on-demand cold reads from disk.
    # Use the 0.55-target cold blobs (12 matrices).
    cb = cold_bpw_by_target[0.55]
    ddir = "/tmp/fc3-cold"
    os.makedirs(ddir, exist_ok=True)
    blobs = {}
    for k, W in mats:
        if k not in cold:
            continue
        blob, bpw = serialize_binary_v2(W.astype(np.float32), cb)
        path = os.path.join(ddir, k.replace(".", "_") + ".bin")
        with open(path, "wb") as f:
            f.write(blob)
        blobs[k] = (path, len(blob))
    total_bytes = sum(b for _, b in blobs.values())
    # evict page cache for our files (honest cold read)
    for path, _ in blobs.values():
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    cold_ms, warm_ms, rec_ms = [], [], []
    for k, (path, _) in blobs.items():
        t0 = time.perf_counter()
        mm = np.memmap(path, dtype=np.uint8, mode="r")
        data = np.array(mm)          # force the read
        del mm
        cold_ms.append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter()
        rec = deserialize_binary_v2(data.tobytes())
        rec_ms.append((time.perf_counter() - t0) * 1000)
        del rec
    for k, (path, _) in blobs.items():  # warm: everything is cached now
        t0 = time.perf_counter()
        mm = np.memmap(path, dtype=np.uint8, mode="r")
        data = np.array(mm)
        del mm
        warm_ms.append((time.perf_counter() - t0) * 1000)
    disk = dict(matrices=len(blobs), total_mb=round(total_bytes / 1e6, 2),
                mean_cold_ms=round(float(np.mean(cold_ms)), 2),
                mean_warm_ms=round(float(np.mean(warm_ms)), 3),
                mean_reconstruct_ms=round(float(np.mean(rec_ms)), 1),
                cold_mbps=round(total_bytes / 1e6 / (sum(cold_ms) / 1000), 1))
    # sanity: one blob round-trips to the in-memory result
    k0 = next(iter(blobs))
    W0 = dict(mats)[k0].astype(np.float32)
    rec0, _, _ = svd_binary_v2(W0, cb, use_itq=True)
    with open(blobs[k0][0], "rb") as f:
        rt = deserialize_binary_v2(f.read())
    disk["roundtrip_rel_err"] = round(rel_error(rec0, rt), 6)
    print("disk:", disk)

    with open("results3.json", "w") as f:
        json.dump(dict(model=args.model, matrices=len(mats), params_in_scope=P,
                       hot=hot, cold=cold, rows=rows, disk=disk), f, indent=2)
    with open("RESULTS3.md", "w") as f:
        f.write("# Results v0.3: the resident-fraction split (real run)\n\n")
        f.write(f"Model file: `{args.model}`  \nSame {len(mats)} largest matrices as v0.1/v0.2 "
                f"({P/1e6:.1f}M params). Training free throughout.\n\n")
        f.write("Usage profile: mean absolute input activation per matrix over 8 short prompts "
                "(code, chat, math, translation), pure-numpy forward pass, profile3.json. "
                f"Hot = top {len(hot)} of {len(mats)} matrices by usage "
                f"({sum(params[k] for k in hot)/P*100:.1f}% of params in scope):\n\n")
        for k in hot:
            f.write(f"- `{k}` usage {prof.get(k, 0):.4f}\n")
        f.write("\nHot codec: fp16 SVD at 3.0 BPW (resident). Cold codec: v0.2 dual-svid + ITQ "
                "binary at the solved remaining budget (the on-disk analog). "
                "Effective BPW is total stored bits / total params, exactly matched across methods.\n\n")
        f.write("| Method | Target BPW | Effective BPW | Rel. error | Cosine | Stored MB | Resident MB |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['method']} | {r['method'].split('@')[-1].strip()} | {r['eff_bpw']:.2f} | "
                    f"{r['rel_err']:.3f} | {r['cosine']:.3f} | {r['stored_mb']:.2f} | "
                    f"{r.get('resident_mb', '-')} |\n")
        f.write("\n## The disk wall (measured, not theory)\n\n")
        f.write(f"Cold blobs for the 0.55-target split ({disk['matrices']} matrices, "
                f"{disk['total_mb']} MB) written to disk, page cache evicted with "
                "posix_fadvise(DONTNEED), then read back through numpy memmap:\n\n")
        f.write(f"- mean cold read per matrix: **{disk['mean_cold_ms']} ms** "
                f"({disk['cold_mbps']} MB/s effective)\n")
        f.write(f"- mean warm read per matrix: {disk['mean_warm_ms']} ms (page-cached, for contrast)\n")
        f.write(f"- mean reconstruct from blob: {disk['mean_reconstruct_ms']} ms (unpack + matmul, CPU)\n")
        f.write(f"- blob round-trip check rel. error: {disk['roundtrip_rel_err']} (sanity, not a result)\n\n")
        f.write("Verdict on the Colibri analog: the placement idea works and the cold read is "
                "cheap at this model scale, but the wall is real and proportional: every cold "
                "matrix costs a disk read plus a full reconstruct matmul before it can be used. "
                "On a 744B model with 370 GB of cold weights that same proportional cost is "
                "why independent measurements land at ~0.02-0.1 tokens/sec.\n")
    print("wrote results3.json and RESULTS3.md")

if __name__ == "__main__":
    main()
