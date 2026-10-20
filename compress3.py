"""
fraction-compressor v0.3 methods: the resident-fraction split.

Colibri's idea, clean-room and honestly scoped for dense models:
measure which matrices actually see signal (profile3.json), keep the hot
fraction resident at high precision (fp16 SVD), and compress the cold
fraction hard (the v0.2 binary ITQ path, the "streams from disk" analog).

Nothing here is copied from Colibri. Only the placement idea is stolen:
resident = hot at high precision, cold = compressed on disk, measured.

Functions:
  load_profile()              -> dict name -> mean_abs usage
  split_hot_cold(names, prof) -> (hot_names, cold_names), top-4 by usage
  solve_cold_bpw(...)         -> cold per-weight budget after hot takes 3.0 BPW
  serialize_binary_v2(W, target) -> bytes blob of the cold codec
      (sign bits packed + fp16 scales, matching bpw_dualsvid accounting)
  deserialize_binary_v2(blob) -> reconstructed matrix
  usage_weighted_bpw(names, prof, target) -> per-matrix BPW matching budget
  resident_bytes_fp16(m, n, r) -> bytes of stored fp16 factors
"""
import json
import numpy as np
from compress import rank_for_bpw, effective_bpw as eff_bpw_fp16
from compress2 import (_binary_path, _rank1_abs_scales, bpw_dualsvid,
                       _r_for_bpw_dualsvid)

HOT_BPW = 3.0      # hot matrices are always stored at this budget (fp16 SVD)
HOT_COUNT = 4      # hot = top 4 of the 16 matrices by measured usage

def load_profile(path="profile3.json"):
    with open(path) as f:
        d = json.load(f)
    return {k: v["mean_abs"] for k, v in d["profile"].items()}

def split_hot_cold(names, prof):
    ranked = sorted(names, key=lambda k: -prof.get(k, 0.0))
    hot = ranked[:HOT_COUNT]
    cold = ranked[HOT_COUNT:]
    return hot, cold

def solve_cold_bpw(params, hot, cold, target_bpw):
    """Hot takes HOT_BPW per weight; cold gets whatever balances the budget."""
    P = sum(params[n] for n in hot + cold)
    hot_bits = sum(params[n] for n in hot) * HOT_BPW
    cold_params = sum(params[n] for n in cold)
    cold_bpw = (target_bpw * P - hot_bits) / cold_params
    return cold_bpw, P

def hot_rank(m, n):
    from compress import rank_for_bpw
    return rank_for_bpw(m, n, HOT_BPW, binary=False)

def resident_bytes_fp16(m, n, r):
    # svd_fp16 stores A (m x r) and B (n x r) in fp16, nothing else
    return r * (m + n) * 2

def serialize_binary_v2(W, target_bpw):
    """Pack the v0.2 dual-svid + ITQ cold codec into bytes.

    Layout: 256-byte JSON header, then packed sign bits for A and B,
    then fp16 scale vectors a, b, c, d. Clean-room: uses the repo's own
    _binary_path factors, packed here.
    """
    m, n = W.shape
    r = _r_for_bpw_dualsvid(m, n, target_bpw, residual=False)
    from compress import svd_factor
    from compress2 import _itq_rotation
    A, B = svd_factor(W.astype(np.float32), r)
    X = np.concatenate([A, B], axis=0)
    R = _itq_rotation(X, n_iter=50)
    A, B = A @ R, B @ R
    a, b = _rank1_abs_scales(A)   # |A| ~= outer(a, b)
    c, d = _rank1_abs_scales(B)   # |B| ~= outer(c, d)
    sA = (np.sign(A) > 0).astype(np.uint8).ravel()
    sB = (np.sign(B) > 0).astype(np.uint8).ravel()
    pA = np.packbits(sA)
    pB = np.packbits(sB)
    body = (pA.tobytes() + pB.tobytes() + a.astype(np.float16).tobytes()
            + b.astype(np.float16).tobytes() + c.astype(np.float16).tobytes()
            + d.astype(np.float16).tobytes())
    header = json.dumps(dict(m=m, n=n, r=r, len_pA=len(pA), len_pB=len(pB),
                             len_a=len(a), len_b=len(b), len_c=len(c),
                             len_d=len(d))).encode()
    header = header + b"\x00" * (256 - len(header))
    return header + body, bpw_dualsvid(m, n, r, residual=False)

def deserialize_binary_v2(blob):
    header = json.loads(bytes(blob[:256]).rstrip(b"\x00").decode())
    m, n, r = header["m"], header["n"], header["r"]
    off = 256
    pA = np.frombuffer(blob[off:off + header["len_pA"]], dtype=np.uint8)
    off += header["len_pA"]
    pB = np.frombuffer(blob[off:off + header["len_pB"]], dtype=np.uint8)
    off += header["len_pB"]
    def take(k):
        nonlocal off
        v = np.frombuffer(blob[off:off + 2 * k], dtype=np.float16).astype(np.float32)
        off += 2 * k
        return v
    a, b, c, d = take(header["len_a"]), take(header["len_b"]), take(header["len_c"]), take(header["len_d"])
    sA = np.unpackbits(pA)[:m * r].reshape(m, r).astype(np.float32)
    sB = np.unpackbits(pB)[:n * r].reshape(n, r).astype(np.float32)
    sA = np.where(sA > 0, 1.0, -1.0)
    sB = np.where(sB > 0, 1.0, -1.0)
    A_hat = sA * np.outer(a, b)
    B_hat = sB * np.outer(c, d)
    return (A_hat @ B_hat.T).astype(np.float32)

def usage_weighted_bpw(names, params, shapes, prof, target_bpw, strength=1.0):
    """Per-matrix BPW from the measured profile, exactly matching the budget.

    b_i = target * (1 + s * (u_i - 1)), u_i normalized to params-weighted mean 1.
    Sum p_i * b_i == target * P by construction (before flooring).
    """
    P = sum(params[n] for n in names)
    u = np.array([prof.get(n, 0.0) for n in names], dtype=np.float64)
    p = np.array([params[n] for n in names], dtype=np.float64)
    u = u / (p @ u / P + 1e-12)
    b = target_bpw * (1.0 + strength * (u - 1.0))
    out = {}
    for n, bi in zip(names, b):
        m_, n_ = shapes[n]
        min_bpw = bpw_dualsvid(m_, n_, 1, residual=False)
        out[n] = max(float(bi), min_bpw)
    return out
