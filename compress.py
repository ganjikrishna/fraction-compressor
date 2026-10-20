"""
fraction-compressor v0.1
LittleBit-style weight compression, no training (PTQ) + baselines.
Pure numpy. Works matrix by matrix so it runs on 8GB RAM.

Methods:
  int8 / int4   : uniform group quantization (group=128), fp16 scales
  svd_fp16      : low-rank SVD, factors kept in fp16
  svd_binary    : low-rank SVD, factors binarized to sign(),
                  with row / latent / column scales (fp16)
                  -- a simplified, training-free take on LittleBit's
                     latent factorization + multi-scale compensation.
                     It is NOT LittleBit QAT. Expect it to be worse.
                     That gap is the point: we measure it honestly.

Effective bits-per-weight (BPW) counts everything stored:
binary signs + all fp16 scales, divided by the original weight count.
"""
import numpy as np

def bpw_int(bits, group=128):
    # 1 scale (fp16) per group
    return bits + 16 / group

def quantize_uniform(W, bits, group=128):
    flat = W.reshape(-1)
    pad = (-len(flat)) % group
    if pad:
        flat = np.pad(flat, (0, pad))
    g = flat.reshape(-1, group).astype(np.float32)
    levels = 2 ** bits - 1
    mn = g.min(axis=1, keepdims=True)
    mx = g.max(axis=1, keepdims=True)
    scale = (mx - mn) / levels + 1e-12
    q = np.round((g - mn) / scale)
    rec = (q * scale + mn).reshape(-1)[: W.size].reshape(W.shape)
    return rec.astype(np.float32), bpw_int(bits, group)

def rank_for_bpw(m, n, target_bpw, binary=True):
    # solve for r: bits ~= r*(m+n) [binary] or r*(m+n)*16 [fp16], plus scales
    per_r = (m + n) * (1 if binary else 16) + 16  # + latent scale per rank
    fixed = (m + n) * 16  # row + col scales (binary path only)
    budget = target_bpw * m * n - (fixed if binary else 0)
    r = int(max(1, budget // max(per_r, 1)))
    return min(r, min(m, n))

def effective_bpw(m, n, r, binary=True):
    if binary:
        bits = r * (m + n) * 1 + (m + n + r) * 16
    else:
        bits = r * (m + n) * 16
    return bits / (m * n)

def svd_factor(W, r):
    U, S, Vt = np.linalg.svd(W.astype(np.float32), full_matrices=False)
    sqrt_s = np.sqrt(S[:r])
    A = U[:, :r] * sqrt_s          # m x r
    B = Vt[:r, :].T * sqrt_s       # n x r
    return A, B

def svd_fp16(W, target_bpw):
    m, n = W.shape
    r = rank_for_bpw(m, n, target_bpw, binary=False)
    A, B = svd_factor(W, r)
    return (A @ B.T).astype(np.float32), effective_bpw(m, n, r, binary=False), r

def svd_binary(W, target_bpw):
    m, n = W.shape
    r = rank_for_bpw(m, n, target_bpw, binary=True)
    A, B = svd_factor(W, r)
    # multi-scale: row scale from A, column scale from B, latent scale = mean abs
    row_scale = np.mean(np.abs(A), axis=1, keepdims=True) + 1e-12   # m x 1
    col_scale = np.mean(np.abs(B), axis=1, keepdims=True) + 1e-12   # n x 1
    A_norm = A / row_scale
    B_norm = B / col_scale
    latent_scale = np.mean(np.abs(A_norm), axis=0) * np.mean(np.abs(B_norm), axis=0) + 1e-12  # r
    A_hat = np.sign(A_norm) * (row_scale * latent_scale)
    # fold latent into A only once: B uses col scale only
    B_hat = np.sign(B_norm) * col_scale
    # correct double counting: latent was derived from both, applied once above
    rec = (A_hat @ B_hat.T).astype(np.float32)
    return rec, effective_bpw(m, n, r, binary=True), r

def rel_error(W, rec):
    return float(np.linalg.norm(W - rec) / (np.linalg.norm(W) + 1e-12))

def cosine(W, rec):
    a, b = W.ravel().astype(np.float64), rec.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
