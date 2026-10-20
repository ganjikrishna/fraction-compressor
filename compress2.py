"""
fraction-compressor v0.2 methods.

Clean-room numpy implementations of two published ideas, written from the
algorithm descriptions in the papers (no SamsungLabs code used; their repo
is CC BY-NC 4.0, ours is MIT):

1. Dual-SVID style scales (LittleBit, arXiv:2506.13771):
   instead of a single mean-abs scale per row, approximate |factor| by a
   rank-1 outer product from its SVD: |F| ~= outer(a, b). The stored
   scale cost is (rows + cols) fp16 numbers per factor.
   Reconstruction: F_hat = sign(F) * outer(a, b).

2. Joint-ITQ rotation (LittleBit-2, arXiv:2603.00042; ITQ itself is
   Gong & Lazebnik, CVPR 2011):
   before binarizing, rotate BOTH latent factors by one shared orthogonal
   matrix R, chosen iteratively to push the rotated values toward the
   binary hypercube corners (where sign() loses least):
       repeat: B = sign(X @ R);  R = argmin orthogonal fit via SVD of B^T X
   Because R is orthogonal, A @ B^T == (A R) @ (B R)^T exactly before
   binarization, and R is folded into the factors, so nothing extra is
   stored and inference cost is unchanged.

3. Residual path: factorize the leftover W - W_hat a second time and add
   it back, splitting the rank budget so total BPW stays matched.
"""
import numpy as np
from compress import rank_for_bpw, svd_factor, rel_error

def _rank1_abs_scales(F):
    """Rank-1 SVD of |F| -> (row_vec, col_vec) with outer(row, col) ~= |F|."""
    U, S, Vt = np.linalg.svd(np.abs(F).astype(np.float32), full_matrices=False)
    s = np.sqrt(S[0])
    return (U[:, 0] * s).astype(np.float32), (Vt[0, :] * s).astype(np.float32)

def _binarize_with_dualsvid(F):
    a, b = _rank1_abs_scales(F)
    return np.sign(F) * np.outer(a, b), a, b

def bpw_dualsvid(m, n, r, residual=False):
    # signs: r*(m+n) bits. scales: A needs (m+r), B needs (n+r) fp16.
    paths = 2 if residual else 1
    bits = paths * (r * (m + n) + (m + n + 2 * r) * 16)
    return bits / (m * n)

def _r_for_bpw_dualsvid(m, n, target, residual=False):
    paths = 2 if residual else 1
    # bits(r) = paths * (r*(m+n) + (m+n+2r)*16); solve roughly then adjust
    r = int(target * m * n / (paths * ((m + n) + (m + n) * 0 + 32 + (m + n))))
    r = max(1, r)
    while bpw_dualsvid(m, n, r, residual) > target and r > 1:
        r -= 1
    while bpw_dualsvid(m, n, r + 1, residual) <= target:
        r += 1
    # NOTE: bpw_dualsvid(..., residual=True) already counts BOTH paths,
    # so the r it solves for is the per-path rank, not a total to split.
    return min(r, min(m, n))

def _itq_rotation(X, n_iter=50, seed=0):
    """Standard ITQ: orthogonal R minimizing ||sign(X R) - X R||_F."""
    rng = np.random.default_rng(seed)
    r = X.shape[1]
    # init R from QR of random gaussian (orthogonal)
    q, _ = np.linalg.qr(rng.standard_normal((r, r)).astype(np.float32))
    R = q
    X = X.astype(np.float32)
    for _ in range(n_iter):
        Z = X @ R
        B = np.sign(Z)
        B[B == 0] = 1
        M = B.T @ X
        U, _, Vt = np.linalg.svd(M, full_matrices=False)
        R = (Vt.T @ U.T).astype(np.float32)
    return R

def svd_binary_v2(W, target_bpw, use_itq=False, residual=False):
    m, n = W.shape
    r_total = _r_for_bpw_dualsvid(m, n, target_bpw, residual=residual)
    if residual:
        r_each = max(1, r_total)  # per-path rank (see _r_for_bpw_dualsvid note)
        rec = np.zeros_like(W, dtype=np.float32)
        leftover = W.astype(np.float32)
        for _ in range(2):
            part, _, _ = _binary_path(leftover, r_each, use_itq)
            rec = rec + part
            leftover = W.astype(np.float32) - rec
        return rec, bpw_dualsvid(m, n, r_each, residual=True), r_each * 2
    rec, bpw, r = _binary_path(W.astype(np.float32), r_total, use_itq)
    return rec, bpw_dualsvid(m, n, r_total, residual=False), r_total

def _binary_path(W, r, use_itq):
    A, B = svd_factor(W, r)  # W ~= A @ B.T, A: m x r, B: n x r
    if use_itq:
        X = np.concatenate([A, B], axis=0)  # joint: one R for both factors
        R = _itq_rotation(X, n_iter=50)
        A = A @ R
        B = B @ R
    A_hat, _, _ = _binarize_with_dualsvid(A)
    B_hat, _, _ = _binarize_with_dualsvid(B)
    return (A_hat @ B_hat.T).astype(np.float32), None, r
