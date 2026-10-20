"""
fraction-compressor v0.3: usage profiler.

Records per-matrix input activation magnitudes by running a small set of
real prompts through SmolLM2-135M in pure numpy (no torch needed).
The profile is the "routing heat" input to the hot/cold resident-fraction
split: matrices that actually see signal stay resident at high precision,
the rest stream from the cold store.

Usage metric per 2D weight W (applied as y = x @ W.T):
    mean over prompts of mean over tokens of |x|

Saves profile3.json: {tensor_name: {mean_abs, shape, params}}.

Honest limits stated up front: 8 short prompts is a toy workload, not a
claim about Vinay's real task distribution. The metric is input magnitude,
not a causal sensitivity. It is a measured signal, not marketing.
"""
import json, time
import numpy as np
from tokenizers import Tokenizer
from benchmark import load_safetensors

# SmolLM2-135M config (HuggingFaceTB/SmolLM2-135M config.json)
VOCAB = 49152
HIDDEN = 576
LAYERS = 30
HEADS = 9
KV_HEADS = 3
HEAD_DIM = 64
INTER = 1536
EPS = 1e-5
ROPE_THETA = 100000.0

PROMPTS = [
    "Write a Python function that reverses a string in place.",
    "What is the capital of France? Answer in one sentence.",
    "def fibonacci(n):\n    # return the nth Fibonacci number",
    "Explain photosynthesis in one simple sentence.",
    "2 + 2 =",
    "Translate to French: good morning, how are you?",
    "Name three prime numbers less than 20.",
    "Write a short haiku about the ocean at night.",
]

def rmsnorm(x, w):
    return x / np.sqrt((x ** 2).mean(-1, keepdims=True) + EPS) * w

def rope_freqs(T):
    inv = 1.0 / (ROPE_THETA ** (np.arange(0, HEAD_DIM, 2, dtype=np.float32) / HEAD_DIM))
    f = np.outer(np.arange(T, dtype=np.float32), inv)          # (T, 32)
    e = np.concatenate([f, f], axis=-1)                        # (T, 64)
    return np.cos(e).astype(np.float32), np.sin(e).astype(np.float32)

def apply_rope(x, cos, sin):
    x1, x2 = x[..., :HEAD_DIM // 2], x[..., HEAD_DIM // 2:]
    xr = np.concatenate([-x2, x1], axis=-1)
    return x * cos[:, None, :] + xr * sin[:, None, :]

def softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)

def silu(x):
    return x / (1.0 + np.exp(-x))

def run_prompt(ids, sd, usage):
    """One forward pass. usage[name] accumulates (token_sum, token_count)."""
    T = len(ids)
    cos, sin = rope_freqs(T)
    x = sd["model.embed_tokens.weight"][np.array(ids, dtype=np.int64)]  # (T, 576)
    usage["model.embed_tokens.weight"][0] += float(np.abs(x).sum())
    usage["model.embed_tokens.weight"][1] += x.size
    for i in range(LAYERS):
        p = f"model.layers.{i}"
        h = rmsnorm(x, sd[f"{p}.input_layernorm.weight"])
        for nm in ("q_proj", "k_proj", "v_proj", "o_proj"):
            key = f"{p}.self_attn.{nm}.weight"
            usage[key][0] += float(np.abs(h).sum())
            usage[key][1] += h.size
        q = (h @ sd[f"{p}.self_attn.q_proj.weight"].T).reshape(T, HEADS, HEAD_DIM)
        k = (h @ sd[f"{p}.self_attn.k_proj.weight"].T).reshape(T, KV_HEADS, HEAD_DIM)
        v = (h @ sd[f"{p}.self_attn.v_proj.weight"].T).reshape(T, KV_HEADS, HEAD_DIM)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        k = np.repeat(k, HEADS // KV_HEADS, axis=1)
        v = np.repeat(v, HEADS // KV_HEADS, axis=1)
        att = np.einsum("thd,shd->hts", q, k) / np.sqrt(HEAD_DIM)
        mask = np.triu(np.ones((T, T), dtype=bool), 1)
        att[:, mask] = -1e9
        att = softmax(att)
        o = np.einsum("hts,shd->thd", att, v).reshape(T, HIDDEN)
        x = x + o @ sd[f"{p}.self_attn.o_proj.weight"].T
        h2 = rmsnorm(x, sd[f"{p}.post_attention_layernorm.weight"])
        for nm in ("gate_proj", "up_proj", "down_proj"):
            key = f"{p}.mlp.{nm}.weight"
            xin = h2 if nm != "down_proj" else None
            if xin is not None:
                usage[key][0] += float(np.abs(xin).sum())
                usage[key][1] += xin.size
        gate = silu(h2 @ sd[f"{p}.mlp.gate_proj.weight"].T)
        up = h2 @ sd[f"{p}.mlp.up_proj.weight"].T
        mid = gate * up
        usage[f"{p}.mlp.down_proj.weight"][0] += float(np.abs(mid).sum())
        usage[f"{p}.mlp.down_proj.weight"][1] += mid.size
        x = x + mid @ sd[f"{p}.mlp.down_proj.weight"].T
    # tied lm head: same matrix as embed_tokens; record its input usage too
    usage["lm_head (tied to embed_tokens)"][0] += float(np.abs(x).sum())
    usage["lm_head (tied to embed_tokens)"][1] += x.size
    return x

def main():
    t0 = time.time()
    print("loading weights...")
    sd = load_safetensors("data/smollm2-135m.safetensors")
    usage = {k: [0.0, 0] for k in sd if sd[k].ndim == 2}
    usage["lm_head (tied to embed_tokens)"] = [0.0, 0]
    tok = Tokenizer.from_file("data/tokenizer.json")
    for j, pr in enumerate(PROMPTS):
        ids = tok.encode(pr).ids[:64]
        run_prompt(ids, sd, usage)
        print(f"prompt {j+1}/{len(PROMPTS)} ({len(ids)} tokens)")
    prof = {}
    for k, (s, n) in usage.items():
        w = sd[k] if k in sd else sd["model.embed_tokens.weight"]
        prof[k] = dict(mean_abs=float(s / max(n, 1)),
                       shape=list(w.shape), params=int(w.size))
    with open("profile3.json", "w") as f:
        json.dump(dict(prompts=PROMPTS, profile=prof), f, indent=2)
    # quick look at spread among the 16 matrices the benchmarks use
    mats = [(k, v) for k, v in sd.items() if v.ndim == 2 and v.size >= 200_000]
    mats.sort(key=lambda kv: -kv[1].size)
    mats = mats[:16]
    vals = [prof[k]["mean_abs"] for k, _ in mats]
    print(f"16-mat usage mean_abs: min {min(vals):.4f} max {max(vals):.4f} "
          f"median {float(np.median(vals)):.4f}")
    for k, _ in mats:
        print(f"  {k}: {prof[k]['mean_abs']:.4f}")
    print(f"wrote profile3.json in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
