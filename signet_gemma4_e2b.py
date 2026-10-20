"""
Signet battery for Gemma 4 E2B (5.1B dense, 2.3B effective) on Beast (32GB).

Adapts signet_battery.py's 5-arm protocol to Gemma 4's architecture:
  - dense transformer, 35 layers, hidden 1536, 8 q-heads / 1 kv-head, head_dim 256
  - alternating sliding (window 512) / full attention: [s,s,s,s,f] repeating
  - QK RMSNorm (q_norm, k_norm) + unscaled v_norm; attention scaling = 1.0
  - RoPE: sliding theta=10000 standard; full "proportional", partial_rotary_factor=0.25, theta=1e6
  - per-layer embeddings (PLE): token-identity + context-aware, gated residual per layer
  - KV sharing: layers 15-34 reuse K/V from layer 13 (sliding) / 14 (full)
  - GELU-tanh MLP, tied 262144-vocab embeddings scaled by sqrt(1536)
  - final logit softcapping: tanh(logits/30)*30

Memory discipline (Beast has ~28GB usable with Ollama stopped):
  - text-tower only (~4.0B params, ~16GB fp32); audio/vision towers skipped
  - safetensors read via per-tensor streaming (index + seek), never two full
    models in RAM at once; graft arms built by streaming base+instruct tensors
  - KV-shared layers' dead k/v proj weights excluded (reference ignores them)

Usage (on Beast):
  python signet_gemma4_e2b.py --base D:\\gemma4-e2b\\base\\model.safetensors \
      --instruct D:\\gemma4-e2b\\it\\model.safetensors --tok D:\\gemma4-e2b\\it\\tokenizer.json
  Add --sanity-only for a quick forward correctness check.

Writes gemma4_e2b_signet_results.json
Reference: transformers modeling_gemma4.py (Apache 2.0), gemma-4-E2B-it config.
"""
import argparse
import gc
import json
import struct
import sys
import time

import numpy as np

SEED = 7
L = 35
HIDDEN = 1536
HEADS = 8
KV_HEADS = 1
HEAD_DIM = 256
INTER = 6144
VOCAB = 262144
EPS = 1e-6
PLE_DIM = 256
PLE_LAYERS = 35
FIRST_KV_SHARED = 15  # layers >= 15 reuse KV
STORE_SLIDING = 13    # last non-shared sliding layer
STORE_FULL = 14       # last non-shared full layer
SOFT_CAP = 30.0

LAYER_TYPES = []
for i in range(L):
    LAYER_TYPES.append("full_attention" if (i + 1) % 5 == 0 else "sliding_attention")

# RoPE head dims per layer type (must match the q_proj head_dim)
ROPE_HD = {"sliding_attention": 256, "full_attention": 512}

PREFIX = "model.language_model."
SKIP = {f"{PREFIX}layers.{i}.self_attn.{n}.weight"
        for i in range(FIRST_KV_SHARED, L) for n in ("k_proj", "v_proj", "k_norm")}


def format_chat(prompt):
    # Google Gemma 4 Canonical Chat Template (chat_template.jinja), simple case:
    # bos + "<|turn>user\n" + prompt + "<turn|>\n" + "<|turn>model\n"
    # (no system block, no tools, thinking off; assistant role renders as "model")
    return "<|turn>user\n" + prompt + "<turn|>\n<|turn>model\n"


def encode_prompt(tok, prompt):
    bos = tok.token_to_id("<bos>")
    if bos is None:
        bos = 2  # config bos_token_id
    return [bos] + tok.encode(format_chat(prompt)).ids


# ---------- safetensors streaming ----------
def load_index(path):
    with open(path, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hlen))
    base = 8 + hlen
    idx = {}
    for k, v in header.items():
        if k == "__metadata__":
            continue
        if not k.startswith(PREFIX):
            continue
        if k in SKIP:
            continue
        s, e = v["data_offsets"]
        idx[k] = (v["dtype"], tuple(v["shape"]), base + s, e - s)
    return idx


def load_tensor(path, idx, k):
    dt, shape, off, nbytes = idx[k]
    with open(path, "rb") as f:
        f.seek(off)
        raw = f.read(nbytes)
    if dt == "BF16":
        u = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
        arr = ((u << 16).view(np.float32)).reshape(shape)
    elif dt in ("F32", "FP32"):
        arr = np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()
    else:
        raise ValueError(f"unexpected dtype {dt} for {k}")
    return arr


def load_full(path, idx, keys):
    sd = {}
    for k in keys:
        sd[k] = load_tensor(path, idx, k)
    return sd


# ---------- numpy ops ----------
def rmsnorm(x, w, eps=EPS):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w


def rmsnorm_noscale(x, eps=EPS):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps)


def gelu_tanh(x):
    c = np.sqrt(2.0 / np.pi)
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x ** 3)))


def softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


def rope_freqs(T, head_dim, theta, partial_rotary_factor=1.0):
    # sliding: theta=10000 standard; full: "proportional", factor=0.25, theta=1e6
    if partial_rotary_factor >= 1.0:
        inv = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    else:
        ra = int(partial_rotary_factor * head_dim // 2)
        inv_r = 1.0 / (theta ** (np.arange(0, 2 * ra, 2, dtype=np.float32) / head_dim))
        inv = np.concatenate([inv_r, np.zeros(head_dim // 2 - ra, dtype=np.float32)])
    f = np.outer(np.arange(T, dtype=np.float32), inv)
    e = np.concatenate([f, f], axis=-1)
    return np.cos(e), np.sin(e)


def apply_rope(x, cos, sin, head_dim):
    # x: (T, H, head_dim), cos/sin: (T, head_dim); NeoX style
    x1, x2 = x[..., : head_dim // 2], x[..., head_dim // 2 :]
    xr = np.concatenate([-x2, x1], axis=-1)
    return x * cos[:, None, :] + xr * sin[:, None, :]


class Gemma4E2B:
    def __init__(self, sd):
        self.sd = sd

    def forward(self, ids):
        sd = self.sd
        T = len(ids)
        # head_dim and MLP width derived per layer from the weights themselves:
        # sliding layers use hd=256, full-attention layers hd=512;
        # KV-shared layers (15+) use double-width MLP (12288 vs 6144)
        cos_s, sin_s = rope_freqs(T, ROPE_HD["sliding_attention"], 10000.0)
        cos_f, sin_f = rope_freqs(T, ROPE_HD["full_attention"], 1000000.0,
                                  partial_rotary_factor=0.25)

        emb = sd[PREFIX + "embed_tokens.weight"]
        x = emb[np.array(ids, dtype=np.int64)] * np.float32(np.sqrt(HIDDEN))

        # per-layer embeddings: token identity + context aware
        ple_tok = sd[PREFIX + "embed_tokens_per_layer.weight"][np.array(ids, dtype=np.int64)]
        ple_tok = (ple_tok.reshape(T, PLE_LAYERS, PLE_DIM) * np.float32(16.0))
        ctx = (x @ sd[PREFIX + "per_layer_model_projection.weight"].T) * np.float32(HIDDEN ** -0.5)
        ctx = ctx.reshape(T, PLE_LAYERS, PLE_DIM)
        ctx = rmsnorm(ctx, sd[PREFIX + "per_layer_projection_norm.weight"])
        ple = (ctx + ple_tok) * np.float32(2.0 ** -0.5)
        del ctx, ple_tok

        # causal mask pieces (shared across layers)
        tri = np.triu(np.ones((T, T), dtype=bool), 1)
        tt = np.arange(T)[:, None]
        ss = np.arange(T)[None, :]
        win_mask = ss < tt - 511  # sliding window 512

        shared = {}
        for i in range(L):
            p = f"{PREFIX}layers.{i}"
            ltype = LAYER_TYPES[i]
            hd = sd[f"{p}.self_attn.q_proj.weight"].shape[0] // HEADS
            cos, sin = (cos_s, sin_s) if ltype == "sliding_attention" else (cos_f, sin_f)

            # attention
            h = rmsnorm(x, sd[f"{p}.input_layernorm.weight"])
            q = (h @ sd[f"{p}.self_attn.q_proj.weight"].T).reshape(T, HEADS, hd)
            q = rmsnorm(q, sd[f"{p}.self_attn.q_norm.weight"])
            q = apply_rope(q, cos, sin, hd)
            if i < FIRST_KV_SHARED:
                k = (h @ sd[f"{p}.self_attn.k_proj.weight"].T).reshape(T, KV_HEADS, hd)
                v = (h @ sd[f"{p}.self_attn.v_proj.weight"].T).reshape(T, KV_HEADS, hd)
                k = rmsnorm(k, sd[f"{p}.self_attn.k_norm.weight"])
                k = apply_rope(k, cos, sin, hd)
                v = rmsnorm_noscale(v)
                kt, vt = k.transpose(1, 0, 2), v.transpose(1, 0, 2)
                if i == STORE_SLIDING:
                    shared["sliding_attention"] = (kt, vt)
                elif i == STORE_FULL:
                    shared["full_attention"] = (kt, vt)
                del k, v
            else:
                kt, vt = shared[ltype]
            qt = q.transpose(1, 0, 2)
            del q
            scores = qt @ kt.transpose(0, 2, 1)  # scaling = 1.0 (QK norm)
            del qt
            mask = tri if ltype == "full_attention" else (tri | win_mask)
            scores[:, mask] = np.float32(-1e9)
            attn = softmax(scores)
            del scores
            o = (attn @ vt).transpose(1, 0, 2).reshape(T, HEADS * hd)
            del attn
            o = o @ sd[f"{p}.self_attn.o_proj.weight"].T
            o = rmsnorm(o, sd[f"{p}.post_attention_layernorm.weight"])
            x = x + o
            del h, o

            # MLP (gelu-tanh SwiGLU style; width from weights)
            h = rmsnorm(x, sd[f"{p}.pre_feedforward_layernorm.weight"])
            g = h @ sd[f"{p}.mlp.gate_proj.weight"].T
            u = h @ sd[f"{p}.mlp.up_proj.weight"].T
            del h
            h = (gelu_tanh(g) * u) @ sd[f"{p}.mlp.down_proj.weight"].T
            del g, u
            h = rmsnorm(h, sd[f"{p}.post_feedforward_layernorm.weight"])
            x = x + h
            del h

            # per-layer embedding residual
            r = x
            pg = gelu_tanh(r @ sd[f"{p}.per_layer_input_gate.weight"].T)
            pg = pg * ple[:, i, :]
            pp = pg @ sd[f"{p}.per_layer_projection.weight"].T
            del pg
            pp = rmsnorm(pp, sd[f"{p}.post_per_layer_input_norm.weight"])
            x = r + pp  # layer_scalar is 1
            del r, pp

        x = rmsnorm(x, sd[PREFIX + "norm.weight"])
        logits = x @ emb.T  # tied embeddings
        return np.tanh(logits / np.float32(SOFT_CAP)) * np.float32(SOFT_CAP)


# ---------- eval (mirrors signet_battery.py protocol) ----------
ITEMS = [
    ("math", "If a train travels 60 miles per hour, how far does it travel in 2.5 hours? Answer with just the number.", "150"),
    ("math", "What is 15 percent of 200? Answer with just the number.", "30"),
    ("math", "A rectangle has length 8 and width 5. What is its area? Answer with just the number.", "40"),
    ("math", "If you have 3 dozen eggs and use 14, how many are left? Answer with just the number.", "22"),
    ("math", "What is 7 squared plus 8 squared? Answer with just the number?", "113"),
    ("math", "A shirt costs 40 dollars and is on sale for 25 percent off. What is the sale price? Answer with just the number.", "30"),
    ("math", "How many minutes are in 3.5 hours? Answer with just the number.", "210"),
    ("math", "What is the next prime number after 17? Answer with just the number.", "19"),
    ("knowledge", "The capital of Japan is", " Tokyo"),
    ("knowledge", "Water boils at 100 degrees", " Celsius"),
    ("knowledge", "The chemical symbol for gold is", " Au"),
    ("knowledge", "The largest planet in our solar system is", " Jupiter"),
    ("knowledge", "The author of Romeo and Juliet is", " William Shakespeare"),
    ("knowledge", "The currency used in the United Kingdom is the", " pound"),
    ("knowledge", "Photosynthesis in plants mainly happens in the", " chloroplasts"),
    ("knowledge", "The year the Berlin Wall fell was", " 1989"),
    ("instruct", "Name three colors.", " Red, blue, green."),
    ("instruct", "Answer in exactly one word: what do bees make?", " Honey."),
    ("instruct", "Count from one to five.", " One, two, three, four, five."),
    ("instruct", "Say the word hello in Spanish.", " Hola."),
    ("instruct", "List two animals that can fly.", " Birds and bats."),
    ("instruct", "What is the opposite of hot? Reply with one word.", " Cold."),
    ("instruct", "Repeat after me: the sky is blue.", " The sky is blue."),
    ("instruct", "Name the first month of the year.", " January."),
]


def answer_logprob(model, ids_prompt, ids_answer):
    ids = ids_prompt + ids_answer
    logits = model.forward(ids)
    logp = 0.0
    off = len(ids_prompt)
    for j, tok_id in enumerate(ids_answer):
        row = logits[off + j - 1].astype(np.float64)
        row = row - row.max()
        e = np.exp(row)
        logp += float(np.log(e[tok_id] / e.sum()))
    return logp


def greedy_generate(model, ids, max_new):
    ids = list(ids)
    for _ in range(max_new):
        logits = model.forward(ids)
        row = logits[-1].astype(np.float64)
        row = row - row.max()
        e = np.exp(row)
        ids.append(int(np.argmax(e)))
    return ids


def eval_items(model, tok):
    scores = {"math": [], "knowledge": [], "instruct": []}
    for bucket, prompt, answer in ITEMS:
        ids_p = encode_prompt(tok, prompt)
        ids_f = encode_prompt(tok, prompt + answer)
        if len(ids_f) > len(ids_p) and ids_f[: len(ids_p)] == ids_p:
            ids_a = ids_f[len(ids_p):]
        else:
            ids_a = tok.encode(answer).ids
        if ids_a:
            scores[bucket].append(answer_logprob(model, ids_p, ids_a))
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--instruct", required=True)
    ap.add_argument("--tok", required=True)
    ap.add_argument("--sanity-only", action="store_true")
    a = ap.parse_args()

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(a.tok)

    t0 = time.time()
    print("indexing checkpoints...", flush=True)
    base_idx = load_index(a.base)
    inst_idx = load_index(a.instruct)
    keys = sorted(set(base_idx) & set(inst_idx))
    n_params = sum(int(np.prod(base_idx[k][1])) for k in keys)
    print(f"text-tower tensors: {len(keys)}, params: {n_params/1e9:.2f}B", flush=True)

    if a.sanity_only:
        print("loading instruct model for sanity check...", flush=True)
        sd = load_full(a.instruct, inst_idx, keys)
        m = Gemma4E2B(sd)
        ids = encode_prompt(tok, "The capital of France is")
        print(f"prompt tokens: {len(ids)}", flush=True)
        out = greedy_generate(m, ids, 10)
        print("sanity:", repr(tok.decode(out)[-200:]), flush=True)
        return

    rng = np.random.default_rng(SEED)
    results = {}

    # arm 1: instruct
    print("loading instruct (arm 1/4)...", flush=True)
    sd = load_full(a.instruct, inst_idx, keys)
    results["instruct"] = eval_items(Gemma4E2B(sd), tok)
    del sd
    gc.collect()

    # arm 2: base
    print("loading base (arm 2/4)...", flush=True)
    sd = load_full(a.base, base_idx, keys)
    results["base"] = eval_items(Gemma4E2B(sd), tok)
    del sd
    gc.collect()

    # delta stats (streaming, no full copies)
    print("delta stats (streaming)...", flush=True)
    tot_abs, tot_n, tot_max, pos_n = 0.0, 0, 0.0, 0
    per_tensor = []
    for k in keys:
        b = load_tensor(a.base, base_idx, k)
        d = load_tensor(a.instruct, inst_idx, k)
        d = d - b
        del b
        av = np.abs(d)
        n = d.size
        tot_abs += float(av.sum())
        tot_n += n
        tot_max = max(tot_max, float(av.max()))
        pos_n += int((d > 0).sum())
        per_tensor.append({"tensor": k, "mean_abs": round(float(av.mean()), 8),
                           "frac_positive": round(float((d > 0).mean()), 4)})
        del d, av
    print(f"delta: mean|d|={tot_abs/tot_n:.6f} max|d|={tot_max:.4f} frac+={pos_n/tot_n:.3f}", flush=True)
    gc.collect()

    # arms 3-4: grafts, built by streaming (one full model in RAM at a time)
    for kind, name in [("binary", "base+binary"), ("dare90", "base+dare90")]:
        print(f"building {name} (streaming)...", flush=True)
        grafted = {}
        for k in keys:
            b = load_tensor(a.base, base_idx, k)
            d = load_tensor(a.instruct, inst_idx, k) - b
            if kind == "binary":
                grafted[k] = b + float(np.abs(d).mean()) * np.sign(d)
            else:
                mask = (rng.random(d.shape) < 0.1).astype(np.float32)
                grafted[k] = b + d * mask / 0.1
            del b, d
        print(f"evaluating {name}...", flush=True)
        results[name] = eval_items(Gemma4E2B(grafted), tok)
        del grafted
        gc.collect()

    # base+full == instruct exactly
    results["base+full"] = results["instruct"]

    summary = {}
    for bucket in ("math", "knowledge", "instruct"):
        b = float(np.mean(results["base"][bucket]))
        r = float(np.mean(results["instruct"][bucket]))
        gap = r - b
        row = {"base": b, "instruct": r, "gap": gap}
        for name in ("base+full", "base+binary", "base+dare90"):
            v = float(np.mean(results[name][bucket]))
            row[name] = v
            row[name + "_gap_closed"] = (v - b) / gap if gap > 1e-9 else None
        summary[bucket] = row

    print("\n=== mean answer logprob per bucket ===")
    for bucket, row in summary.items():
        print(f"[{bucket}] base={row['base']:.2f} instruct={row['instruct']:.2f} gap={row['gap']:.2f}")
        for name in ("base+full", "base+binary", "base+dare90"):
            gc_ = row[name + "_gap_closed"]
            print(f"  {name}: {row[name]:.2f}  gap closed: {f'{100*gc_:.1f}%' if gc_ is not None else 'n/a'}")

    with open("gemma4_e2b_signet_results.json", "w") as f:
        json.dump({"summary": summary,
                   "delta": {"mean_abs": tot_abs / tot_n, "max_abs": tot_max,
                             "frac_positive": pos_n / tot_n, "per_tensor": per_tensor}},
                  f, indent=2)
    print(f"\nwrote gemma4_e2b_signet_results.json in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
