"""Per-layer RMS stats to find where Gemma4 E2B forward goes wrong."""
import sys
import numpy as np

sys.path.insert(0, r"D:\gemma4-e2b")
import signet_gemma4_e2b as G
from tokenizers import Tokenizer

tok = Tokenizer.from_file(r"D:\gemma4-e2b\it\tokenizer.json")
idx = G.load_index(r"D:\gemma4-e2b\it\model.safetensors")
keys = sorted(set(idx))
sd = G.load_full(r"D:\gemma4-e2b\it\model.safetensors", idx, keys)

def rms(x):
    return float(np.sqrt((x.astype(np.float64) ** 2).mean()))

ids = G.encode_prompt(tok, "The capital of France is")
T = len(ids)
print("T =", T)

# replicate forward with stats
P, HIDDEN, HEADS = G.PREFIX, G.HIDDEN, G.HEADS
cos_s, sin_s = G.rope_freqs(T, G.ROPE_HD["sliding_attention"], 10000.0)
cos_f, sin_f = G.rope_freqs(T, G.ROPE_HD["full_attention"], 1000000.0,
                            partial_rotary_factor=0.25)
emb = sd[P + "embed_tokens.weight"]
x = emb[np.array(ids, dtype=np.int64)] * np.float32(np.sqrt(HIDDEN))
print(f"after embed: rms={rms(x):.3f}")

ple_tok = sd[P + "embed_tokens_per_layer.weight"][np.array(ids, dtype=np.int64)]
ple_tok = ple_tok.reshape(T, G.PLE_LAYERS, G.PLE_DIM) * np.float32(16.0)
ctx = (x @ sd[P + "per_layer_model_projection.weight"].T) * np.float32(HIDDEN ** -0.5)
ctx = ctx.reshape(T, G.PLE_LAYERS, G.PLE_DIM)
ctx = G.rmsnorm(ctx, sd[P + "per_layer_projection_norm.weight"])
ple = (ctx + ple_tok) * np.float32(2.0 ** -0.5)
print(f"ple: rms={rms(ple):.3f}")
del ctx, ple_tok

tri = np.triu(np.ones((T, T), dtype=bool), 1)
win_mask = np.arange(T)[None, :] < np.arange(T)[:, None] - 511
shared = {}
for i in list(range(3)) + [13, 14, 15, 34]:
    p = f"{P}layers.{i}"
    ltype = G.LAYER_TYPES[i]
    hd = sd[f"{p}.self_attn.q_proj.weight"].shape[0] // HEADS
    cos, sin = (cos_s, sin_s) if ltype == "sliding_attention" else (cos_f, sin_f)
    h = G.rmsnorm(x, sd[f"{p}.input_layernorm.weight"])
    q = (h @ sd[f"{p}.self_attn.q_proj.weight"].T).reshape(T, HEADS, hd)
    q = G.rmsnorm(q, sd[f"{p}.self_attn.q_norm.weight"])
    q = G.apply_rope(q, cos, sin, hd)
    if i < G.FIRST_KV_SHARED:
        k = (h @ sd[f"{p}.self_attn.k_proj.weight"].T).reshape(T, G.KV_HEADS, hd)
        v = (h @ sd[f"{p}.self_attn.v_proj.weight"].T).reshape(T, G.KV_HEADS, hd)
        k = G.rmsnorm(k, sd[f"{p}.self_attn.k_norm.weight"])
        k = G.apply_rope(k, cos, sin, hd)
        v = G.rmsnorm_noscale(v)
        kt, vt = k.transpose(1, 0, 2), v.transpose(1, 0, 2)
        if i == G.STORE_SLIDING:
            shared["sliding_attention"] = (kt, vt)
        elif i == G.STORE_FULL:
            shared["full_attention"] = (kt, vt)
    else:
        kt, vt = shared[ltype]
    qt = q.transpose(1, 0, 2)
    scores = qt @ kt.transpose(0, 2, 1)
    mask = tri if ltype == "full_attention" else (tri | win_mask)
    scores[:, mask] = np.float32(-1e9)
    # attention entropy for last query position
    sm = G.softmax(scores)
    ent = float(-(sm[0, -1] * np.log(sm[0, -1] + 1e-12)).sum())
    attn = sm
    o = (attn @ vt).transpose(1, 0, 2).reshape(T, HEADS * hd)
    o = o @ sd[f"{p}.self_attn.o_proj.weight"].T
    o = G.rmsnorm(o, sd[f"{p}.post_attention_layernorm.weight"])
    x = x + o
    print(f"layer {i} ({ltype[:4]}, hd={hd}): rms(x)={rms(x):.3f} rms(o)={rms(o):.3f} attn_ent={ent:.2f}")
print("done")
