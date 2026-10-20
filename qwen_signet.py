"""
Signet on Qwen2.5-0.5B (+ reasoning grid probe). Runs on Flash (11GB RAM).

Part A: binary general-capability delta. Same 5 arms as the SmolLM2 run:
  base, base+full, base+binary (alpha*sign), base+dare90, instruct.
  Metric: answer logprob gap closed per bucket (math/knowledge/instruct).

Part B: reasoning grid. For trivial/easy/hard prompts, measure on instruct
  (and base for contrast on two):
  - verbosity: greedy generation, max 48 new tokens (slow on CPU, few prompts)
  - decisiveness: entropy of the first response-token distribution (cheap)

Qwen2.5-0.5B config: 24 layers, hidden 896, 14 Q heads, 2 KV heads,
head_dim 64, inter 4864, vocab 151936, rms eps 1e-6, rope theta 1e6,
tied embeddings. Full causal attention (prompts are short; the 32768
sliding window never binds).

Usage: .venv/bin/python qwen_signet.py
Writes qwen_signet_results.json and qwen_reasoning_grid.json.
"""
import json
import struct
import time

import numpy as np

DATA = "data"
BASE_W = f"{DATA}/qwen2.5-0.5b.safetensors"
INST_W = f"{DATA}/qwen2.5-0.5b-instruct.safetensors"
TOK = f"{DATA}/tokenizer.json"
TOK_CFG = f"{DATA}/tokenizer_config.json"
SEED = 7

VOCAB = 151936
HIDDEN = 896
LAYERS = 24
HEADS = 14
KV_HEADS = 2
HEAD_DIM = HIDDEN // HEADS  # 64
INTER = 4864
EPS = 1e-6
ROPE_THETA = 1000000.0

SYSTEM_DEFAULT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


def load_safetensors(path):
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
            else:
                continue
            out[name] = arr.reshape(meta["shape"]).copy()
    return out


def get_system():
    try:
        cfg = json.load(open(TOK_CFG))
        tpl = cfg.get("chat_template", "")
        # Qwen default system line inside the template; fall back if absent
        if "You are" in tpl:
            return SYSTEM_DEFAULT
    except Exception:
        pass
    return SYSTEM_DEFAULT


def format_chat(prompt, system=None):
    system = system or get_system()
    return (
        "<|im_start|>system\n" + system + "<|im_end|>\n"
        "<|im_start|>user\n" + prompt + "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def rmsnorm(x, w):
    return x / np.sqrt((x ** 2).mean(-1, keepdims=True) + EPS) * w


def rope_freqs(T):
    inv = 1.0 / (ROPE_THETA ** (np.arange(0, HEAD_DIM, 2, dtype=np.float32) / HEAD_DIM))
    f = np.outer(np.arange(T, dtype=np.float32), inv)
    e = np.concatenate([f, f], axis=-1)
    return np.cos(e).astype(np.float32), np.sin(e).astype(np.float32)


def apply_rope(x, cos, sin):
    x1, x2 = x[..., : HEAD_DIM // 2], x[..., HEAD_DIM // 2 :]
    xr = np.concatenate([-x2, x1], axis=-1)
    return x * cos[:, None, :] + xr * sin[:, None, :]


def softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


def silu(x):
    return x / (1.0 + np.exp(-x))


def linear(x, sd, key):
    y = x @ sd[key].T
    bkey = key[: -len(".weight")] + ".bias" if key.endswith(".weight") else key + ".bias"
    if bkey in sd:
        y = y + sd[bkey]
    return y


def forward(ids, sd):
    """Full forward pass -> logits (T, VOCAB)."""
    T = len(ids)
    cos, sin = rope_freqs(T)
    x = sd["model.embed_tokens.weight"][np.array(ids, dtype=np.int64)]
    for i in range(LAYERS):
        p = f"model.layers.{i}"
        h = rmsnorm(x, sd[f"{p}.input_layernorm.weight"])
        q = linear(h, sd, f"{p}.self_attn.q_proj.weight").reshape(T, HEADS, HEAD_DIM)
        k = linear(h, sd, f"{p}.self_attn.k_proj.weight").reshape(T, KV_HEADS, HEAD_DIM)
        v = linear(h, sd, f"{p}.self_attn.v_proj.weight").reshape(T, KV_HEADS, HEAD_DIM)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        k = np.repeat(k, HEADS // KV_HEADS, axis=1)
        v = np.repeat(v, HEADS // KV_HEADS, axis=1)
        att = np.einsum("thd,shd->hts", q, k) / np.sqrt(HEAD_DIM)
        mask = np.triu(np.ones((T, T), dtype=bool), 1)
        att[:, mask] = -1e9
        att = softmax(att)
        o = np.einsum("hts,shd->thd", att, v).reshape(T, HIDDEN)
        x = x + linear(o, sd, f"{p}.self_attn.o_proj.weight")
        h2 = rmsnorm(x, sd[f"{p}.post_attention_layernorm.weight"])
        gate = silu(linear(h2, sd, f"{p}.mlp.gate_proj.weight"))
        up = linear(h2, sd, f"{p}.mlp.up_proj.weight")
        x = x + linear(gate * up, sd, f"{p}.mlp.down_proj.weight")
    x = rmsnorm(x, sd["model.norm.weight"])
    return x @ sd["model.embed_tokens.weight"].T  # tied head


def answer_logprob(ids_prompt, ids_answer, sd):
    ids = ids_prompt + ids_answer
    logits = forward(ids, sd)
    logp = 0.0
    off = len(ids_prompt)
    for j, tok_id in enumerate(ids_answer):
        row = logits[off + j - 1].astype(np.float64)
        row = row - row.max()
        e = np.exp(row)
        logp += float(np.log(e[tok_id] / e.sum()))
    return logp


def greedy_generate(prompt, sd, tok, im_end_id, max_new=36):
    ids = tok.encode(format_chat(prompt)).ids
    for _ in range(max_new):
        logits = forward(ids, sd)
        nxt = int(np.argmax(logits[-1]))
        ids.append(nxt)
        if nxt == im_end_id:
            break
    return tok.decode(ids), len(ids)


ITEMS = [
    ("math", "If a train travels 60 miles per hour, how far does it travel in 2.5 hours? Answer with just the number.", "150"),
    ("math", "What is 15 percent of 200? Answer with just the number.", "30"),
    ("math", "A rectangle has length 8 and width 5. What is its area? Answer with just the number.", "40"),
    ("math", "If you have 3 dozen eggs and use 14, how many are left? Answer with just the number.", "22"),
    ("math", "What is 7 squared plus 8 squared? Answer with just the number.", "113"),
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

GRID = [
    ("trivial", "hi"),
    ("trivial", "how are you"),
    ("trivial", "good morning"),
    ("easy", "What is the capital of France?"),
    ("easy", "Name a prime number."),
    ("hard", "If a train travels 60 miles per hour for 2.5 hours, how far does it go? Explain step by step."),
    ("hard", "Write a Python function that reverses a string, with a docstring."),
]


def eval_items(tok, sd):
    scores = {"math": [], "knowledge": [], "instruct": []}
    for bucket, prompt, answer in ITEMS:
        ids_p = tok.encode(format_chat(prompt)).ids
        ids_f = tok.encode(format_chat(prompt) + answer).ids
        if len(ids_f) > len(ids_p) and ids_f[: len(ids_p)] == ids_p:
            ids_a = ids_f[len(ids_p):]
        else:
            ids_a = tok.encode(answer).ids
        if ids_a:
            scores[bucket].append(answer_logprob(ids_p, ids_a, sd))
    return scores


def summarize(results):
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
    return summary


def main():
    import gc
    from tokenizers import Tokenizer
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    tok = Tokenizer.from_file(TOK)
    im_end_id = tok.encode("<|im_end|>").ids[0]

    print("loading weights...", flush=True)
    base = load_safetensors(BASE_W)
    inst = load_safetensors(INST_W)
    assert set(base) == set(inst), "tensor name mismatch"
    print(f"loaded {len(base)} tensors", flush=True)

    # ---- Part B: reasoning grid (needs base + instruct together) ----
    print("=== part B: entropy grid ===", flush=True)
    grid = []
    for level, prompt in GRID:
        ids_p = tok.encode(format_chat(prompt)).ids
        row = {"level": level, "prompt": prompt}
        for name, sd in (("base", base), ("instruct", inst)):
            logits = forward(ids_p, sd)
            p = softmax(logits[-1].astype(np.float64))
            row[f"{name}_top1"] = round(float(p.max()), 4)
            row[f"{name}_entropy"] = round(float(-(p * np.log(p + 1e-12)).sum()), 3)
        grid.append(row)
        print(f"[{level}] '{prompt[:36]}' ent base={row['base_entropy']:.2f} instruct={row['instruct_entropy']:.2f}", flush=True)

    print("=== part B: greedy generations (instruct) ===", flush=True)
    gens = []
    for level, prompt in [GRID[0], GRID[1], GRID[3], GRID[5]]:
        g0 = time.time()
        text, n = greedy_generate(prompt, inst, tok, im_end_id)
        resp = text.split("<|im_start|>assistant\n")[-1].replace("<|im_end|>", "").strip()
        gens.append({"level": level, "prompt": prompt, "n_tokens": n, "response": resp[:300]})
        print(f"[{level}] {n} tok in {time.time()-g0:.0f}s: {resp[:100]}", flush=True)

    # ---- Part A: delta + arms, evaluated sequentially to stay in RAM ----
    keys = [k for k in base if base[k].dtype.kind == "f"]
    deltas = {k: inst[k] - base[k] for k in keys}
    for k in (keys[0], keys[len(keys) // 2], keys[-1]):
        chk = base[k] + deltas[k]
        print(f"graft sanity {k}: max|diff|={float(np.abs(chk - inst[k]).max()):.2e}", flush=True)
        del chk

    results = {}
    print("evaluating instruct arm...", flush=True)
    results["instruct"] = eval_items(tok, inst)
    print("evaluating base arm...", flush=True)
    results["base"] = eval_items(tok, base)
    del inst
    gc.collect()

    # delta stats streamed per tensor: no second full copy (the old
    # np.concatenate(all) needed another 2GB and OOM-killed Flash)
    tot, cnt, mx = 0.0, 0, 0.0
    for k in keys:
        a = np.abs(deltas[k])
        tot += float(a.sum()); cnt += a.size; mx = max(mx, float(a.max()))
        del a
    mean_abs = tot / cnt
    print(f"delta: mean|d|={mean_abs:.6f} max|d|={mx:.4f}", flush=True)

    def build_graft(kind):
        g = {}
        for k in base:
            if k not in deltas:
                g[k] = base[k].copy()
            elif kind == "full":
                g[k] = base[k] + deltas[k]
            elif kind == "binary":
                d = deltas[k]
                g[k] = base[k] + float(np.abs(d).mean()) * np.sign(d)
            elif kind == "dare90":
                mask = (rng.random(deltas[k].shape) < 0.1).astype(np.float32)
                g[k] = base[k] + deltas[k] * mask / 0.1
        return g

    for kind, name in [("full", "base+full"), ("binary", "base+binary"), ("dare90", "base+dare90")]:
        print(f"evaluating {name}...", flush=True)
        grafted = build_graft(kind)
        results[name] = eval_items(tok, grafted)
        del grafted
        gc.collect()

    summary = summarize(results)
    print("\n=== part A: mean answer logprob per bucket ===")
    for bucket, row in summary.items():
        print(f"[{bucket}] base={row['base']:.2f} instruct={row['instruct']:.2f} gap={row['gap']:.2f}")
        for name in ("base+full", "base+binary", "base+dare90"):
            gc_ = row[name + "_gap_closed"]
            print(f"  {name}: {row[name]:.2f}  gap closed: {f'{100*gc_:.1f}%' if gc_ is not None else 'n/a'}")

    with open("qwen_signet_results.json", "w") as f:
        json.dump({"part_a_summary": summary, "delta_mean_abs": mean_abs}, f, indent=2)
    with open("qwen_reasoning_grid.json", "w") as f:
        json.dump({"grid": grid, "generations": gens}, f, indent=2)
    print(f"\nwrote qwen_signet_results.json + qwen_reasoning_grid.json in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
