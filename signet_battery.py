"""
Signet battery runner: the three staged experiments on Flash (11GB RAM).

Models (all numpy, no torch/transformers):
  smollm-135m   5-arm sign test (base, base+full, base+binary, base+dare90, instruct)
                HuggingFaceTB/SmolLM-135M vs SmolLM-135M-Instruct (ChatML template)
  smollm2-360m  5-arm sign test, same protocol
                HuggingFaceTB/SmolLM2-360M vs SmolLM2-360M-Instruct (ChatML template)
  qwen3-0.6b    thinking-mode proportionality test on Qwen3-0.6B(-Instruct):
                thinking ON vs OFF across a trivial/easy/hard grid; measures
                thinking-token count, answer verbosity, latency, and first-token
                entropy of the answer. Plus streaming base->instruct delta sign
                statistics (serves the cross-model delta/sign comparison).

Memory discipline (Flash OOM lesson, Oct 13): arms built and evaluated
sequentially, delta statistics streamed tensor-by-tensor, explicit gc,
no | tail masking of exit codes. Peak stays well under 6GB.

Usage (from ~/qwen-test on Flash):
  .venv/bin/python signet_battery.py --model smollm-135m \
      --base data/<base>.safetensors --instruct data/<instruct>.safetensors \
      --tok data/<tokenizer>.json
  .venv/bin/python signet_battery.py --model qwen3-0.6b \
      --instruct data/<qwen3-instruct>.safetensors --base data/<qwen3-base>.safetensors \
      --tok data/<tokenizer>.json

Writes <model>_signet_results.json (+ reasoning grid for the sign tests,
thinking results for qwen3).
"""
import argparse
import gc
import json
import struct
import sys
import time

import numpy as np

SEED = 7

# ---- model configs (verified against HF config.json values) ----
MODELS = {
    "smollm-135m": dict(
        layers=30, hidden=576, heads=9, kv_heads=3, inter=1536,
        vocab=49152, eps=1e-5, theta=10000.0, qk_norm=False,
        out="smollm135_signet_results.json", grid_out="smollm135_reasoning_grid.json",
        kind="sign",
    ),
    "smollm2-360m": dict(
        layers=16, hidden=960, heads=15, kv_heads=5, inter=2560,
        vocab=49152, eps=1e-5, theta=100000.0, qk_norm=False,
        out="smollm2_360_signet_results.json", grid_out="smollm2_360_reasoning_grid.json",
        kind="sign",
    ),
    "qwen3-0.6b": dict(
        layers=28, hidden=1024, heads=16, kv_heads=8, inter=3072,
        vocab=151936, eps=1e-6, theta=1000000.0, qk_norm=True,
        out="qwen3_thinking_results.json",
        kind="thinking",
    ),
}

SYSTEM_DEFAULT = "You are a helpful assistant."


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


def format_chat(prompt, system=None):
    system = system or SYSTEM_DEFAULT
    return (
        "<|im_start|>system\n" + system + "<|im_end|>\n"
        "<|im_start|>user\n" + prompt + "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def rmsnorm(x, w, eps):
    return x / np.sqrt((x ** 2).mean(-1, keepdims=True) + eps) * w


def rope_freqs(T, head_dim, theta):
    inv = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    f = np.outer(np.arange(T, dtype=np.float32), inv)
    e = np.concatenate([f, f], axis=-1)
    return np.cos(e).astype(np.float32), np.sin(e).astype(np.float32)


def apply_rope(x, cos, sin, head_dim):
    x1, x2 = x[..., : head_dim // 2], x[..., head_dim // 2 :]
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
    # AGENTS.md lesson: derive bias as key.replace(".weight", ".bias"),
    # never key + ".bias" (silently never matches -> silently skipped bias).
    bkey = key.replace(".weight", ".bias")
    if bkey in sd:
        y = y + sd[bkey]
    return y


class Model:
    def __init__(self, cfg, sd):
        self.cfg = cfg
        self.sd = sd
        self.layers = cfg["layers"]
        self.hidden = cfg["hidden"]
        self.heads = cfg["heads"]
        self.kv_heads = cfg["kv_heads"]
        self.head_dim = cfg["hidden"] // cfg["heads"]
        self.inter = cfg["inter"]
        self.eps = cfg["eps"]
        self.theta = cfg["theta"]
        self.qk_norm = cfg["qk_norm"]
        self.tied = "lm_head.weight" not in sd
        self._assert_shapes()

    def _assert_shapes(self):
        sd, c = self.sd, self.cfg
        hd = self.head_dim
        assert sd["model.embed_tokens.weight"].shape == (c["vocab"], c["hidden"]), \
            f"embed shape {sd['model.embed_tokens.weight'].shape}"
        p = "model.layers.0"
        assert sd[f"{p}.self_attn.q_proj.weight"].shape == (c["heads"] * hd, c["hidden"])
        assert sd[f"{p}.self_attn.k_proj.weight"].shape == (c["kv_heads"] * hd, c["hidden"])
        assert sd[f"{p}.mlp.gate_proj.weight"].shape == (c["inter"], c["hidden"])
        if self.qk_norm:
            assert f"{p}.self_attn.q_norm.weight" in sd, "missing q_norm for qk_norm model"
        if not self.tied:
            assert sd["lm_head.weight"].shape == (c["vocab"], c["hidden"])
        print(f"shape check ok: tied={self.tied} qk_norm={self.qk_norm}", flush=True)

    def forward(self, ids):
        sd, c = self.sd, self.cfg
        T = len(ids)
        cos, sin = rope_freqs(T, self.head_dim, self.theta)
        x = sd["model.embed_tokens.weight"][np.array(ids, dtype=np.int64)]
        for i in range(self.layers):
            p = f"model.layers.{i}"
            h = rmsnorm(x, sd[f"{p}.input_layernorm.weight"], self.eps)
            q = linear(h, sd, f"{p}.self_attn.q_proj.weight").reshape(T, self.heads, self.head_dim)
            k = linear(h, sd, f"{p}.self_attn.k_proj.weight").reshape(T, self.kv_heads, self.head_dim)
            v = linear(h, sd, f"{p}.self_attn.v_proj.weight").reshape(T, self.kv_heads, self.head_dim)
            if self.qk_norm:
                q = rmsnorm(q, sd[f"{p}.self_attn.q_norm.weight"], self.eps)
                k = rmsnorm(k, sd[f"{p}.self_attn.k_norm.weight"], self.eps)
            q, k = apply_rope(q, cos, sin, self.head_dim), apply_rope(k, cos, sin, self.head_dim)
            rep = self.heads // self.kv_heads
            if rep > 1:
                k = np.repeat(k, rep, axis=1)
                v = np.repeat(v, rep, axis=1)
            att = np.einsum("thd,shd->hts", q, k) / np.sqrt(self.head_dim)
            mask = np.triu(np.ones((T, T), dtype=bool), 1)
            att[:, mask] = -1e9
            att = softmax(att)
            o = np.einsum("hts,shd->thd", att, v).reshape(T, self.hidden)
            x = x + linear(o, sd, f"{p}.self_attn.o_proj.weight")
            h2 = rmsnorm(x, sd[f"{p}.post_attention_layernorm.weight"], self.eps)
            gate = silu(linear(h2, sd, f"{p}.mlp.gate_proj.weight"))
            up = linear(h2, sd, f"{p}.mlp.up_proj.weight")
            x = x + linear(gate * up, sd, f"{p}.mlp.down_proj.weight")
        x = rmsnorm(x, sd["model.norm.weight"], self.eps)
        head = sd["model.embed_tokens.weight"] if self.tied else sd["lm_head.weight"]
        return x @ head.T


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


def greedy_generate(model, ids, max_new, stop_ids=()):
    """Greedy decode. Returns (ids_out, per_step list of (top1_prob, entropy))."""
    ids = list(ids)
    steps = []
    stop = set(stop_ids)
    for _ in range(max_new):
        logits = model.forward(ids)
        row = logits[-1].astype(np.float64)
        row = row - row.max()
        e = np.exp(row)
        p = e / e.sum()
        nxt = int(np.argmax(p))
        steps.append((round(float(p.max()), 4), round(float(-(p * np.log(p + 1e-12)).sum()), 3)))
        ids.append(nxt)
        if nxt in stop:
            break
    return ids, steps


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


def eval_items(model, tok):
    scores = {"math": [], "knowledge": [], "instruct": []}
    for bucket, prompt, answer in ITEMS:
        ids_p = tok.encode(format_chat(prompt)).ids
        ids_f = tok.encode(format_chat(prompt) + answer).ids
        if len(ids_f) > len(ids_p) and ids_f[: len(ids_p)] == ids_p:
            ids_a = ids_f[len(ids_p):]
        else:
            ids_a = tok.encode(answer).ids
        if ids_a:
            scores[bucket].append(answer_logprob(model, ids_p, ids_a))
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


def sanity_generate(model, tok, label):
    """Coherence check (lesson of the voided qwen run): base should at least
    complete a factual prompt sensibly. Prints; does not hard-fail."""
    ids = tok.encode(format_chat("The capital of France is")).ids
    out, _ = greedy_generate(model, ids, 12)
    text = tok.decode(out).split("<|im_start|>assistant\n")[-1].replace("<|im_end|>", "").strip()
    print(f"sanity [{label}]: {text[:120]!r}", flush=True)
    return text


def run_sign_test(cfg, base_sd, inst_sd, tok):
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    im_end_id = tok.encode("<|im_end|>").ids[0]

    print("building models...", flush=True)
    base_m = Model(cfg, base_sd)
    inst_m = Model(cfg, inst_sd)
    sanity_generate(base_m, tok, "base")
    sanity_generate(inst_m, tok, "instruct")

    # ---- reasoning grid (needs base + instruct together) ----
    print("=== entropy grid ===", flush=True)
    grid = []
    for level, prompt in GRID:
        ids_p = tok.encode(format_chat(prompt)).ids
        row = {"level": level, "prompt": prompt}
        for name, m in (("base", base_m), ("instruct", inst_m)):
            logits = m.forward(ids_p)
            p = softmax(logits[-1].astype(np.float64))
            row[f"{name}_top1"] = round(float(p.max()), 4)
            row[f"{name}_entropy"] = round(float(-(p * np.log(p + 1e-12)).sum()), 3)
        grid.append(row)
        print(f"[{level}] '{prompt[:36]}' ent base={row['base_entropy']:.2f} instruct={row['instruct_entropy']:.2f}", flush=True)

    print("=== greedy generations (instruct) ===", flush=True)
    gens = []
    for level, prompt in [GRID[0], GRID[1], GRID[3], GRID[5]]:
        g0 = time.time()
        out, _ = greedy_generate(inst_m, tok.encode(format_chat(prompt)).ids, 36, (im_end_id,))
        text = tok.decode(out)
        resp = text.split("<|im_start|>assistant\n")[-1].replace("<|im_end|>", "").strip()
        gens.append({"level": level, "prompt": prompt, "n_tokens": len(out), "response": resp[:300]})
        print(f"[{level}] {time.time()-g0:.0f}s: {resp[:100]}", flush=True)

    # ---- delta + arms, evaluated sequentially (memory discipline) ----
    keys = [k for k in base_sd if base_sd[k].dtype.kind == "f"]
    deltas = {k: inst_sd[k] - base_sd[k] for k in keys}
    for k in (keys[0], keys[len(keys) // 2], keys[-1]):
        chk = base_sd[k] + deltas[k]
        print(f"graft sanity {k}: max|diff|={float(np.abs(chk - inst_sd[k]).max()):.2e}", flush=True)
        del chk

    results = {}
    print("evaluating instruct arm...", flush=True)
    results["instruct"] = eval_items(inst_m, tok)
    print("evaluating base arm...", flush=True)
    results["base"] = eval_items(base_m, tok)
    del inst_m, inst_sd
    gc.collect()

    tot, cnt, mx = 0.0, 0, 0.0
    for k in keys:
        a = np.abs(deltas[k])
        tot += float(a.sum()); cnt += a.size; mx = max(mx, float(a.max()))
        del a
    mean_abs = tot / cnt
    print(f"delta: mean|d|={mean_abs:.6f} max|d|={mx:.4f}", flush=True)

    def build_graft(kind):
        g = {}
        for k in base_sd:
            if k not in deltas:
                g[k] = base_sd[k].copy()
            elif kind == "full":
                g[k] = base_sd[k] + deltas[k]
            elif kind == "binary":
                d = deltas[k]
                g[k] = base_sd[k] + float(np.abs(d).mean()) * np.sign(d)
            elif kind == "dare90":
                mask = (rng.random(deltas[k].shape) < 0.1).astype(np.float32)
                g[k] = base_sd[k] + deltas[k] * mask / 0.1
        return g

    for kind, name in [("full", "base+full"), ("binary", "base+binary"), ("dare90", "base+dare90")]:
        print(f"evaluating {name}...", flush=True)
        grafted = build_graft(kind)
        results[name] = eval_items(Model(cfg, grafted), tok)
        del grafted
        gc.collect()

    summary = summarize(results)
    print("\n=== part A: mean answer logprob per bucket ===")
    for bucket, row in summary.items():
        print(f"[{bucket}] base={row['base']:.2f} instruct={row['instruct']:.2f} gap={row['gap']:.2f}")
        for name in ("base+full", "base+binary", "base+dare90"):
            gc_ = row[name + "_gap_closed"]
            print(f"  {name}: {row[name]:.2f}  gap closed: {f'{100*gc_:.1f}%' if gc_ is not None else 'n/a'}")

    with open(cfg["out"], "w") as f:
        json.dump({"part_a_summary": summary, "delta_mean_abs": mean_abs}, f, indent=2)
    with open(cfg["grid_out"], "w") as f:
        json.dump({"grid": grid, "generations": gens}, f, indent=2)
    print(f"\nwrote {cfg['out']} + {cfg['grid_out']} in {time.time()-t0:.0f}s")


def delta_sign_stats(base_sd, inst_sd):
    """Streaming base->instruct delta sign statistics (serves the cross-model
    delta/sign comparison). No second full model copy held."""
    keys = [k for k in base_sd if base_sd[k].dtype.kind == "f" and k in inst_sd]
    tot_abs, tot_n, tot_max = 0.0, 0, 0.0
    pos_n = 0
    per_tensor = []
    for k in keys:
        d = inst_sd[k] - base_sd[k]
        a = np.abs(d)
        n = d.size
        tot_abs += float(a.sum()); tot_n += n
        tot_max = max(tot_max, float(a.max()))
        pos_n += int((d > 0).sum())
        per_tensor.append({"tensor": k, "mean_abs": round(float(a.mean()), 8),
                           "frac_positive": round(float((d > 0).mean()), 4)})
        del d, a
    stats = {"n_tensors": len(keys),
             "mean_abs_delta": tot_abs / tot_n,
             "max_abs_delta": tot_max,
             "frac_positive_overall": pos_n / tot_n,
             "per_tensor": per_tensor}
    print(f"delta stats: {len(keys)} tensors mean|d|={stats['mean_abs_delta']:.6f} "
          f"frac+={stats['frac_positive_overall']:.3f}", flush=True)
    return stats


def run_thinking_test(cfg, base_sd, inst_sd, tok):
    """Qwen3-0.6B thinking proportionality: thinking ON vs OFF over the grid.
    Hypothesis (Vinay): thinking effort should scale with input difficulty."""
    t0 = time.time()

    # delta sign stats first, then drop base to keep peak memory down
    print("=== delta sign stats (base -> instruct) ===", flush=True)
    stats = delta_sign_stats(base_sd, inst_sd)
    del base_sd
    gc.collect()

    print("building instruct model...", flush=True)
    m = Model(cfg, inst_sd)
    sanity_generate(m, tok, "instruct")

    im_end_ids = tok.encode("<|im_end|>").ids
    think_close_ids = tok.encode("</think>").ids

    def find_subseq(hay, needle):
        n = len(needle)
        for i in range(len(hay) - n + 1):
            if hay[i:i + n] == needle:
                return i + n
        return -1

    rows = []
    for level, prompt in GRID:
        base_ids = tok.encode(format_chat(prompt)).ids
        for mode in ("on", "off"):
            if mode == "on":
                ids = base_ids + tok.encode("<|im_start|>assistant\n").ids
                max_new = 160
            else:
                # Qwen3 convention for disabled thinking: empty think block
                ids = base_ids + tok.encode("<|im_start|>assistant\n<think>\n\n</think>\n\n").ids
                max_new = 48
            g0 = time.time()
            out, steps = greedy_generate(m, ids, max_new, tuple(im_end_ids))
            dt = time.time() - g0
            gen_ids = out[len(ids):]
            text = tok.decode(out)
            tail = text.split("<|im_start|>assistant\n")[-1]
            if mode == "on":
                pos = find_subseq(gen_ids, think_close_ids)
                if pos >= 0:
                    think_ids, ans_ids = gen_ids[:pos], gen_ids[pos:]
                    ans_entropy = steps[pos][1] if pos < len(steps) else None
                else:
                    think_ids, ans_ids = gen_ids, []
                    ans_entropy = None
                think_text = tok.decode(think_ids)
                ans_text = tok.decode(ans_ids).replace("<|im_end|>", "").strip()
                row = {"mode": "thinking_on",
                       "thinking_tokens": len(think_ids), "answer_tokens": len(ans_ids),
                       "answer_first_token_entropy": ans_entropy,
                       "thinking_preview": think_text[:200],
                       "answer_preview": ans_text[:200]}
            else:
                ans_text = tail.replace("<|im_end|>", "").strip()
                row = {"mode": "thinking_off",
                       "thinking_tokens": 0, "answer_tokens": len(gen_ids),
                       "answer_first_token_entropy": steps[0][1] if steps else None,
                       "answer_preview": ans_text[:200]}
            row.update({"level": level, "prompt": prompt,
                        "total_new_tokens": len(gen_ids),
                        "seconds": round(dt, 1),
                        "tok_per_s": round(len(gen_ids) / max(dt, 1e-6), 2)})
            rows.append(row)
            print(f"[{level}][{row['mode']}] think={row['thinking_tokens']} "
                  f"ans={row['answer_tokens']} {row['seconds']}s "
                  f"ent={row['answer_first_token_entropy']}: {row['answer_preview'][:80]}",
                  flush=True)

    with open(cfg["out"], "w") as f:
        json.dump({"delta_sign_stats": stats, "thinking_grid": rows}, f, indent=2)
    print(f"\nwrote {cfg['out']} in {time.time()-t0:.0f}s")


def check_chat_template(tok_path, expect_im_start=True):
    """Warn if the tokenizer's chat template does not look like ChatML.
    (v0.2 lesson: the actual instruct template must be applied to every arm.)"""
    try:
        cfg = json.load(open(tok_path.replace("tokenizer.json", "tokenizer_config.json")))
    except Exception:
        try:
            cfg = json.load(open("tokenizer_config.json"))
        except Exception:
            print("template check: no tokenizer_config.json found, proceeding", flush=True)
            return
    tpl = cfg.get("chat_template", "")
    has = "<|im_start|>" in tpl
    print(f"template check: chat_template has <|im_start|>={has}", flush=True)
    if expect_im_start and not has:
        print("WARNING: tokenizer template is not ChatML; hardcoded format may mismatch!", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--base", required=True)
    ap.add_argument("--instruct", required=True)
    ap.add_argument("--tok", required=True)
    a = ap.parse_args()
    cfg = MODELS[a.model]

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(a.tok)
    if cfg["kind"] == "sign":
        check_chat_template(a.tok, expect_im_start=True)

    print(f"loading {a.model} weights...", flush=True)
    base_sd = load_safetensors(a.base)
    inst_sd = load_safetensors(a.instruct)
    bkeys = {k for k in base_sd if base_sd[k].dtype.kind == "f"}
    ikeys = {k for k in inst_sd if inst_sd[k].dtype.kind == "f"}
    assert bkeys == ikeys, f"tensor name mismatch: {len(bkeys ^ ikeys)} differ"
    print(f"loaded {len(base_sd)} tensors", flush=True)

    if cfg["kind"] == "sign":
        run_sign_test(cfg, base_sd, inst_sd, tok)
    else:
        run_thinking_test(cfg, base_sd, inst_sd, tok)


if __name__ == "__main__":
    main()
