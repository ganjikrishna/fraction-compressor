"""
Signet v0.2: binary general-capability delta experiment.

v0.2 fix: v0.1 ran prompts raw, so instruct was out of its native format and
scored below base on math/instruction items (no measurable gap). Now every
prompt is wrapped in the SmolLM2-Instruct chat template for all arms.

Question: how much of the base -> instruct capability gap is carried by the
SIGN of the weight delta alone (1 bit per parameter)?

Arms (all on SmolLM2-135M, pure numpy):
  base            control
  base+full       base + (instruct - base)   -- must equal instruct; validates the graft pipeline
  base+binary     base + alpha * sign(delta), alpha = mean|delta| per tensor (BitDelta-style)
  base+dare90     base + delta with 90% dropped at random, survivors x10 (sparsity comparison)
  instruct        reference

Metric: for each eval item (prompt + reference answer), sum of log-probs of
the answer tokens given the prompt. Gap closed per bucket:
    (variant - base) / (instruct - base)
Buckets: math (8), knowledge (8), instruction-following (8). All auto-scored,
no judge model. Honest limit: logprob of a reference answer is a proxy for
capability, not a full benchmark; 24 items is a probe, not a leaderboard.

Usage: .venv/bin/python signet.py
Writes signet_results.json.
"""
import json
import time

import numpy as np
from tokenizers import Tokenizer

from benchmark import load_safetensors
from profile3 import run_prompt

BASE_W = "data/smollm2-135m.safetensors"
INST_W = "data/smollm2-135m-instruct.safetensors"
TOK = "data/tokenizer.json"
SEED = 7

# SmolLM2-Instruct chat template, verbatim from tokenizer_config.json on
# HuggingFace (single user turn, add_generation_prompt=True). Applied to ALL
# arms: the delta was learned in this format, so this is the fair habitat.
SYSTEM = "You are a helpful AI assistant named SmolLM, trained by Hugging Face"


def format_chat(prompt):
    return (
        "<|im_start|>system\n" + SYSTEM + "<|im_end|>\n"
        "<|im_start|>user\n" + prompt + "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

# (bucket, prompt, reference answer)
ITEMS = [
    # math: short numeric answers
    ("math", "If a train travels 60 miles per hour, how far does it travel in 2.5 hours? Answer with just the number.", "150"),
    ("math", "What is 15 percent of 200? Answer with just the number.", "30"),
    ("math", "A rectangle has length 8 and width 5. What is its area? Answer with just the number.", "40"),
    ("math", "If you have 3 dozen eggs and use 14, how many are left? Answer with just the number.", "22"),
    ("math", "What is 7 squared plus 8 squared? Answer with just the number.", "113"),
    ("math", "A shirt costs 40 dollars and is on sale for 25 percent off. What is the sale price? Answer with just the number.", "30"),
    ("math", "How many minutes are in 3.5 hours? Answer with just the number.", "210"),
    ("math", "What is the next prime number after 17? Answer with just the number.", "19"),
    # knowledge: short factual answers
    ("knowledge", "The capital of Japan is", " Tokyo"),
    ("knowledge", "Water boils at 100 degrees", " Celsius"),
    ("knowledge", "The chemical symbol for gold is", " Au"),
    ("knowledge", "The largest planet in our solar system is", " Jupiter"),
    ("knowledge", "The author of Romeo and Juliet is", " William Shakespeare"),
    ("knowledge", "The currency used in the United Kingdom is the", " pound"),
    ("knowledge", "Photosynthesis in plants mainly happens in the", " chloroplasts"),
    ("knowledge", "The year the Berlin Wall fell was", " 1989"),
    # instruction-following: prompt formatted uniformly, reference follows it
    ("instruct", "Instruction: Name three colors.\nAnswer:", " Red, blue, green."),
    ("instruct", "Instruction: Answer in exactly one word: what do bees make?\nAnswer:", " Honey."),
    ("instruct", "Instruction: Count from one to five.\nAnswer:", " One, two, three, four, five."),
    ("instruct", "Instruction: Say the word hello in Spanish.\nAnswer:", " Hola."),
    ("instruct", "Instruction: List two animals that can fly.\nAnswer:", " Birds and bats."),
    ("instruct", "Instruction: What is the opposite of hot? Reply with one word.\nAnswer:", " Cold."),
    ("instruct", "Instruction: Repeat after me: the sky is blue.\nAnswer:", " The sky is blue."),
    ("instruct", "Instruction: Name the first month of the year.\nAnswer:", " January."),
]


def answer_logprob(ids_prompt, ids_answer, sd):
    """Sum of log-probs of answer tokens given the prompt (single forward)."""
    ids = ids_prompt + ids_answer
    # run_prompt expects a usage accumulator keyed like profile3.main builds it
    usage = {k: [0.0, 0] for k in sd if sd[k].ndim == 2}
    usage["lm_head (tied to embed_tokens)"] = [0.0, 0]
    hidden = run_prompt(ids, sd, usage)  # (T, 576); usage discarded
    # final rmsnorm (missing from run_prompt, which was written for profiling)
    # then tied lm head
    from profile3 import rmsnorm
    normed = rmsnorm(hidden, sd["model.norm.weight"])
    logits = normed @ sd["model.embed_tokens.weight"].T
    logp = 0.0
    off = len(ids_prompt)
    for j, tok_id in enumerate(ids_answer):
        row = logits[off + j - 1]  # predict token j from prefix ending before it
        row = row - row.max()
        e = np.exp(row.astype(np.float64))
        logp += float(np.log(e[tok_id] / e.sum()))
    return logp


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    print("loading weights...", flush=True)
    base = load_safetensors(BASE_W)
    inst = load_safetensors(INST_W)
    assert set(base) == set(inst), "tensor name mismatch"
    for k in base:
        assert base[k].shape == inst[k].shape, f"shape mismatch {k}"
    print(f"loaded {len(base)} tensors in {time.time()-t0:.1f}s", flush=True)

    # delta stats
    keys = [k for k in base if base[k].dtype.kind == "f"]
    deltas = {k: inst[k] - base[k] for k in keys}
    all_abs = np.concatenate([np.abs(deltas[k]).ravel() for k in keys])
    print(f"delta: mean|d|={all_abs.mean():.6f} max|d|={all_abs.max():.4f} "
          f"median|d|={np.median(all_abs):.6f}", flush=True)

    # build arms
    arms = {}
    arms["base"] = base
    full = {k: base[k] + deltas[k] if k in deltas else base[k].copy() for k in base}
    arms["base+full"] = full
    binary = {}
    for k in base:
        if k not in deltas:
            binary[k] = base[k].copy()
            continue
        d = deltas[k]
        alpha = float(np.abs(d).mean())
        binary[k] = base[k] + alpha * np.sign(d)
    arms["base+binary"] = binary
    dare = {}
    for k in base:
        if k not in deltas:
            dare[k] = base[k].copy()
            continue
        mask = (rng.random(deltas[k].shape) < 0.1).astype(np.float32)
        dare[k] = base[k] + deltas[k] * mask / 0.1
    arms["base+dare90"] = dare
    arms["instruct"] = inst

    # sanity: base+full must equal instruct
    diffs = [float(np.abs(arms["base+full"][k] - inst[k]).max()) for k in keys]
    print(f"graft sanity: max|base+full - instruct| = {max(diffs):.2e} (want ~0)", flush=True)

    tok = Tokenizer.from_file(TOK)
    results = {name: {"math": [], "knowledge": [], "instruct": []} for name in arms}
    n = len(ITEMS)
    for i, (bucket, prompt, answer) in enumerate(ITEMS):
        ids_p = tok.encode(format_chat(prompt)).ids
        ids_full = tok.encode(prompt + answer).ids
        # answer ids = suffix beyond the prompt encoding (handles spacing)
        if len(ids_full) > len(ids_p) and ids_full[: len(ids_p)] == ids_p:
            ids_a = ids_full[len(ids_p):]
        else:  # fallback: re-encode answer alone
            ids_a = tok.encode(answer).ids
        if not ids_a:
            continue
        for name, sd in arms.items():
            lp = answer_logprob(ids_p, ids_a, sd)
            results[name][bucket].append(lp)
        print(f"item {i+1}/{n} [{bucket}] done", flush=True)

    # summarize: mean logprob per bucket, gap closed vs instruct
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

    out = {
        "arms": list(arms),
        "n_items": n,
        "delta_stats": {
            "mean_abs": float(all_abs.mean()),
            "max_abs": float(all_abs.max()),
            "median_abs": float(np.median(all_abs)),
        },
        "summary": summary,
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open("signet_results.json", "w") as f:
        json.dump(out, f, indent=2)

    print("\n=== signet v0.1: mean answer logprob per bucket ===")
    for bucket, row in summary.items():
        print(f"[{bucket}] base={row['base']:.2f} instruct={row['instruct']:.2f} "
              f"gap={row['gap']:.2f}")
        for name in ("base+full", "base+binary", "base+dare90"):
            gc = row[name + "_gap_closed"]
            gc_s = f"{100*gc:.1f}%" if gc is not None else "n/a"
            print(f"  {name}: {row[name]:.2f}  gap closed: {gc_s}")
    print(f"\nwrote signet_results.json in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
