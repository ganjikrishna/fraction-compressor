"""Diagnostic for Gemma4 E2B: tokenizer roundtrip + weight stats + forward stats."""
import sys
import numpy as np

print("=== 1. tokenizer roundtrip ===")
from tokenizers import Tokenizer
tok = Tokenizer.from_file(r"D:\gemma4-e2b\it\tokenizer.json")
print("vocab size:", tok.get_vocab_size())
print("bos id:", tok.token_to_id("<bos>"))
for s in ["<|turn>", "<turn|>", "<bos>", "Hello world",
          "<|turn>user\nThe capital of France is<turn|>\n<|turn>model\n"]:
    ids = tok.encode(s).ids
    print(repr(s[:40]), "->", len(ids), "tokens", ids[:8], "->", repr(tok.decode(ids)[:60]))

print("\n=== 2. weight stats (bf16 conversion check) ===")
sys.path.insert(0, r"D:\gemma4-e2b")
import signet_gemma4_e2b as G
idx = G.load_index(r"D:\gemma4-e2b\it\model.safetensors")
for k in ["model.language_model.embed_tokens.weight",
          "model.language_model.layers.0.self_attn.q_proj.weight",
          "model.language_model.layers.0.input_layernorm.weight",
          "model.language_model.norm.weight"]:
    w = G.load_tensor(r"D:\gemma4-e2b\it\model.safetensors", idx, k)
    print(k.split(".")[-2], k.split(".")[-1], w.dtype, w.shape,
          "mean=%.4f std=%.4f max=%.4f min=%.4f finite=%s"
          % (w.mean(), w.std(), w.max(), w.min(), np.all(np.isfinite(w))))

print("\n=== 3. forward stats ===")
keys = sorted(set(idx))
sd = G.load_full(r"D:\gemma4-e2b\it\model.safetensors", idx, keys)
m = G.Gemma4E2B(sd)
ids = G.encode_prompt(tok, "The capital of France is")
print("prompt ids:", len(ids), ids[:12])
logits = m.forward(ids)
print("logits shape:", logits.shape, "finite:", np.all(np.isfinite(logits)))
print("logits[−1] top5:", np.sort(logits[-1])[-5:])
row = logits[-1].astype(np.float64)
row -= row.max()
e = np.exp(row)
p = e / e.sum()
top5 = np.argsort(p)[-5:][::-1]
print("top5 tokens:", [(int(t), tok.decode([int(t)]), round(float(p[t]), 4)) for t in top5])
print("entropy:", round(float(-(p * np.log(p + 1e-12)).sum()), 2))
