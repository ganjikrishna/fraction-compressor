"""Print norm weight stats per layer to check for corruption."""
import sys
import numpy as np

sys.path.insert(0, r"D:\gemma4-e2b")
import signet_gemma4_e2b as G

idx = G.load_index(r"D:\gemma4-e2b\it\model.safetensors")
P = G.PREFIX
for i in [0, 1, 2, 13, 14, 15, 34]:
    for n in ["input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
              "post_feedforward_layernorm", "post_per_layer_input_norm"]:
        k = f"{P}layers.{i}.{n}.weight"
        w = G.load_tensor(r"D:\gemma4-e2b\it\model.safetensors", idx, k)
        print(f"L{i} {n[:18]:18s} mean={w.mean():8.3f} std={w.std():7.3f} "
              f"max={w.max():8.2f} min={w.min():8.3f}")
    print()
