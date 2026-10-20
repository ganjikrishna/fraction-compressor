# NOTICE: attribution, licenses, and originality

## This project's own work

All code, documentation, evaluation items, and results in this repository
are original work by Vinay K Ganji Krishna, released under the MIT License
(see LICENSE). No third-party code is included in this repository.

## Methods and prior work

Several methods here are clean-room implementations written from the
algorithm descriptions in published papers. The ideas are cited; no code
was copied from any of these projects.

- LittleBit (arXiv:2506.13771) and LittleBit-2 (arXiv:2603.00042),
  Samsung AI. The official implementation is licensed CC BY-NC 4.0 and
  none of it is used here. Only the published method descriptions
  informed the implementations in compress.py and compress2.py.
- ITQ (iterative quantization): Gong and Lazebnik, CVPR 2011.
- DARE (drop and rescale for delta parameters): Yu et al.,
  arXiv:2311.03099. Reimplemented as a comparison arm in the Signet tests.
- BitDelta (1-bit delta quantization): Liu et al., NeurIPS 2024,
  arXiv:2402.10193. The Signet binary arm follows the same published
  idea (sign of the base-to-instruct delta with a per-tensor scale),
  reimplemented here.
- Colibri: only the hot/cold placement idea, adapted and honestly
  scoped for dense models. No Colibri code is used.

Naming a published method and reporting its published results for
comparison is standard research practice. If you reuse this repository's
implementations, the MIT License applies to this code, and the papers
above are the right citations for the underlying methods.

## Models

This repository distributes no model weights and no tokenizer files.
Experiments download models separately from Hugging Face, under each
model's own license:

- HuggingFaceTB/SmolLM-135M and SmolLM-135M-Instruct: Apache 2.0
- HuggingFaceTB/SmolLM2-135M, SmolLM2-360M, and their Instruct
  variants: Apache 2.0
- Qwen/Qwen2.5-0.5B and Qwen2.5-0.5B-Instruct: Apache 2.0
- Qwen/Qwen3-0.6B: Apache 2.0

The Gemma diagnostic scripts (diag_e2b.py, diag_layer.py, diag_norm.py,
signet_gemma4_e2b.py) operate on Gemma weights that the user downloads
under Google's Gemma Terms of Use. No Gemma weights are distributed
here.

## Evaluation items

The 24 Signet probe items (math, knowledge, instruction-following) were
written for this project. They are not copied from MMLU, GSM8K, or any
other benchmark dataset.

## Results data

The JSON result files and profile3.json are measurements produced by
this project's own code, run on the models listed above.
