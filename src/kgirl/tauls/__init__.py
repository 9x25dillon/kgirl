"""TA-ULS v2: a causal Two-level Trans-Algorithmic Universal Learning System language model,
trained on your own repositories. See docs/tauls/README.md.

Modules: chaos (no-PRNG streams), tokenizer (byte-level BPE), data (corpus → shards, sampler),
model (TAULS), train (loop, checkpoints), plan (VRAM estimate), cli (`python -m kgirl.tauls`).
torch and numpy are imported by the modules that need them, not here.
"""

__version__ = "2.0.0"
