"""VRAM planning for an 8 GB card: count, measure, then decide batch size before buying time.

    static  = 16 B/param   (fp32 weights 4 + grads 4 + AdamW m, v 8)
    acts    = measured: bytes of the storages autograd saves (each counted once), under bf16 autocast
              on the CPU at T = 32, 64, 128, fitted to a + b·T + c·T²; b × batch × ctx. Flash attention
              on the GPU keeps attention memory linear in T, so the per-token slope b is the right model.
    loss    = one chunk of 4096 rows × vocab fp32 logits (+ grad) from the chunked loss
    cuda    = ~0.5 GiB CUDA context and allocator slack

Level 1: an estimate, typically within ±25 % of `torch.cuda.max_memory_allocated`. `train
--auto-batch` measures the real card and backs off on out-of-memory, so this only picks a start.
"""

from __future__ import annotations

from dataclasses import replace

import torch

from .model import PRESETS, TAULS, TAULSConfig

GIB = 1 << 30


def _saved_bytes(cfg: TAULSConfig, ctx: int) -> int:
    model = TAULS(replace(cfg, ctx=ctx))
    seen: set[tuple[int, int]] = set()
    total = [0]

    def pack(t: torch.Tensor):
        key = (t.untyped_storage().data_ptr(), t.untyped_storage().nbytes())
        if key not in seen:                       # count each storage once, however many ops save it
            seen.add(key)
            total[0] += key[1]
        return t

    x = torch.zeros(1, ctx, dtype=torch.long)
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            model(x, x, logits=False)
    return total[0]


def activation_bytes_per_token(cfg: TAULSConfig) -> float:
    """Slope b of saved bytes = a + b·T + c·T², fitted exactly at T = 32, 64, 128.

    a holds per-step constants (weights saved by reference, their bf16 autocast copies); c holds the
    CPU math kernel's attention probabilities, which flash attention never stores. b is what grows
    with every token of every sequence in the batch."""
    ts = (32, 64, 128)
    ys = [_saved_bytes(cfg, t) for t in ts]
    (t1, t2, t3), (y1, y2, y3) = ts, ys
    c = ((y3 - y2) / (t3 - t2) - (y2 - y1) / (t2 - t1)) / (t3 - t1)
    b = (y2 - y1) / (t2 - t1) - c * (t1 + t2)
    return max(0.0, b)


def plan(preset: str, overrides: dict, vocab: int, batch: int, vram_gib: float, grad_checkpoint: bool) -> str:
    cfg = PRESETS[preset].to_dict()
    cfg.update(overrides)
    if vocab:
        cfg["vocab_size"] = vocab
    cfg = TAULSConfig.from_dict(cfg)
    n = sum(p.numel() for p in _meta_model(cfg).parameters())
    static = 16 * n
    per_tok = activation_bytes_per_token(cfg)
    if grad_checkpoint:                         # only block inputs are kept; one block is live in backward
        per_tok = per_tok / cfg.n_layers + 2 * cfg.d_model * 2
    acts = per_tok * batch * cfg.ctx
    loss = 4096 * cfg.vocab_size * 4 * 2
    cuda = 0.5 * GIB
    total = static + acts + loss + cuda
    fits = total <= vram_gib * GIB
    best = batch
    while best > 1 and static + per_tok * best * cfg.ctx + loss + cuda > vram_gib * GIB:
        best //= 2
    rows = [
        f"preset {preset}: {n / 1e6:.2f}M params, ctx {cfg.ctx}, vocab {cfg.vocab_size}, "
        f"TA-ULS kfp={cfg.kfp} entropy={cfg.entropy_reg} two_level={cfg.two_level} recurrence={cfg.control_recurrence}",
        f"  weights+grads+AdamW  {static / GIB:6.2f} GiB",
        f"  activations          {acts / GIB:6.2f} GiB  ({per_tok / 1024:.1f} KiB/token × {batch}×{cfg.ctx})"
        + ("  [grad checkpoint]" if grad_checkpoint else ""),
        f"  loss chunk           {loss / GIB:6.2f} GiB",
        f"  CUDA context/slack   {cuda / GIB:6.2f} GiB",
        f"  total ≈ {total / GIB:.2f} GiB of {vram_gib:g} → {'fits' if fits else 'does NOT fit'}"
        + ("" if fits else f"; largest power-of-two micro-batch that fits ≈ {best}"),
        "  (Level 1 estimate; `train --auto-batch` measures the real card)",
    ]
    return "\n".join(rows)


def _meta_model(cfg: TAULSConfig) -> torch.nn.Module:
    """The model's parameter shapes without allocating or initialising them."""
    import torch.nn as nn
    from .model import Block, RMSNorm, CausalKFP
    m = nn.Module()
    with torch.device("meta"):
        m.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        m.blocks = nn.ModuleList(Block(cfg, first=i == 0) for i in range(cfg.n_layers))
        m.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        if cfg.final_kfp:
            m.final_kfp = CausalKFP(cfg.d_model, cfg.kfp_rank or max(8, cfg.d_model // 4), cfg.kfp_eps)
        if not cfg.tie_embeddings:
            m.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
    return m
