"""TA-ULS v2: a causal, cacheable Two-level Trans-Algorithmic Universal Learning System LM.

The ideas of `kgirl/llm/tauls_transformer.py`, rebuilt so they can be trained as a language model:

    x ─┬─ RMSNorm ─ causal attention (RoPE, SDPA, GQA, KV cache) ─┐
       │                                     stress_t = ‖a_t‖ / ‖x_t‖ │
       +◀──────────────────────────────────────────────────────────┘
       ├─ RMSNorm ─ EntropyRegulator(stress) ─ TwoLevelControl(prev_control) ─ CausalKFP ─┐
       +◀──────────────────────────────────────────────────────────────────────────────┘
       └─▶ next block (its control output is the next block's prev_control)

What changed from v1 (each was a defect for an LM; see docs/tauls/README.md §2):
  * attention is causal; v1 attended to future tokens unless a caller passed a mask.
  * KFP fluctuation is the variance of each feature over positions ≤ t of the same sequence
    (cumulative sums), so it neither leaks the future nor couples samples in a batch; v1 used
    the variance over the whole batch and every position, kept in a buffer mutated by forward().
  * entropy regulation is per token; v1 reduced it to one scalar over batch × sequence.
  * prev_control is real: the meta controller of block l sees block l−1's control signal
    (v1 always passed zeros).
  * pre-norm RMSNorm, RoPE, fused SwiGLU paths, tied embeddings, SDPA (flash attention on GPU).
  * TA-ULS paths start as the identity (zero-initialised gains) and learn their effect, so
    training starts as a plain transformer and every component can be switched off: the
    `baseline` ablation is a standard pre-norm SwiGLU transformer of the same shape.

Hierarchical mode (h_cycles > 0) adds the nested recurrence of HRM, the Hierarchical Reasoning
Model (Wang et al. 2025, arXiv:2506.21734; adapted from 9x25dillon/Fractal_cascade_simulation,
models/hrm/hrm_act_v1.py, MIT), made causal and cacheable:

    z_H = z_L = 0
    repeat h_cycles:   repeat l_cycles:  z_L ← L-stack(norm_L(z_L + z_H + x))     fast, low level
                       z_H ← H-stack(norm_H(z_H + z_L))                           slow, high level
    logits = head(norm(z_H))

The first n_layers − h_layers blocks form the L-stack and the rest the H-stack; their weights are
shared across cycles. Only the last L and the last H application carry gradients (HRM's one-step
gradient), so activation memory is that of n_layers blocks whatever the cycle count, while compute
and effective depth grow with the cycles. Each (application, block) pair has its own KV/KFP cache
slot, so incremental decoding matches the full forward exactly.

Shapes: B batch, S new positions, T cached + new positions, D d_model, H heads, K kv heads.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields, replace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from . import chaos


# ---------------------------------------------------------------------------- config

@dataclass(frozen=True)
class TAULSConfig:
    vocab_size: int = 8192
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    n_kv_heads: int = 0               # 0 → n_heads (no grouped-query attention)
    ctx: int = 512
    control_mult: float = 2.0         # meta hidden = control_mult·D; auto hidden = half of it
    mlp_mult: float = 8 / 3           # hidden width when two_level is off (baseline SwiGLU MLP)
    kfp: bool = True                  # Kinetic Force Principle damping after each control unit
    kfp_rank: int = 0                 # rank of the force projection; 0 → D // 4
    final_kfp: bool = True            # global stability tracker before the LM head
    entropy_reg: bool = True          # per-token entropy regulation of the control input
    entropy_target: float = 0.8
    two_level: bool = True            # meta + automatic control paths; False → one SwiGLU MLP
    control_recurrence: bool = True   # meta controller sees the previous block's control output
    dropout: float = 0.0
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    kfp_eps: float = 1e-4
    tie_embeddings: bool = True
    init_std: float = 0.02
    init_seed: float = 0.4123
    h_cycles: int = 0                 # > 0 → hierarchical (HRM) recurrence; 0 → one flat stack
    l_cycles: int = 2                 # low-level applications per high-level update
    h_layers: int = 0                 # blocks in the H-stack (the rest form the L-stack)
    one_step_grad: bool = True        # gradients through the last L and H application only
    inject_norm: bool = False         # RMS-normalise injected sums (made embeddings scale-free: worse; docs §5)
    loss: str = "softmax"             # softmax | stablemax (HRM's numerically stable alternative)

    @property
    def kv_heads(self) -> int:
        return self.n_kv_heads or self.n_heads

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def validate(self) -> "TAULSConfig":
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.head_dim % 2:
            raise ValueError("head_dim must be even for RoPE")
        if self.h_cycles and not (0 < self.h_layers < self.n_layers):
            raise ValueError("hierarchical mode needs 0 < h_layers < n_layers")
        if self.h_cycles and self.l_cycles < 1:
            raise ValueError("l_cycles must be at least 1")
        if self.loss not in ("softmax", "stablemax"):
            raise ValueError("loss must be softmax or stablemax")
        return self

    @property
    def l_layers(self) -> int:
        return self.n_layers - self.h_layers if self.h_cycles else self.n_layers

    def schedule(self) -> list[tuple[str, int]]:
        """Stack applications in order, as (stack, application index): one flat 'F' unless hierarchical."""
        if not self.h_cycles:
            return [("F", 0)]
        out, li = [], 0
        for h in range(self.h_cycles):
            for _ in range(self.l_cycles):
                out.append(("L", li))
                li += 1
            out.append(("H", h))
        return out

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TAULSConfig":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names}).validate()


TAULS_OFF = dict(kfp=False, final_kfp=False, entropy_reg=False, two_level=False, control_recurrence=False)

PRESETS: dict[str, TAULSConfig] = {
    # unit tests and CPU smoke runs
    "tiny": TAULSConfig(d_model=64, n_layers=2, n_heads=4, ctx=64),
    # repo-only corpus (a few million tokens): ~12M parameters at an 8k vocabulary
    "small": TAULSConfig(d_model=384, n_layers=6, n_heads=6, ctx=512),
    # 8 GB VRAM desktop, bf16/fp16: ~60M parameters at a 16k vocabulary; wants ≥ 0.5B tokens
    "desk-8gb": TAULSConfig(vocab_size=16384, d_model=640, n_layers=10, n_heads=10, n_kv_heads=5, ctx=1024),
}


def _round(x: float, multiple: int = 64) -> int:
    return max(multiple, int(multiple * math.ceil(x / multiple)))


def block_params(cfg: TAULSConfig, first: bool = False) -> int:
    """Parameters of one block (a non-first block unless `first`), from the config alone."""
    d, hd = cfg.d_model, cfg.head_dim
    n = 2 * d + d * (cfg.n_heads + 2 * cfg.kv_heads) * hd + d * d          # norms, qkv, out
    if cfg.two_level:
        h = _round(cfg.control_mult * d)
        ha = _round(h / 2)
        n += 3 * d * h + 3 * d * ha + d
        n += d * d if cfg.control_recurrence and not first else 0
    else:
        n += 3 * d * _round(cfg.mlp_mult * d)
    if cfg.kfp:
        n += 2 * d * (cfg.kfp_rank or max(8, d // 4)) + d
    if cfg.entropy_reg:
        e = max(8, d // 8)
        n += d * e + e + e + 1 + 2 * d
    return n


def matched_baseline(cfg: TAULSConfig) -> TAULSConfig:
    """The plain-transformer ablation of `cfg`, its MLP widened to the same block parameter count."""
    target = block_params(cfg)
    base = replace(cfg, **TAULS_OFF)
    best = min((replace(base, mlp_mult=k / 64) for k in range(64, 64 * 12)),
               key=lambda c: abs(block_params(c) - target))
    return best


PRESETS["tiny-baseline"] = matched_baseline(PRESETS["tiny"])
PRESETS["tiny-hrm"] = replace(PRESETS["tiny-baseline"], n_layers=2, h_layers=1, h_cycles=2, l_cycles=2)
PRESETS["small-baseline"] = matched_baseline(PRESETS["small"])
PRESETS["desk-8gb-baseline"] = matched_baseline(PRESETS["desk-8gb"])
# HRM recurrence: L-stack 4 blocks × 2 cycles, H-stack 2 blocks; 4 + 2 blocks of weights and activation
# memory, 2·(2·4 + 2) = 20 block applications of depth per token. Same parameters as small-baseline − 0.
PRESETS["small-hrm"] = replace(PRESETS["small-baseline"], n_layers=6, h_layers=2, h_cycles=2, l_cycles=2)
PRESETS["small-hrm-tauls"] = replace(PRESETS["small"], n_layers=6, h_layers=2, h_cycles=2, l_cycles=2)


# ---------------------------------------------------------------------------- cache

@dataclass
class KFPState:
    """Running per-feature sums over the positions seen so far: (Σx, Σx², count)."""
    s1: torch.Tensor | None = None
    s2: torch.Tensor | None = None
    n: int = 0


@dataclass
class LayerCache:
    k: torch.Tensor | None = None
    v: torch.Tensor | None = None
    kfp: KFPState = field(default_factory=KFPState)


@dataclass
class Cache:
    layers: list[LayerCache]
    final: KFPState = field(default_factory=KFPState)
    pos: int = 0


# ---------------------------------------------------------------------------- components

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype) * self.weight


class Rotary(nn.Module):
    def __init__(self, head_dim: int, ctx: int, theta: float):
        super().__init__()
        inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim))
        ang = torch.outer(torch.arange(ctx, dtype=torch.float64), inv)
        self.register_buffer("cos", ang.cos().float(), persistent=False)
        self.register_buffer("sin", ang.sin().float(), persistent=False)

    def forward(self, x: torch.Tensor, pos0: int) -> torch.Tensor:
        """x: (B, heads, S, hd) → rotated, positions pos0 … pos0+S−1."""
        S = x.shape[-2]
        if pos0 + S > self.cos.shape[0]:
            raise ValueError(f"position {pos0 + S} exceeds the model context {self.cos.shape[0]}")
        cos = self.cos[pos0:pos0 + S].to(x.dtype)
        sin = self.sin[pos0:pos0 + S].to(x.dtype)
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, cfg: TAULSConfig):
        super().__init__()
        self.h, self.kv, self.hd = cfg.n_heads, cfg.kv_heads, cfg.head_dim
        self.qkv = nn.Linear(cfg.d_model, (self.h + 2 * self.kv) * self.hd, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor, rope: Rotary, pos0: int, cache: LayerCache | None) -> torch.Tensor:
        B, S, D = x.shape
        q, k, v = self.qkv(x).split([self.h * self.hd, self.kv * self.hd, self.kv * self.hd], dim=-1)
        q = rope(q.view(B, S, self.h, self.hd).transpose(1, 2), pos0)
        k = rope(k.view(B, S, self.kv, self.hd).transpose(1, 2), pos0)
        v = v.view(B, S, self.kv, self.hd).transpose(1, 2)
        past = 0
        if cache is not None:
            if cache.k is not None:
                past = cache.k.shape[2]
                k = torch.cat([cache.k, k], dim=2)
                v = torch.cat([cache.v, v], dim=2)
            cache.k, cache.v = k, v
        if self.kv != self.h:
            rep = self.h // self.kv
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        drop = self.dropout if self.training else 0.0
        if past == 0:
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=True)
        elif S == 1:
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=drop)
        else:                                            # chunked prefill on top of a cache
            T = past + S
            mask = torch.ones(S, T, dtype=torch.bool, device=x.device).tril(diagonal=past)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=drop)
        return self.out(y.transpose(1, 2).reshape(B, S, D))


class CausalKFP(nn.Module):
    """Kinetic Force Principle damping, causal.

    For feature d at position t, the fluctuation intensity is the variance of that feature over
    positions 1…t of the same sequence. Features that fluctuate more are damped harder (the gains
    are a stop-gradient control signal, as v1's no-grad fluctuation buffer was):

        g_t = (var_t + ε)⁻¹ / mean_d (var_t + ε)⁻¹            (relative gain, mean 1 over features)
        y_t = x_t − w ⊙ g_t ⊙ U V x_t                         (low-rank force projection, w zero-init)
    """

    def __init__(self, d: int, rank: int, eps: float):
        super().__init__()
        self.eps = eps
        self.down = nn.Linear(d, rank, bias=False)
        self.up = nn.Linear(rank, d, bias=False)
        self.weight = nn.Parameter(torch.zeros(d))
        self.last_fluctuation: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, state: KFPState | None = None, collect: bool = False) -> torch.Tensor:
        B, S, D = x.shape
        with torch.no_grad():                    # fluctuation is a control signal, not a path for gradients
            xf = x.detach().float()              # (v1 kept it in a no-grad buffer); saves the fp32 graph
            s1 = xf.cumsum(1)
            s2 = (xf * xf).cumsum(1)
            n0 = 0
            if state is not None:
                if state.s1 is not None:
                    s1 = s1 + state.s1[:, None]
                    s2 = s2 + state.s2[:, None]
                n0 = state.n
                state.s1, state.s2, state.n = s1[:, -1], s2[:, -1], n0 + S
            n = torch.arange(n0 + 1, n0 + S + 1, device=x.device, dtype=torch.float32).view(1, S, 1)
            mean = s1 / n
            var = (s2 / n - mean * mean).clamp_min(0.0)
            gains = 1.0 / (var + self.eps)
            gains = (gains / gains.mean(-1, keepdim=True)).to(x.dtype)
            if collect:
                self.last_fluctuation = var.mean()
        return x - self.weight * gains * self.up(self.down(x))


class EntropyRegulator(nn.Module):
    """Per-token entropy regulation: s_t = 2·σ(W·σ(ê_t − target + stress_t) + b), ê_t = σ(MLP(x_t)).

    W and b start at zero, so s_t ≡ 1 (identity) until training finds a use for it.
    """

    def __init__(self, d: int, target: float):
        super().__init__()
        self.target = target
        self.est_in = nn.Linear(d, max(8, d // 8))
        self.est_out = nn.Linear(max(8, d // 8), 1)
        self.ctrl = nn.Linear(1, d)
        self.last: dict[str, torch.Tensor] = {}

    def forward(self, x: torch.Tensor, stress: torch.Tensor, collect: bool = False) -> torch.Tensor:
        ent = torch.sigmoid(self.est_out(F.gelu(self.est_in(x))))           # (B,S,1)
        intensity = torch.sigmoid(ent - self.target + stress.to(ent.dtype))  # (B,S,1)
        scale = 2.0 * torch.sigmoid(self.ctrl(intensity))                   # (B,S,D), 1 at init
        if collect:
            self.last = {"entropy": ent.mean().detach(), "stress": stress.float().mean().detach(),
                         "scale_dev": (scale - 1).abs().mean().detach()}
        return x * scale


class SwiGLU(nn.Module):
    def __init__(self, d_in: int, hidden: int, d_out: int):
        super().__init__()
        self.gate_up = nn.Linear(d_in, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, d_out, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g, u = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(g) * u)


class TwoLevelControl(nn.Module):
    """Meta control (learning level) mixed with automatic control (reflex level):

        y = α ⊙ Meta(x + R·c_prev) + (1 − α) ⊙ Auto(x),   α = σ(m)

    c_prev is the previous block's control output, R (D×D) and m start at zero (α = ½)."""

    def __init__(self, cfg: TAULSConfig, first: bool = False):
        super().__init__()
        d = cfg.d_model
        self.two_level = cfg.two_level
        if not cfg.two_level:
            self.mlp = SwiGLU(d, _round(cfg.mlp_mult * d), d)
            return
        h = _round(cfg.control_mult * d)
        self.meta = SwiGLU(d, h, d)
        self.auto = SwiGLU(d, _round(h / 2), d)
        self.recur = nn.Linear(d, d, bias=False) if cfg.control_recurrence and not first else None
        self.mix = nn.Parameter(torch.zeros(d))
        self.last_alpha: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, prev: torch.Tensor | None, collect: bool = False) -> torch.Tensor:
        if not self.two_level:
            return self.mlp(x)
        xm = x + self.recur(prev) if (self.recur is not None and prev is not None) else x
        alpha = torch.sigmoid(self.mix)
        if collect:
            self.last_alpha = alpha.mean().detach()
        return alpha * self.meta(xm) + (1 - alpha) * self.auto(x)


class Block(nn.Module):
    def __init__(self, cfg: TAULSConfig, first: bool = False):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.regulator = EntropyRegulator(cfg.d_model, cfg.entropy_target) if cfg.entropy_reg else None
        self.control = TwoLevelControl(cfg, first)
        self.kfp = CausalKFP(cfg.d_model, cfg.kfp_rank or max(8, cfg.d_model // 4), cfg.kfp_eps) if cfg.kfp else None
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, prev: torch.Tensor | None, rope: Rotary, pos0: int,
                cache: LayerCache | None, collect: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        a = self.drop(self.attn(self.norm1(x), rope, pos0, cache))
        u = self.norm2(x + a)
        if self.regulator is not None:
            stress = a.float().norm(dim=-1, keepdim=True) / (x.float().norm(dim=-1, keepdim=True) + 1e-6)
            u = self.regulator(u, stress, collect)
        x = x + a
        c = self.control(u, prev, collect)
        if self.kfp is not None:
            c = self.kfp(c, cache.kfp if cache is not None else None, collect)
        return x + self.drop(c), c


# ---------------------------------------------------------------------------- model

@dataclass
class Output:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    metrics: dict[str, float] | None = None


class TAULS(nn.Module):
    def __init__(self, cfg: TAULSConfig):
        super().__init__()
        self.cfg = cfg.validate()
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.rope = Rotary(cfg.head_dim, cfg.ctx, cfg.rope_theta)
        self.blocks = nn.ModuleList(Block(cfg, first=i in (0, cfg.l_layers)) for i in range(cfg.n_layers))
        self.norm_l = RMSNorm(cfg.d_model, cfg.norm_eps) if cfg.h_cycles and cfg.inject_norm else nn.Identity()
        self.norm_h = RMSNorm(cfg.d_model, cfg.norm_eps) if cfg.h_cycles and cfg.inject_norm else nn.Identity()
        self._slots = self._slot_plan()
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.final_kfp = CausalKFP(cfg.d_model, cfg.kfp_rank or max(8, cfg.d_model // 4), cfg.kfp_eps) \
            if cfg.final_kfp else None
        self.head = None if cfg.tie_embeddings else nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.grad_checkpoint = False
        chaos_init_(self)

    # -------------------------------------------------------------- bookkeeping

    def num_params(self, exclude_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        return n - self.embed.weight.numel() if exclude_embedding else n

    def _slot_plan(self) -> list[tuple[str, int, list[int], list[int]]]:
        """(stack, app, block indices, cache slots) for each stack application in schedule order."""
        L = list(range(self.cfg.l_layers))
        Hs = list(range(self.cfg.l_layers, self.cfg.n_layers))
        plan, slot = [], 0
        for stack, app in self.cfg.schedule():
            idx = Hs if stack == "H" else L
            plan.append((stack, app, idx, list(range(slot, slot + len(idx)))))
            slot += len(idx)
        return plan

    def new_cache(self) -> Cache:
        return Cache([LayerCache() for _ in range(sum(len(p[3]) for p in self._slots))])

    def flops_per_token(self) -> float:
        """Forward+backward FLOPs per token: 6·N plus attention (PaLM appendix B estimate)."""
        c = self.cfg
        apps = sum(len(p[2]) for p in self._slots)          # block applications per token
        per_block = (self.num_params(exclude_embedding=True)) / max(1, c.n_layers)
        return 6 * per_block * apps + 12 * apps * c.d_model * c.ctx

    # -------------------------------------------------------------- forward

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None, cache: Cache | None = None,
                collect: bool = False, logits: bool = True) -> Output:
        """logits=False with targets: the loss is computed in row chunks whose logits are recomputed in
        backward (activation checkpointing), so B·S·V logits never exist at once. Training uses it."""
        x = self.features(idx, cache, collect)
        if not logits and targets is not None:
            return Output(None, self._chunked_loss(x, targets), self.metrics() if collect else None)
        out = self._logits(x)
        loss = None
        if targets is not None:
            loss = self._nll(out.reshape(-1, out.shape[-1]), targets.reshape(-1), "mean")
        return Output(out, loss, self.metrics() if collect else None)

    def features(self, idx: torch.Tensor, cache: Cache | None = None, collect: bool = False) -> torch.Tensor:
        """Final hidden states (B, S, D): what the LM head reads. KGIRL memory keys on these."""
        S = idx.shape[1]
        pos0 = cache.pos if cache is not None else 0
        x = self.embed(idx)
        if self.cfg.h_cycles:
            x = self._hierarchical(x, pos0, cache, collect)
        else:
            x = self._stack(x, self._slots[0], pos0, cache, collect)
        x = self.norm(x)
        if self.final_kfp is not None:
            x = self.final_kfp(x, cache.final if cache is not None else None, collect)
        if cache is not None:
            cache.pos += S
        return x

    def logits_of(self, x: torch.Tensor) -> torch.Tensor:
        return self._logits(x)

    def _nll(self, logits: torch.Tensor, t: torch.Tensor, reduction: str) -> torch.Tensor:
        if self.cfg.loss == "softmax":
            return F.cross_entropy(logits.float(), t, ignore_index=-1, reduction=reduction)
        return F.nll_loss(log_stablemax(logits.float()), t, ignore_index=-1, reduction=reduction)

    def _stack(self, h: torch.Tensor, plan, pos0: int, cache: Cache | None, collect: bool) -> torch.Tensor:
        _, _, blocks, slots = plan
        prev = None
        ckpt = self.grad_checkpoint and self.training and cache is None and torch.is_grad_enabled()
        for b, slot in zip(blocks, slots):
            block = self.blocks[b]
            lc = cache.layers[slot] if cache is not None else None
            if ckpt:
                h, prev = checkpoint(block, h, prev, self.rope, pos0, None, False, use_reentrant=False)
            else:
                h, prev = block(h, prev, self.rope, pos0, lc, collect)
            if not self.cfg.control_recurrence:
                prev = None
        return h

    def _hierarchical(self, x: torch.Tensor, pos0: int, cache: Cache | None, collect: bool) -> torch.Tensor:
        z_h = torch.zeros_like(x)
        z_l = torch.zeros_like(x)
        last_l = max(i for i, p in enumerate(self._slots) if p[0] == "L")
        last_h = len(self._slots) - 1
        one_step = self.cfg.one_step_grad and torch.is_grad_enabled()
        for i, plan in enumerate(self._slots):
            graded = not one_step or i in (last_l, last_h)
            with torch.set_grad_enabled(graded and torch.is_grad_enabled()):
                if plan[0] == "L":
                    z_l = self._stack(self.norm_l(z_l + z_h + x), plan, pos0, cache, collect)
                else:
                    z_h = self._stack(self.norm_h(z_h + z_l), plan, pos0, cache, collect)
        return z_h

    def _logits(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.embed.weight) if self.head is None else self.head(x)

    def _ce_sum(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self._nll(self._logits(x), t, "sum")

    def _chunked_loss(self, x: torch.Tensor, targets: torch.Tensor, rows: int = 4096) -> torch.Tensor:
        xf = x.reshape(-1, x.shape[-1])
        tf = targets.reshape(-1)
        total = x.new_zeros((), dtype=torch.float32)
        for i in range(0, xf.shape[0], rows):
            if torch.is_grad_enabled():
                total = total + checkpoint(self._ce_sum, xf[i:i + rows], tf[i:i + rows],
                                                                  use_reentrant=False)
            else:
                total = total + self._ce_sum(xf[i:i + rows], tf[i:i + rows])
        return total / (tf != -1).sum().clamp_min(1)

    def metrics(self) -> dict[str, float]:
        """TA-ULS state of the last forward run with collect=True (layer means, as floats)."""
        out: dict[str, list[float]] = {}

        def put(k, v):
            if v is not None:
                out.setdefault(k, []).append(float(v))
        for b in self.blocks:
            if b.kfp is not None:
                put("kfp_fluct", b.kfp.last_fluctuation)
                put("kfp_w", b.kfp.weight.detach().abs().mean())
            if b.regulator is not None:
                for k, v in b.regulator.last.items():
                    put(f"reg_{k}", v)
            if b.control.two_level:
                put("alpha", b.control.last_alpha)
        if self.final_kfp is not None:
            put("final_kfp_fluct", self.final_kfp.last_fluctuation)
        return {k: sum(v) / len(v) for k, v in out.items()}

    # -------------------------------------------------------------- generation

    @torch.no_grad()
    def generate(self, prompt: list[int], max_new: int, temperature: float = 0.8, top_k: int = 50,
                 top_p: float = 0.95, stop: set[int] | None = None, sampler: chaos.Sampler | None = None,
                 prefill_chunk: int = 256) -> list[int]:
        """Sample with a KV cache. Randomness comes from a chaos stream; temperature 0 is greedy."""
        was = self.training
        self.eval()
        sampler = sampler or chaos.Sampler(self.cfg.init_seed, chaos.stream_id("generate"))
        device = self.embed.weight.device
        keep = max(1, self.cfg.ctx - max_new)
        prompt = prompt[-keep:] or [0]
        cache = self.new_cache()
        logits = None
        for i in range(0, len(prompt), prefill_chunk):
            chunk = torch.tensor([prompt[i:i + prefill_chunk]], device=device)
            logits = self(chunk, cache=cache).logits[0, -1]
        out: list[int] = []
        for _ in range(max_new):
            if cache.pos >= self.cfg.ctx:
                break
            tok = sample_token(logits.float(), temperature, top_k, top_p, float(sampler.uniform(1)[0]), self.cfg.loss)
            out.append(tok)
            if stop and tok in stop:
                break
            logits = self(torch.tensor([[tok]], device=device), cache=cache).logits[0, -1]
        self.train(was)
        return out


def stablemax_s(x: torch.Tensor) -> torch.Tensor:
    """s(x) = x + 1 for x ≥ 0, 1 / (1 − x) for x < 0: positive, monotone, linear growth (no exp overflow)."""
    return torch.where(x < 0, 1.0 / (1.0 - x.clamp_max(0)), x.clamp_min(0) + 1.0)


def log_stablemax(x: torch.Tensor) -> torch.Tensor:
    """log of s(x_i) / Σ_j s(x_j) (Prieto et al. 2025, used by HRM): a proper distribution over the vocab."""
    s = stablemax_s(x)
    return torch.log(s) - torch.log(s.sum(-1, keepdim=True))


def sample_token(logits: torch.Tensor, temperature: float, top_k: int, top_p: float, u: float,
                 loss: str = "softmax") -> int:
    if temperature <= 0:
        return int(logits.argmax())
    probs = torch.softmax(logits / temperature, dim=-1) if loss == "softmax" else \
        torch.exp(log_stablemax(logits / temperature))
    p, ix = probs.sort(descending=True)
    if top_k > 0:
        p, ix = p[:top_k], ix[:top_k]
    if top_p < 1.0:
        keep = (p.cumsum(0) - p) < top_p
        p, ix = p[keep], ix[keep]
    cdf = (p / p.sum()).cumsum(0)
    j = int(torch.searchsorted(cdf, torch.tensor([u], dtype=cdf.dtype, device=cdf.device)).clamp_max(len(ix) - 1))
    return int(ix[j])


# ---------------------------------------------------------------------------- init

_ZERO = ("kfp.weight", "final_kfp.weight", "regulator.ctrl.weight", "regulator.ctrl.bias",
         "control.mix", "control.recur.weight")
_RESIDUAL_OUT = ("attn.out.weight", ".down.weight")


@torch.no_grad()
def chaos_init_(model: TAULS) -> None:
    """Initialise every parameter from a chaos stream keyed by its name (order-independent).

    Matrices ~ N(0, init_std²), residual output projections scaled by 1/√(2L) (GPT-2), RMSNorms = 1,
    biases = 0, and the TA-ULS gains (KFP weight, regulator control, mixer, recurrence) = 0 so each
    block starts as the plain transformer it extends.
    """
    cfg = model.cfg
    for name, p in model.named_parameters():
        if name.endswith(_ZERO) or name.endswith(".bias"):
            p.zero_()
        else:
            std = cfg.init_std
            if name.endswith(_RESIDUAL_OUT) and "kfp" not in name:
                std /= math.sqrt(2 * cfg.n_layers)
            z = chaos.normal(p.numel(), cfg.init_seed, chaos.stream_id("init", name))
            p.copy_(torch.from_numpy(z).view_as(p).to(p.dtype) * std)
    for m in model.modules():
        if isinstance(m, RMSNorm):
            m.weight.fill_(1.0)
