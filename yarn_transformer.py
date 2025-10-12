#!/usr/bin/env python3
"""
CPU-friendly transformer variant for smoke testing.

This implements a simplified, robust subset of the large transformer code you provided:
- safe stubs for fp8/bf16 kernels
- RMSNorm implemented manually
- simplified MLA attention (no caching, no distributed ops)
- lightweight MLP and Block
- Transformer wrapper with precomputed rotary freqs

This file is intended for local smoke tests on CPU. It reduces defaults so a small forward
pass is cheap.
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

# --- Stubs for specialized kernels ---
def act_quant(x, block_size, scale_fmt=None):
    # very small behavioral stub: return x and a dummy scale
    return x, torch.ones(1, dtype=torch.float32)

def weight_dequant(weight, scale, *args, **kwargs):
    return weight.float()

def fp8_gemm(x, scale, weight, wscale):
    # fallback to float linear
    return F.linear(x.float(), weight.float())


world_size = 1
rank = 0
block_size = 128
gemm_impl = "bf16"


@dataclass
class ModelArgs:
    # small defaults for smoke test
    max_batch_size: int = 2
    max_seq_len: int = 128
    dtype: str = "bf16"
    scale_fmt: Optional[str] = None
    vocab_size: int = 4096
    dim: int = 64
    inter_dim: int = 256
    n_layers: int = 2
    n_dense_layers: int = 2
    n_heads: int = 4
    q_lora_rank: int = 0
    kv_lora_rank: int = 0
    qk_nope_head_dim: int = 16
    qk_rope_head_dim: int = 16
    v_head_dim: int = 16
    original_seq_len: int = 128
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.0


class ParallelEmbedding(nn.Module):
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        # for smoke test we just use a full embedding
        self.emb = nn.Embedding(vocab_size, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.emb(x)


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, dtype=None):
        super().__init__()
        self.lin = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)


class ColumnParallelLinear(Linear):
    def __init__(self, in_features, out_features, bias: bool = True, dtype=None):
        super().__init__(in_features, out_features, bias, dtype)


class RowParallelLinear(Linear):
    def __init__(self, in_features, out_features, bias: bool = True, dtype=None):
        super().__init__(in_features, out_features, bias, dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        # x: (B, T, D) or (B, D)
        if x.dim() == 3:
            norm = x.pow(2).mean(-1, keepdim=True).sqrt()
            return x / (norm + self.eps) * self.weight
        else:
            norm = x.pow(2).mean(-1, keepdim=True).sqrt()
            return x / (norm + self.eps) * self.weight


def precompute_freqs_cis(args: ModelArgs) -> torch.Tensor:
    dim = args.qk_rope_head_dim
    seqlen = args.max_seq_len
    base = args.rope_theta
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(seqlen, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    # complex numbers as two channels
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    freqs_cis = torch.stack([cos, sin], dim=-1)  # (T, dim/2, 2)
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    # x: (..., head_dim) where head_dim is even and equals 2 * (dim/2)
    orig_shape = x.shape
    last = x.size(-1)
    x = x.view(*x.shape[:-1], -1, 2)
    # freqs_cis shaped (T, dim/2, 2) or (1, T, 1, dim/2, 2)
    # broadcast appropriately
    freqs = freqs_cis
    while freqs.dim() < x.dim():
        freqs = freqs.unsqueeze(0)
    out = torch.zeros_like(x)
    # complex multiply: (a+ib)*(c+id) = (ac-bd) + i(ad+bc)
    a = x[..., 0]
    b = x[..., 1]
    c = freqs[..., 0]
    d = freqs[..., 1]
    out[..., 0] = a * c - b * d
    out[..., 1] = a * d + b * c
    return out.view(*orig_shape)


class MLA(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.head_dim = args.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.q_proj = Linear(self.dim, self.n_heads * self.head_dim)
        self.k_proj = Linear(self.dim, self.n_heads * self.head_dim)
        self.v_proj = Linear(self.dim, self.n_heads * args.v_head_dim)
        self.out = Linear(self.n_heads * args.v_head_dim, self.dim)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor] = None):
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(b, t, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(b, t, self.n_heads, -1)

        # apply rotary on the latter half of head_dim
        q_ro = q[..., -args.qk_rope_head_dim:]
        k_ro = k[..., -args.qk_rope_head_dim:]
        # freqs_cis: (T, dim/2, 2) -> expand
        q[..., -args.qk_rope_head_dim:] = apply_rotary_emb(q_ro, freqs_cis)
        k[..., -args.qk_rope_head_dim:] = apply_rotary_emb(k_ro, freqs_cis)

        # scaled dot-product
        scores = torch.einsum('bthd,bshd->bhts', q, k) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask.unsqueeze(0).unsqueeze(0)
        probs = torch.softmax(scores, dim=-1)
        context = torch.einsum('bhts,bshd->bthd', probs, v)
        context = context.reshape(b, t, -1)
        return self.out(context)


class MLP(nn.Module):
    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = Linear(dim, inter_dim)
        self.w2 = Linear(inter_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)))


class Block(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attn_norm = RMSNorm(args.dim)
        self.ffn_norm = RMSNorm(args.dim)
        self.attn = MLA(args)
        self.ffn = MLP(args.dim, args.inter_dim)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor] = None):
        x = x + self.attn(self.attn_norm(x), freqs_cis, mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed = ParallelEmbedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList([Block(args) for _ in range(args.n_layers)])
        self.norm = RMSNorm(args.dim)
        self.head = Linear(args.dim, args.vocab_size)
        self.register_buffer('freqs_cis', precompute_freqs_cis(args), persistent=False)

    def forward(self, tokens: torch.Tensor, start_pos: int = 0):
        b, seqlen = tokens.shape
        h = self.embed(tokens)
        freqs = self.freqs_cis[start_pos:start_pos+seqlen]
        mask = None
        for layer in self.layers:
            h = layer(h, freqs, mask)
        h = self.norm(h)[:, -1]
        logits = self.head(h)
        return logits


if __name__ == '__main__':
    # smoke test
    args = ModelArgs()
    torch.manual_seed(0)
    model = Transformer(args)
    x = torch.randint(0, args.vocab_size, (2, 16))
    logits = model(x)
    print('logits shape:', logits.shape)
