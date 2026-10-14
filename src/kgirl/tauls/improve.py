"""KGIRL recursive self-improvement: memory from the model, distilled back into the model, coarse → fine.

    round ℓ = 1 … R   (memory depth grows with ℓ: coarse cells first, finer cells later)
      1. collect   z_t = P·h_t of the current model on train tokens [0, X)            (the database)
      2. build     HoloMemory(depth = ℓ) with MDL-gated splits                         (holo.py)
      3. fit gate  (η, π, T) on validation half A                                      (free energy)
      4. distil    on train windows from [X, end), disjoint from the database:
                   loss = CE(y) + α · KL(teacher ‖ model),  teacher = (1 − w) p_LM + w p_mem
      5. verify    plain-model loss on validation half B; keep the round only if it improved,
                   else restore the weights (a rejected round changes nothing)

Self-improvement is a measured claim here. Step 5 is a monotone acceptance rule, so the accepted model
never gets worse on half B. Whether rounds are accepted at all is an empirical outcome, reported per round
in telemetry.json (L2 until measured on a real run).

Randomness: window offsets and position subsets come from chaos streams (kgirl.tauls.chaos).
"""

from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

from . import chaos
from .data import TokenData
from .holo import SETTINGS, HoloMemory, MemoryConfig, entropy, fit_gate, gate, mixed_nll
from .model import TAULS


@dataclass
class ImproveConfig:
    rounds: int = 3
    depths: list = field(default_factory=lambda: [1, 2, 3, 4])   # memory depth per round (coarse → fine)
    memory_tokens: int = 262144          # train tokens [0, X) collected into the database
    eval_tokens: int = 65536             # validation tokens (split into halves A and B)
    distil_steps: int = 200
    batch: int = 8
    distil_positions: int = 512          # positions per batch that get the KL term (p_mem is dense in V)
    alpha: float = 0.5
    lr: float = 1e-4
    min_gain: float = 1e-4               # nats/token on half B needed to accept a round
    seed: float = 0.4123
    memory: dict = field(default_factory=dict)   # MemoryConfig overrides

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------- collection and scoring

@torch.no_grad()
def collect(model: TAULS, mem: HoloMemory, data: TokenData, split: str, max_tokens: int, batch: int,
            device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    zs, ys = [], []
    for x, y in data.sequential(split, batch, model.cfg.ctx, max_tokens):
        h = model.features(torch.from_numpy(x).to(device)).float().cpu().numpy()
        zs.append(mem.project(h))
        ys.append(y.reshape(-1))
    return np.concatenate(zs), np.concatenate(ys).astype(np.int64)


@torch.no_grad()
def token_stats(model: TAULS, mem: HoloMemory, data: TokenData, split: str, max_tokens: int, batch: int,
                device: torch.device, settings=SETTINGS) -> dict[tuple, dict[str, np.ndarray]]:
    """Per validation token: p_LM(y), H_LM and, for each memory setting (β, η, κ, τ), p_mem(y), H_mem."""
    model.eval()
    betas = tuple(sorted({s[0] for s in settings}))
    taus = tuple(sorted({s[3] for s in settings if s[2] > 0}))
    lm_t, lm_h = [], []
    mem_t = {s: [] for s in settings}
    mem_h = {s: [] for s in settings}
    for x, y in data.sequential(split, batch, model.cfg.ctx, max_tokens):
        h = model.features(torch.from_numpy(x).to(device))
        p = torch.softmax(model.logits_of(h).float(), -1).reshape(-1, mem.vocab).cpu().numpy()
        yy = y.reshape(-1)
        rows = np.arange(len(yy))
        lm_t.append(p[rows, yy])
        lm_h.append(entropy(p))
        comp = mem.components(mem.project(h.float().cpu().numpy()), betas=betas, taus=taus)
        for st in settings:
            pm = mem.mix(comp, *st)
            mem_t[st].append(pm[rows, yy])
            mem_h[st].append(entropy(pm))
    base = {"p_lm_t": np.concatenate(lm_t), "h_lm": np.concatenate(lm_h)}
    return {st: {**base, "p_mem_t": np.concatenate(mem_t[st]), "h_mem": np.concatenate(mem_h[st])} for st in settings}


def memory_kwargs(g: dict) -> dict:
    """The fitted memory setting as HoloMemory.distribution keyword arguments."""
    return {"beta": g.get("beta"), "eta": g.get("eta", 0.0), "knn_eta": g.get("knn_eta", 0.0), "tau": g.get("tau", 0.05)}


def score(stats: dict[float, dict[str, np.ndarray]]) -> dict[str, Any]:
    """Fit the gate on half A, report on half B (contiguous halves of the validation stream)."""
    n = len(next(iter(stats.values()))["p_lm_t"])
    a, b = np.arange(n) < n // 2, np.arange(n) >= n // 2
    g = fit_gate(stats, a)
    s = stats[(g["beta"], g["eta"], g["knn_eta"], g["tau"])]
    w = gate(s["h_lm"][b], s["h_mem"][b], g["prior"], g["temp"])
    lm = float(-np.log(np.clip(s["p_lm_t"][b], 1e-30, None)).mean())
    mem_only = float(-np.log(np.clip(s["p_mem_t"][b], 1e-30, None)).mean())
    mix = float(mixed_nll(s["p_lm_t"][b], s["p_mem_t"][b], w).mean())
    # memory-only NLL of every setting on half B: which ingredients carry information
    per_setting = {f"beta={st[0]:g} eta={st[1]:g} knn={st[2]:g} tau={st[3]:g}":
                   float(-np.log(np.clip(v["p_mem_t"][b], 1e-30, None)).mean()) for st, v in stats.items()}
    return {"tokens_fit": int(a.sum()), "tokens_report": int(b.sum()),
            "gate": {k: g[k] for k in ("beta", "eta", "knn_eta", "tau", "prior", "temp")},
            "memory_only_by_setting": per_setting,
            "nll_lm": lm, "nll_memory_only": mem_only, "nll_mixed": mix,
            "bits_per_token_gain": (lm - mix) / math.log(2), "mean_memory_weight": float(w.mean())}


@torch.no_grad()
def plain_loss(model: TAULS, data: TokenData, split: str, max_tokens: int, batch: int, device: torch.device,
               half: str = "B") -> float:
    """Mean NLL of the plain model on half A or B of the first max_tokens of `split`."""
    model.eval()
    nll = []
    for x, y in data.sequential(split, batch, model.cfg.ctx, max_tokens):
        out = model(torch.from_numpy(x).to(device), torch.from_numpy(y).to(device))
        logp = torch.log_softmax(out.logits.float(), -1).reshape(-1, out.logits.shape[-1])
        nll.append((-logp[torch.arange(logp.shape[0]), torch.from_numpy(y.reshape(-1)).to(device)]).cpu().numpy())
    v = np.concatenate(nll)
    return float(v[len(v) // 2:].mean() if half == "B" else v[:len(v) // 2].mean())


# ---------------------------------------------------------------------- distillation

def _windows(data: TokenData, lo: int, ctx: int, batch: int, step: int, seed: float):
    arr = data.split["train"]
    hi = len(arr) - ctx - 1
    if hi <= lo:
        raise ValueError(f"train split too short for distillation windows after the first {lo} tokens")
    u = chaos.uniform(batch, seed, chaos.stream_id("kgirl", "distil", step))
    offs = lo + (u * (hi - lo)).astype(np.int64)
    x = np.stack([np.asarray(arr[o:o + ctx], np.int64) for o in offs])
    y = np.stack([np.asarray(arr[o + 1:o + ctx + 1], np.int64) for o in offs])
    return x, y


def distil(model: TAULS, mem: HoloMemory, g: dict, data: TokenData, ic: ImproveConfig, lo: int,
           device: torch.device, round_: int, log: Callable = print) -> dict[str, float]:
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=ic.lr, betas=(0.9, 0.95), weight_decay=0.0)
    ce_sum = kl_sum = 0.0
    for step in range(ic.distil_steps):
        x, y = _windows(data, lo, model.cfg.ctx, ic.batch, round_ * 1_000_003 + step, ic.seed)
        xt, yt = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
        h = model.features(xt)
        logits = model.logits_of(h).float()
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), yt.reshape(-1))
        flat_h = h.reshape(-1, h.shape[-1])
        k = min(ic.distil_positions, flat_h.shape[0])
        pos = np.argsort(chaos.uniform(flat_h.shape[0], ic.seed, chaos.stream_id("kgirl", "pos", round_, step)))[:k]
        pos_t = torch.from_numpy(pos).to(device)
        with torch.no_grad():
            lp = logits.reshape(-1, logits.shape[-1])[pos_t]
            p_lm = torch.softmax(lp, -1).cpu().numpy()
            p_mem, _ = mem.distribution(mem.project(flat_h[pos_t].float().cpu().numpy()), **memory_kwargs(g))
            w = gate(entropy(p_lm), entropy(p_mem), g["prior"], g["temp"])[:, None]
            teacher = torch.from_numpy(((1 - w) * p_lm + w * p_mem).astype(np.float32)).to(device)
        logq = torch.log_softmax(logits.reshape(-1, logits.shape[-1])[pos_t], -1)
        kl = (teacher * (torch.log(teacher.clamp_min(1e-30)) - logq)).sum(-1).mean()
        loss = ce + ic.alpha * kl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ce_sum += float(ce)
        kl_sum += float(kl)
        if (step + 1) % max(1, ic.distil_steps // 5) == 0:
            log(f"  distil {step + 1}/{ic.distil_steps} ce {ce_sum / (step + 1):.4f} kl {kl_sum / (step + 1):.4f}")
    n = max(1, ic.distil_steps)
    return {"ce": ce_sum / n, "kl": kl_sum / n}


# ---------------------------------------------------------------------- the loop

def improve(model: TAULS, data: TokenData, ic: ImproveConfig, out: str | Path | None, device: torch.device,
            log: Callable = print, ckpt: dict | None = None) -> dict[str, Any]:
    """Run the loop; with `out`, writes telemetry.json, memory_round*.npz and improved.pt (a checkpoint
    that `python -m kgirl.tauls eval|generate|info` load like any other)."""
    out = Path(out) if out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    base = plain_loss(model, data, "val", ic.eval_tokens, ic.batch, device)
    record: dict[str, Any] = {"config": ic.to_dict(), "baseline_val_B_nll": base, "rounds": []}
    log(f"baseline: plain-model NLL on validation half B = {base:.4f}")
    best = base
    for r in range(ic.rounds):
        depth = ic.depths[min(r, len(ic.depths) - 1)]
        t0 = time.perf_counter()
        mcfg = MemoryConfig(**{**ic.memory, "max_depth": depth})
        mem = HoloMemory(model.cfg.d_model, model.cfg.vocab_size, mcfg)
        z, y = collect(model, mem, data, "train", ic.memory_tokens, ic.batch, device)
        mem.build(z, y)
        sc = score(token_stats(model, mem, data, "val", ic.eval_tokens, ic.batch, device))
        log(f"round {r + 1}: depth {depth}, {mem.summary()['cells']} cells; LM {sc['nll_lm']:.4f} → "
            f"mixed {sc['nll_mixed']:.4f} nats ({sc['bits_per_token_gain']:+.4f} bits/token), gate {sc['gate']}")
        saved = copy.deepcopy(model.state_dict())
        d = distil(model, mem, sc["gate"], data, ic, len(y), device, r, log)
        after = plain_loss(model, data, "val", ic.eval_tokens, ic.batch, device)
        accepted = after < best - ic.min_gain
        if accepted:
            best = after
        else:
            model.load_state_dict(saved)
        rec = {"round": r + 1, "depth": depth, "memory": mem.summary(), "memory_score": sc, "distil": d,
               "val_B_before": record["rounds"][-1]["val_B_kept"] if record["rounds"] else base,
               "val_B_after": after, "accepted": accepted, "val_B_kept": best,
               "seconds": round(time.perf_counter() - t0, 2)}
        record["rounds"].append(rec)
        log(f"round {r + 1}: plain NLL {rec['val_B_before']:.4f} → {after:.4f}: {'ACCEPTED' if accepted else 'rejected, weights restored'}")
        if out:
            mem.save(out / f"memory_round{r + 1}.npz")
    record["final_val_B_nll"] = best
    record["total_gain_nats"] = base - best
    if out:
        (out / "telemetry.json").write_text(json.dumps(record, indent=2, default=float))
        meta = {k: v for k, v in (ckpt or {}).items() if k not in ("model", "optimizer", "scaler")}
        meta.setdefault("step", 0)
        meta.setdefault("train_config", {})
        torch.save({**meta, "model": model.state_dict(), "model_config": model.cfg.to_dict(),
                    "tokenizer_sha": (ckpt or {}).get("tokenizer_sha", data.meta.get("tokenizer_sha")),
                    "kgirl": record}, out / "improved.pt")
    return record


# ---------------------------------------------------------------------- inference

@torch.no_grad()
def generate_with_memory(model: TAULS, mem: HoloMemory, g: dict, prompt: list[int], max_new: int,
                         temperature: float = 0.8, top_k: int = 50, top_p: float = 0.95,
                         stop: set[int] | None = None, sampler=None) -> list[int]:
    """Sample from p = (1 − w) p_LM + w p_mem token by token (KV cache; chaos sampler; w from the fitted gate)."""
    from .model import sample_token
    model.eval()
    sampler = sampler or chaos.Sampler(model.cfg.init_seed, chaos.stream_id("generate", "kgirl"))
    device = model.embed.weight.device
    prompt = prompt[-max(1, model.cfg.ctx - max_new):] or [0]
    cache = model.new_cache()
    h = None
    for i in range(0, len(prompt), 256):
        h = model.features(torch.tensor([prompt[i:i + 256]], device=device), cache=cache)[0, -1]
    out: list[int] = []
    for _ in range(max_new):
        if cache.pos >= model.cfg.ctx:
            break
        p_lm = torch.softmax(model.logits_of(h).float(), -1).cpu().numpy()[None]
        p_mem, _ = mem.distribution(mem.project(h.float().cpu().numpy()[None]), **memory_kwargs(g))
        w = gate(entropy(p_lm), entropy(p_mem), g.get("prior", 0.1), g.get("temp", 1.0))[:, None]
        p = (1 - w) * p_lm + w * p_mem
        logits = torch.from_numpy(np.log(np.clip(p[0], 1e-30, None)).astype(np.float32))
        tok = sample_token(logits, temperature, top_k, top_p, float(sampler.uniform(1)[0]), "softmax")
        out.append(tok)
        if stop and tok in stop:
            break
        h = model.features(torch.tensor([[tok]], device=device), cache=cache)[0, -1]
    return out
