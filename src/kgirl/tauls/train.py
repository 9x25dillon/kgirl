"""Training loop for TA-ULS: built to run unattended for days on one 8 GB GPU, and resume exactly.

    step n:  for each of `accum` micro-batches:  windows(train, n·accum + m) → forward (autocast) → loss/accum → backward
             clip ‖g‖ → AdamW step (cosine LR, warmup) → log │ eval (val loss, bits/byte) │ checkpoint

Guarantees (Level 0, tested on CPU in tests/tauls/test_train.py):
  * deterministic batches: a pure function of (seed, step), so resuming replays the same data
  * atomic checkpoints (write to .tmp, then os.replace); `latest.pt` always loads
  * Ctrl-C / SIGTERM saves a checkpoint before exiting
  * bf16 autocast on GPUs that support it, fp16 with a GradScaler otherwise, fp32 on CPU
  * --auto-batch finds the largest micro-batch that fits and raises `accum` to keep tokens/step

  * a coherence guard (κ thermostat from 9x25dillon/Fractal_cascade_simulation's CoherenceController)
    skips loss spikes and non-finite steps instead of letting one bad batch wreck a multi-day run

Metrics go to <out>/metrics.jsonl, one JSON object per log or eval event.
"""

from __future__ import annotations

import json
import math
import os
import signal
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import torch

from .data import TokenData
from .model import PRESETS, TAULS, TAULSConfig


@dataclass
class TrainConfig:
    data: str = "runs/tauls/data"
    out: str = "runs/tauls/model"
    preset: str = "small"
    model: dict = field(default_factory=dict)       # overrides on top of the preset
    batch: int = 16                                  # micro-batch (sequences)
    accum: int = 4                                   # micro-batches per optimizer step
    steps: int = 20000
    lr: float = 6e-4
    min_lr: float = 6e-5
    warmup: int = 500
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_every: int = 500
    eval_tokens: int = 262144
    log_every: int = 20
    save_every: int = 1000
    keep: int = 3                                    # rolling step checkpoints kept besides latest/best
    seed: float = 0.4123
    device: str = "auto"
    precision: str = "auto"                          # auto | bf16 | fp16 | fp32
    compile: bool = False
    grad_checkpoint: bool = False
    auto_batch: bool = False
    max_minutes: float = 0.0                         # 0 = no wall-clock limit
    patience: int = 0                                # evals without val improvement before stopping; 0 = off
    stop_at: int = 0                                 # end this session at this step (schedule unchanged); 0 = off
    guard: bool = True                               # coherence guard on gradient-norm spikes
    guard_beta: float = 0.98
    guard_warmup: int = 50                           # steps of EMA before the guard may act
    guard_max_skips: int = 25                        # consecutive CRITICAL steps → stop (diverged)
    stability: bool = True                           # weight spectral/KFP summary at each eval (numpy or Julia)
    julia_url: str = ""                              # TA-ULS Julia bridge for the summary; "" = numpy only

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


def model_config(tc: TrainConfig, vocab_size: int) -> TAULSConfig:
    if tc.preset not in PRESETS:
        raise ValueError(f"unknown preset {tc.preset!r}; choose from {sorted(PRESETS)}")
    base = PRESETS[tc.preset].to_dict()
    base.update(tc.model)
    base["vocab_size"] = vocab_size
    return TAULSConfig.from_dict(base)


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_precision(name: str, device: torch.device) -> torch.dtype | None:
    """The autocast dtype, or None for plain fp32."""
    if name == "fp32" or (name == "auto" and device.type == "cpu"):
        return None
    if name == "bf16" or (name == "auto" and device.type == "cuda" and torch.cuda.is_bf16_supported()):
        return torch.bfloat16
    return torch.float16


@dataclass
class CoherenceGuard:
    """κ = min(1, EMA(‖g‖) / ‖g‖): how coherent this step's gradient is with the recent trajectory.

    Thresholds follow CoherenceController.decide (Fractal_cascade_simulation, unified_coherence_system):
        κ < 0.2  CRITICAL  skip the update (also for non-finite loss or gradient)
        κ < 0.5  LOW       take the step at half the learning rate
        else     STABLE / HIGH: normal step
    The EMA only absorbs steps that were taken, so a spike cannot raise its own threshold.
    """
    beta: float = 0.98
    warmup: int = 50
    ema: float = 0.0
    seen: int = 0
    skips: int = 0
    run: int = 0

    def decide(self, gnorm: float, loss: float) -> tuple[str, float]:
        if not (math.isfinite(gnorm) and math.isfinite(loss)):
            return "CRITICAL", 0.0
        if self.seen < self.warmup or gnorm <= 0:
            return "STABLE", 1.0
        kappa = min(1.0, (self.ema / (1 - self.beta ** self.seen)) / gnorm)
        if kappa < 0.2:
            return "CRITICAL", kappa
        if kappa < 0.5:
            return "LOW", kappa
        return ("HIGH" if kappa >= 0.8 else "STABLE"), kappa

    def absorb(self, state: str, gnorm: float) -> None:
        if state == "CRITICAL":
            self.skips += 1
            self.run += 1
            return
        self.run = 0
        self.ema = self.beta * self.ema + (1 - self.beta) * gnorm
        self.seen += 1


def lr_at(step: int, tc: TrainConfig) -> float:
    if step < tc.warmup:
        return tc.lr * (step + 1) / tc.warmup
    p = min(1.0, (step - tc.warmup) / max(1, tc.steps - tc.warmup))
    return tc.min_lr + 0.5 * (tc.lr - tc.min_lr) * (1 + math.cos(math.pi * p))


def make_optimizer(model: TAULS, tc: TrainConfig, device: torch.device) -> torch.optim.AdamW:
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    rest = [p for n, p in model.named_parameters() if p.dim() < 2]
    groups = [{"params": decay, "weight_decay": tc.weight_decay}, {"params": rest, "weight_decay": 0.0}]
    kw = {"fused": True} if device.type == "cuda" else {}
    return torch.optim.AdamW(groups, lr=tc.lr, betas=(tc.beta1, tc.beta2), **kw)


# ---------------------------------------------------------------------------- checkpoints

def save_checkpoint(path: Path, model: TAULS, opt, scaler, step: int, tc: TrainConfig, meta: dict, best: float,
                    guard: CoherenceGuard | None = None) -> None:
    raw = getattr(model, "_orig_mod", model)
    state = {"model": raw.state_dict(), "optimizer": opt.state_dict(),
             "scaler": scaler.state_dict() if scaler is not None else None, "step": step,
             "train_config": asdict(tc), "model_config": raw.cfg.to_dict(), "tokenizer_sha": meta["tokenizer_sha"],
             "best_val": best, "time": time.time(), "guard": asdict(guard) if guard else None}
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_model(path: str | Path, device: str | torch.device = "cpu") -> tuple[TAULS, dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    model = TAULS(TAULSConfig.from_dict(ck["model_config"])).to(device)
    model.load_state_dict(ck["model"])
    return model, ck


# ---------------------------------------------------------------------------- evaluation

@torch.no_grad()
def evaluate(model: TAULS, data: TokenData, split: str, batch: int, max_tokens: int, device: torch.device,
             amp: torch.dtype | None) -> dict:
    model.eval()
    ctx = model.cfg.ctx
    nats, count = 0.0, 0
    for x, y in data.sequential(split, batch, ctx, max_tokens):
        x = torch.from_numpy(x).to(device)
        y = torch.from_numpy(y).to(device)
        with torch.autocast(device.type, dtype=amp, enabled=amp is not None):
            loss = model(x, y, logits=False).loss
        nats += float(loss) * y.numel()
        count += y.numel()
    model.train()
    if not count:
        return {}
    loss = nats / count
    return {"loss": loss, "ppl": math.exp(min(loss, 50.0)),
            "bpb": loss / math.log(2) / data.bytes_per_token(split), "tokens": count}


# ---------------------------------------------------------------------------- micro-batch search

def find_batch(model: TAULS, tc: TrainConfig, device: torch.device, amp, log) -> None:
    """Largest power-of-two micro-batch ≤ tc.batch that survives forward+backward; keeps tokens/step."""
    if device.type != "cuda":
        return
    total = tc.batch * tc.accum
    b = tc.batch
    while b >= 1:
        try:
            x = torch.zeros(b, model.cfg.ctx, dtype=torch.long, device=device)
            with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                model(x, x, logits=False).loss.backward()
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            break
        except torch.cuda.OutOfMemoryError:
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            b //= 2
    if b < 1:
        raise RuntimeError("even micro-batch 1 does not fit: lower ctx or d_model, or set grad_checkpoint")
    tc.batch, tc.accum = b, max(1, total // b)
    log(f"auto-batch: micro-batch {tc.batch} × accum {tc.accum}")


# ---------------------------------------------------------------------------- loop

def train(tc: TrainConfig, resume: bool = False, log=print) -> dict:
    out = Path(tc.out)
    out.mkdir(parents=True, exist_ok=True)
    data = TokenData(tc.data)
    meta = data.meta
    device = resolve_device(tc.device)
    amp = resolve_precision(tc.precision, device)
    start, best, ck = 0, float("inf"), None
    if resume and (out / "latest.pt").exists():
        ck = torch.load(out / "latest.pt", map_location="cpu", weights_only=False)
        if ck["tokenizer_sha"] != meta["tokenizer_sha"]:
            raise ValueError("checkpoint and data were made with different tokenizers")
        saved = TrainConfig.from_dict(ck["train_config"])
        for k in ("steps", "max_minutes", "device", "data", "out", "patience", "stop_at"):  # per-session
            setattr(saved, k, getattr(tc, k))
        tc = saved
        mcfg = TAULSConfig.from_dict(ck["model_config"])
        start, best = ck["step"], ck.get("best_val", float("inf"))
    else:
        mcfg = model_config(tc, meta["vocab_size"])
    if mcfg.vocab_size != meta["vocab_size"]:
        raise ValueError("model vocabulary does not match the data's tokenizer")
    if data.tokens("train") <= mcfg.ctx + 1:
        raise ValueError(f"train split has {data.tokens('train')} tokens; need more than ctx = {mcfg.ctx}")

    model = TAULS(mcfg).to(device)
    model.grad_checkpoint = tc.grad_checkpoint
    opt = make_optimizer(model, tc, device)
    scaler = torch.amp.GradScaler("cuda") if amp == torch.float16 and device.type == "cuda" else None
    if ck is not None:
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        if scaler is not None and ck.get("scaler"):
            scaler.load_state_dict(ck["scaler"])
        log(f"resumed at step {start} (best val {best:.4f})")
    elif tc.auto_batch:
        find_batch(model, tc, device, amp, log)
    guard = CoherenceGuard(tc.guard_beta, tc.guard_warmup)
    if ck is not None and ck.get("guard"):
        guard = CoherenceGuard(**ck["guard"])
    (out / "config.json").write_text(json.dumps({"train": asdict(tc), "model": mcfg.to_dict()}, indent=2))
    fwd = torch.compile(model) if tc.compile else model

    n_params = model.num_params()
    tokens_per_step = tc.batch * tc.accum * mcfg.ctx
    log(f"TA-ULS {tc.preset}: {n_params / 1e6:.2f}M params ({model.num_params(True) / 1e6:.2f}M non-embedding) "
        f"on {device} {amp or 'fp32'}; {tokens_per_step:,} tokens/step; "
        f"train {data.tokens('train'):,} tokens = {data.tokens('train') / tokens_per_step:.0f} steps/epoch")

    stop = {"flag": False}

    def _stop(signum, frame):
        stop["flag"] = True
        log(f"signal {signum}: saving and stopping after this step")
    old = {}
    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            old[s] = signal.signal(s, _stop)
        except ValueError:                           # not the main thread: no signal handling
            pass

    metrics = open(out / "metrics.jsonl", "a", encoding="utf-8")

    def emit(rec):
        metrics.write(json.dumps(rec) + "\n")
        metrics.flush()

    t0 = time.time()
    t_log, tok_log = time.time(), 0
    bad_evals = 0
    step = start
    last_eval: dict = {}
    try:
        while step < tc.steps and not stop["flag"]:
            lr = lr_at(step, tc)
            for g in opt.param_groups:
                g["lr"] = lr
            collect = (step + 1) % tc.log_every == 0
            loss_sum = 0.0
            for m in range(tc.accum):
                x, y = data.windows("train", step * tc.accum + m, tc.batch, mcfg.ctx, tc.seed)
                x = torch.from_numpy(x).to(device, non_blocking=True)
                y = torch.from_numpy(y).to(device, non_blocking=True)
                with torch.autocast(device.type, dtype=amp, enabled=amp is not None):
                    res = fwd(x, y, collect=collect and m == tc.accum - 1, logits=False)
                    loss = res.loss / tc.accum
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                loss_sum += float(loss.detach())
            if scaler is not None:
                scaler.unscale_(opt)
            gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip))
            state, kappa = guard.decide(gnorm, loss_sum) if tc.guard else ("STABLE", 1.0)
            if state == "LOW":
                for g in opt.param_groups:
                    g["lr"] = lr * 0.5
            if state == "CRITICAL" and not (scaler is not None and not math.isfinite(gnorm)):
                if scaler is not None:
                    scaler.update()                  # the scaler must close the step it unscaled
                emit({"event": "guard", "step": step + 1, "state": state, "kappa": kappa, "grad_norm": gnorm,
                      "loss": loss_sum})
            elif scaler is not None:
                scaler.step(opt)                     # fp16 overflow: the scaler skips and backs off itself
                scaler.update()
            else:
                opt.step()
            guard.absorb(state, gnorm)
            opt.zero_grad(set_to_none=True)
            if tc.guard and guard.run >= tc.guard_max_skips:
                log(f"coherence guard: {guard.run} consecutive CRITICAL steps; stopping (diverged)")
                step += 1
                break
            step += 1
            tok_log += tokens_per_step

            if collect or step == tc.steps:
                dt = max(1e-9, time.time() - t_log)
                rec = {"event": "train", "step": step, "loss": loss_sum, "lr": lr, "grad_norm": gnorm,
                       "kappa": kappa, "coherence": state, "guard_skips": guard.skips,
                       "tok_s": tok_log / dt, "elapsed_min": (time.time() - t0) / 60}
                if res.metrics:
                    rec["tauls"] = res.metrics
                emit(rec)
                log(f"step {step:>6} loss {loss_sum:.4f} lr {lr:.2e} |g| {gnorm:.2f} κ {kappa:.2f} {state} "
                    f"{rec['tok_s']:,.0f} tok/s")
                t_log, tok_log = time.time(), 0

            if step % tc.eval_every == 0 or step == tc.steps:
                last_eval = evaluate(model, data, "val", tc.batch, tc.eval_tokens, device, amp) if data.tokens("val") > mcfg.ctx else {}
                if tc.stability:
                    from .julia_bridge import JuliaBridge, summarize, weight_report
                    summ = summarize(weight_report(model.state_dict(), JuliaBridge(tc.julia_url) if tc.julia_url else None,
                                                   include_embedding=False))
                    emit({"event": "stability", "step": step, **summ})
                if last_eval:
                    emit({"event": "eval", "step": step, **last_eval})
                    log(f"eval  {step:>6} val loss {last_eval['loss']:.4f} ppl {last_eval['ppl']:.2f} "
                        f"bits/byte {last_eval['bpb']:.4f}")
                    if last_eval["loss"] < best:
                        best, bad_evals = last_eval["loss"], 0
                        save_checkpoint(out / "best.pt", model, opt, scaler, step, tc, meta, best, guard)
                    else:
                        bad_evals += 1
                if tc.patience and bad_evals >= tc.patience:
                    log(f"no val improvement in {bad_evals} evals: stopping")
                    break
            if step % tc.save_every == 0:
                save_checkpoint(out / f"step-{step:07d}.pt", model, opt, scaler, step, tc, meta, best, guard)
                for old_ck in sorted(out.glob("step-*.pt"))[:-tc.keep]:
                    old_ck.unlink()
            if tc.stop_at and step >= tc.stop_at:
                log(f"reached stop_at {tc.stop_at}: stopping (resume with --resume)")
                break
            if tc.max_minutes and (time.time() - t0) / 60 >= tc.max_minutes:
                log(f"reached {tc.max_minutes} min: stopping")
                break
    finally:
        save_checkpoint(out / "latest.pt", model, opt, scaler, step, tc, meta, best, guard)
        metrics.close()
        for s, h in old.items():
            signal.signal(s, h)
    return {"step": step, "best_val": best, "last_eval": last_eval, "params": n_params}
