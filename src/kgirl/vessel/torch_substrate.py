"""TorchSubstrate: host a vessel in any byte-level torch language model, AURIC's LanguageModel included.

The adapter translates the vessel's neutral formats into the model's token space and leaves the model alone:

    byte b              → token b + offset          (AURIC: offset 4, after pad/bos/eos/sep)
    window (c bytes)    → [bos] + tokens            one sequence per probe or training example
    layer activation    → output of a hooked module at the last position (n × d), the state that predicts
                          the next byte, matching the reference substrate's per-window representation
    voice               → log-softmax over the 256 byte tokens at the last position (other tokens excluded),
                          so voices compare across substrates that share bytes, whatever their vocabulary
    step                → autograd on the same objective as ReferenceLM (CKA + KL + CE + EWC), Adam, clipped

The model stays in eval mode throughout (no dropout), so a steering step is a deterministic function of the
weights and batch. Requires torch; this module is imported lazily by the CLI and tests.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Mapping, Optional

import numpy as np
import torch
from torch.nn import functional as F

from ..evolve.chaos import ChaosRNG
from .substrate import Objective, StepReport


def _normalized_gram(X: torch.Tensor) -> torch.Tensor:
    Xc = X - X.mean(dim=0, keepdim=True)
    K = Xc @ Xc.T
    return K / K.norm().clamp_min(1e-12)


class TorchSubstrate:
    def __init__(self, model: torch.nn.Module, layers: Mapping[str, torch.nn.Module], *, offset: int = 4,
                 bos: Optional[int] = 1, name: str = "torch", device: str = "cpu"):
        self.model = model.to(device).eval()
        self.layer_modules = dict(layers)
        self.layers = tuple(self.layer_modules)
        self.offset, self.bos, self.name, self.device = offset, bos, name, torch.device(device)
        self.opt = torch.optim.Adam([p for p in self.model.parameters() if p.requires_grad], lr=1e-3)
        self._captured: dict[str, torch.Tensor] = {}
        self._handles = [module.register_forward_hook(self._hook(lname)) for lname, module in self.layer_modules.items()]

    def close(self) -> None:
        """Remove the forward hooks, so the model can be wrapped again (or used bare) without stacked hooks."""
        for h in self._handles:
            h.remove()
        self._handles = []

    def _hook(self, lname: str):
        def capture(_module, _inputs, output):
            out = output[0] if isinstance(output, tuple) else output
            self._captured[lname] = out[:, -1, :] if out.dim() == 3 else out
        return capture

    # ------------------------------------------------------------------------------------------- formats

    def _ids(self, windows) -> torch.Tensor:
        ids = torch.as_tensor(np.asarray(windows), dtype=torch.long) + self.offset
        if self.bos is not None:
            ids = torch.cat([torch.full((ids.shape[0], 1), self.bos, dtype=torch.long), ids], dim=1)
        return ids.to(self.device)

    def _logits(self, ids: torch.Tensor) -> torch.Tensor:
        out = self.model(ids)
        return out[0] if isinstance(out, tuple) else out

    def _run(self, windows) -> tuple[torch.Tensor, dict]:
        self._captured = {}
        logits = self._logits(self._ids(windows))[:, -1, :]
        return logits, dict(self._captured)

    def _byte_logp(self, logits: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(logits[:, self.offset:self.offset + 256].float(), dim=-1)

    # ------------------------------------------------------------------------------------------- contract

    def describe(self) -> dict:
        return {"substrate": self.name, "parameters": int(sum(p.numel() for p in self.model.parameters())),
                "layers": list(self.layers)}

    def parameters(self) -> dict[str, np.ndarray]:
        return {k: v.detach().cpu().numpy().astype(np.float64, copy=True) for k, v in self.model.named_parameters()}

    def load_parameters(self, params: Mapping[str, np.ndarray]) -> None:
        with torch.no_grad():
            for k, p in self.model.named_parameters():
                src = torch.as_tensor(np.asarray(params[k]), dtype=p.dtype)
                if src.shape != p.shape:
                    raise ValueError(f"shape mismatch for {k}: {tuple(src.shape)} vs {tuple(p.shape)}")
                p.copy_(src)

    def checkpoint(self) -> Any:
        return copy.deepcopy(self.model.state_dict()), copy.deepcopy(self.opt.state_dict())

    def restore(self, state: Any) -> None:
        weights, opt = state
        self.model.load_state_dict(weights)
        self.opt.load_state_dict(opt)

    @torch.no_grad()
    def activations(self, probes, shift=None) -> dict[str, np.ndarray]:
        _, acts = self._run(probes)
        return {k: acts[k].double().cpu().numpy() for k in self.layers}

    @torch.no_grad()
    def log_probs(self, probes) -> np.ndarray:
        logits, _ = self._run(probes)
        return self._byte_logp(logits).double().cpu().numpy()

    @torch.no_grad()
    def loss(self, batch) -> float:
        ctx, y = batch
        logits, _ = self._run(ctx)
        return float(F.cross_entropy(logits.float(), torch.as_tensor(np.asarray(y), device=self.device) + self.offset))

    def step(self, batch, obj: Objective) -> StepReport:
        self.opt.zero_grad(set_to_none=True)
        total = torch.zeros((), device=self.device)
        task = rep = kl = pen = 0.0
        coh = {}
        if batch is not None and obj.task_weight:
            ctx, y = batch
            logits, _ = self._run(ctx)
            ce = F.cross_entropy(logits.float(), torch.as_tensor(np.asarray(y), device=self.device) + self.offset)
            total = total + obj.task_weight * ce
            task = float(ce.detach())
        rep_on = bool(obj.lam_rep and obj.anchors)
        voice_on = bool(obj.lam_voice and obj.voice is not None)
        if rep_on or voice_on:
            logits, acts = self._run(obj.probes)
            if rep_on:
                for layer, A in obj.anchors.items():
                    G = _normalized_gram(acts[layer].double())
                    f = (G * torch.as_tensor(A, dtype=torch.float64, device=self.device)).sum()
                    coh[layer] = float(f.detach())
                    term = obj.lam_rep * obj.omega(layer) * (1.0 - f)
                    total = total + term.float()
                    rep += float(term.detach())
            if voice_on:
                pa = torch.as_tensor(np.exp(obj.voice), dtype=torch.float32, device=self.device)
                la = torch.as_tensor(obj.voice, dtype=torch.float32, device=self.device)
                k = (pa * (la - self._byte_logp(logits))).sum(dim=-1).mean()
                total = total + obj.lam_voice * k
                kl = float(obj.lam_voice * k.detach())
        if obj.lam_w and obj.anchor_params:
            for name, p in self.model.named_parameters():
                if name in obj.anchor_params:
                    star = torch.as_tensor(obj.anchor_params[name], dtype=p.dtype, device=self.device)
                    Fd = torch.as_tensor(obj.fisher.get(name, 1.0), dtype=p.dtype, device=self.device)
                    term = 0.5 * obj.lam_w * (Fd * (p - star) ** 2).sum()
                    total = total + term
                    pen += float(term.detach())
        if total.requires_grad:
            total.backward()
            norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), obj.clip or float("inf")))
            for group in self.opt.param_groups:
                group["lr"] = obj.lr
            self.opt.step()
        else:
            norm = 0.0
        return StepReport(task, rep, kl, pen, coh, norm)

    def generate(self, prompt: bytes, n: int = 64, temperature: float = 0.0, seed: float = 0.4123,
                 shift=None) -> bytes:
        rng = ChaosRNG(seed)
        ids = ([self.bos] if self.bos is not None else []) + [b + self.offset for b in prompt]
        ctx = getattr(getattr(self.model, "config", None), "context", None)
        out = bytearray()
        with torch.no_grad():
            for _ in range(n):
                window = ids[-ctx:] if ctx else ids
                logp = self._byte_logp(self._logits(torch.tensor([window], device=self.device))[:, -1, :])[0]
                if temperature <= 0:
                    b = int(logp.argmax())
                else:
                    pr = torch.softmax(logp / temperature, dim=-1).double().cpu().numpy()
                    b = min(int(np.searchsorted(np.cumsum(pr), rng.uniform(), side="right")), 255)
                out.append(b)
                ids.append(b + self.offset)
        return bytes(out)


# ----------------------------------------------------------------------------------------------- AURIC

def chaos_init(model: torch.nn.Module, seed: float = 0.4123) -> torch.nn.Module:
    """Re-initialize from ChaosRNG with torch's default variances: Linear weights N(0, 1/(3·fan_in)) (the variance
    of kaiming_uniform(a=√5)), embeddings N(0, 0.02²), biases 0, norm gains 1. Zero-initialized gates stay zero."""
    rng = ChaosRNG(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "gate" in name:
                continue
            if p.dim() == 2:
                std = 0.02 if ("embedding" in name or "position" in name) else 1.0 / math.sqrt(3 * p.shape[1])
                p.copy_(torch.as_tensor(rng.normal(tuple(p.shape)) * std, dtype=p.dtype))
            elif "norm" in name and name.endswith("weight"):
                p.fill_(1.0)
            else:
                p.zero_()
    return model


def auric_substrate(checkpoint: Optional[str] = None, *, preset: str = "cpu", seed: float = 0.4123,
                    overrides: Optional[dict] = None, device: str = "cpu") -> TorchSubstrate:
    """Load AURIC (patern-coding) and wrap it. With `checkpoint`, use AURIC's own loader; otherwise build the
    preset (optionally overridden, e.g. {"dim": 64, "layers": 2}) and initialize it from ChaosRNG."""
    from ..paths import ensure_on_path
    found = ensure_on_path("patern-coding")
    if found.get("patern-coding") is None:
        raise ImportError("patern-coding not found; set PATERN_CODING_PATH or clone it next to kgirl")
    from auric.model import PRESETS, LanguageModel, ModelConfig
    if checkpoint:
        from auric.training import load_model
        model, _ = load_model(checkpoint, device)
    else:
        from dataclasses import asdict
        cfg = ModelConfig(**{**asdict(PRESETS[preset]), **(overrides or {})})
        model = chaos_init(LanguageModel(cfg), seed)
    layers = {f"block{i}": block for i, block in enumerate(model.blocks)}
    return TorchSubstrate(model, layers, offset=4, bos=1, name="auric", device=device)
