"""Substrates: the containers a vessel's identity is measured in and steered through.

A substrate is anything that (1) maps a fixed probe set to per-layer activation matrices, (2) exposes its
parameters as named arrays, and (3) can take one gradient step on the vessel objective

    L(θ) = w_task · CE(θ; batch)                                         plasticity: learn the batch
         + λ_rep · Σ_l ω_l · (1 − CKA(X_l(θ; probes), A_l))              identity: stay aligned with anchor Grams
         + λ_voice · KL(p_anchor ‖ p_θ) on probes                         voice: anchored next-token behaviour
         + λ_w / 2 · Σ_i F_i · (θ_i − θ*_i)²                             inertia: EWC / L2-SP weight anchor

The representation term only sees normalized centered Grams, so it is indifferent to how the substrate
stores its features. It is also rotation-invariant by design, which means it cannot pin the readout: a vessel
can keep its internal geometry and still speak differently. The voice term (self-distillation on the probes,
Li & Hoiem 2016) closes that gap and needs only a shared vocabulary, so it too crosses substrates. The weight
term is substrate-specific and optional.

Two implementations:
    ReferenceLM      numpy, byte-level MLP language model with hand-written backprop (this module).
                     Deterministic: initialised and sampled from ChaosRNG, no PRNG.
    TorchSubstrate   any torch module, AURIC's LanguageModel included (torch_substrate.py).
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Protocol, runtime_checkable

import numpy as np

from ..evolve.chaos import ChaosRNG
from .geometry import cka_to_anchor_grad


@dataclass
class Objective:
    lr: float
    task_weight: float = 1.0
    lam_rep: float = 0.0
    anchors: Mapping[str, np.ndarray] = field(default_factory=dict)       # layer → normalized anchor Gram
    layer_weights: Mapping[str, float] = field(default_factory=dict)      # ω_l, default 1 / |anchors|
    probes: Any = None
    lam_voice: float = 0.0
    voice: Optional[np.ndarray] = None                                   # anchor log-probs on probes (n × V)
    lam_w: float = 0.0
    anchor_params: Mapping[str, np.ndarray] = field(default_factory=dict)  # θ*
    fisher: Mapping[str, np.ndarray] = field(default_factory=dict)        # F (default 1)
    clip: float = 1.0

    def omega(self, layer: str) -> float:
        if self.layer_weights:
            return float(self.layer_weights.get(layer, 0.0))
        return 1.0 / max(1, len(self.anchors))


@dataclass(frozen=True)
class StepReport:
    task_loss: float
    rep_loss: float
    voice_kl: float
    weight_penalty: float
    coherence: dict
    grad_norm: float

    @property
    def total(self) -> float:
        return self.task_loss + self.rep_loss + self.voice_kl + self.weight_penalty


@runtime_checkable
class Substrate(Protocol):
    layers: tuple[str, ...]

    def describe(self) -> dict: ...
    def activations(self, probes: Any, shift: Optional[Mapping[str, np.ndarray]] = None) -> dict[str, np.ndarray]: ...
    def parameters(self) -> dict[str, np.ndarray]: ...
    def load_parameters(self, params: Mapping[str, np.ndarray]) -> None: ...
    def loss(self, batch: Any) -> float: ...
    def log_probs(self, probes: Any) -> np.ndarray: ...
    def step(self, batch: Any, objective: Objective) -> StepReport: ...
    def checkpoint(self) -> Any: ...
    def restore(self, state: Any) -> None: ...


# ----------------------------------------------------------------------------------------------- Adam

class Adam:
    """Adam (Kingma & Ba 2015) over a dict of numpy arrays, with global-norm gradient clipping."""

    def __init__(self, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.b1, self.b2, self.eps = beta1, beta2, eps
        self.t = 0
        self.m: dict[str, np.ndarray] = {}
        self.v: dict[str, np.ndarray] = {}

    def update(self, params: dict[str, np.ndarray], grads: Mapping[str, np.ndarray], lr: float, clip: float) -> float:
        norm = math.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
        scale = clip / norm if clip and norm > clip else 1.0
        self.t += 1
        for k, g in grads.items():
            g = g * scale
            m = self.m.setdefault(k, np.zeros_like(g))
            v = self.v.setdefault(k, np.zeros_like(g))
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            mh = m / (1 - self.b1 ** self.t)
            vh = v / (1 - self.b2 ** self.t)
            params[k] -= lr * mh / (np.sqrt(vh) + self.eps)
        return norm


# ----------------------------------------------------------------------------------------------- ReferenceLM

@dataclass(frozen=True)
class RefConfig:
    context: int = 8
    embed: int = 16
    hidden: int = 64
    vocab: int = 256
    seed: float = 0.4123

    def __post_init__(self):
        if min(self.context, self.embed, self.hidden, self.vocab) < 1:
            raise ValueError("ReferenceLM dimensions must be positive")


class ReferenceLM:
    """Byte-level next-byte model: E[ctx] → tanh(W1) → tanh(W2) → W3 → softmax over 256 bytes.

    Layers exposed for identity: 'h1' and 'h2' (post-tanh, n × hidden). A steering shift (persona vector) is
    added after the tanh, so inference-time steering and the measured geometry agree.
    """

    layers = ("h1", "h2")

    def __init__(self, cfg: RefConfig = RefConfig()):
        self.cfg = cfg
        rng = ChaosRNG(cfg.seed)
        c, e, h, V = cfg.context, cfg.embed, cfg.hidden, cfg.vocab

        def init(shape, fan_in):
            return rng.normal(shape) / math.sqrt(fan_in)

        self.p: dict[str, np.ndarray] = {
            "E": init((V, e), 1.0) * 0.1,
            "W1": init((c * e, h), c * e), "b1": np.zeros(h),
            "W2": init((h, h), h), "b2": np.zeros(h),
            "W3": init((h, V), h), "b3": np.zeros(V),
        }
        self.opt = Adam()

    # ------------------------------------------------------------------------------------------- contract

    def describe(self) -> dict:
        return {"substrate": "ReferenceLM", "context": self.cfg.context, "embed": self.cfg.embed,
                "hidden": self.cfg.hidden, "parameters": int(sum(v.size for v in self.p.values())),
                "layers": list(self.layers)}

    def parameters(self) -> dict[str, np.ndarray]:
        return {k: v.copy() for k, v in self.p.items()}

    def load_parameters(self, params: Mapping[str, np.ndarray]) -> None:
        for k in self.p:
            if params[k].shape != self.p[k].shape:
                raise ValueError(f"shape mismatch for {k}: {params[k].shape} vs {self.p[k].shape}")
        self.p = {k: np.array(params[k], dtype=np.float64) for k in self.p}

    def checkpoint(self) -> Any:
        return (self.parameters(), copy.deepcopy(self.opt))

    def restore(self, state: Any) -> None:
        params, opt = state
        self.load_parameters(params)
        self.opt = copy.deepcopy(opt)

    def activations(self, probes, shift=None) -> dict[str, np.ndarray]:
        cache = self._forward(np.asarray(probes), shift)
        return {"h1": cache["h1"], "h2": cache["h2"]}

    def loss(self, batch) -> float:
        ctx, y = batch
        logits = self._forward(np.asarray(ctx))["logits"]
        return float(_cross_entropy(logits, np.asarray(y))[0])

    def log_probs(self, probes) -> np.ndarray:
        return _log_softmax(self._forward(np.asarray(probes))["logits"])

    def step(self, batch, objective: Objective) -> StepReport:
        loss, rep, kl, pen, coh, grads = self.objective_and_grad(batch, objective)
        norm = self.opt.update(self.p, grads, objective.lr, objective.clip)
        return StepReport(loss, rep, kl, pen, coh, norm)

    # ------------------------------------------------------------------------------------------- math

    def objective_and_grad(self, batch, obj: Objective):
        grads = {k: np.zeros_like(v) for k, v in self.p.items()}
        loss = 0.0
        if batch is not None and obj.task_weight:
            ctx, y = batch
            cache = self._forward(np.asarray(ctx))
            loss, dlogits = _cross_entropy(cache["logits"], np.asarray(y))
            _accumulate(grads, self._backward(cache, obj.task_weight * dlogits, {}))
        rep, kl, coh = 0.0, 0.0, {}
        rep_on = bool(obj.lam_rep and obj.anchors)
        voice_on = bool(obj.lam_voice and obj.voice is not None)
        if rep_on or voice_on:
            cache = self._forward(np.asarray(obj.probes))
            dh, dlogits = {}, None
            if rep_on:
                for layer, A in obj.anchors.items():
                    f, g = cka_to_anchor_grad(cache[layer], A)
                    w = obj.omega(layer)
                    coh[layer] = f
                    rep += obj.lam_rep * w * (1.0 - f)
                    dh[layer] = -obj.lam_rep * w * g
            if voice_on:
                k, dk = kl_to_anchor(cache["logits"], obj.voice)
                kl = obj.lam_voice * k
                dlogits = obj.lam_voice * dk
            _accumulate(grads, self._backward(cache, dlogits, dh))
        pen = 0.0
        if obj.lam_w and obj.anchor_params:
            for k, star in obj.anchor_params.items():
                F = obj.fisher.get(k, 1.0)
                d = self.p[k] - star
                pen += 0.5 * obj.lam_w * float(np.sum(F * d * d))
                grads[k] += obj.lam_w * F * d
        return loss, rep, kl, pen, coh, grads

    def _forward(self, ctx: np.ndarray, shift: Optional[Mapping[str, np.ndarray]] = None) -> dict:
        p = self.p
        x = p["E"][ctx].reshape(len(ctx), -1)
        t1 = np.tanh(x @ p["W1"] + p["b1"])
        h1 = t1 + shift["h1"] if shift and "h1" in shift else t1
        t2 = np.tanh(h1 @ p["W2"] + p["b2"])
        h2 = t2 + shift["h2"] if shift and "h2" in shift else t2
        return {"ctx": ctx, "x": x, "t1": t1, "h1": h1, "t2": t2, "h2": h2, "logits": h2 @ p["W3"] + p["b3"]}

    def _backward(self, cache: dict, dlogits: Optional[np.ndarray], dh: Mapping[str, np.ndarray]) -> dict:
        p, g = self.p, {}
        n = len(cache["ctx"])
        dh2 = np.zeros((n, self.cfg.hidden))
        if dlogits is not None:
            g["W3"] = cache["h2"].T @ dlogits
            g["b3"] = dlogits.sum(axis=0)
            dh2 += dlogits @ p["W3"].T
        if "h2" in dh:
            dh2 += dh["h2"]
        da2 = dh2 * (1.0 - cache["t2"] ** 2)
        g["W2"] = cache["h1"].T @ da2
        g["b2"] = da2.sum(axis=0)
        dh1 = da2 @ p["W2"].T
        if "h1" in dh:
            dh1 = dh1 + dh["h1"]
        da1 = dh1 * (1.0 - cache["t1"] ** 2)
        g["W1"] = cache["x"].T @ da1
        g["b1"] = da1.sum(axis=0)
        dx = (da1 @ p["W1"].T).reshape(n, self.cfg.context, self.cfg.embed)
        gE = np.zeros_like(p["E"])
        np.add.at(gE, cache["ctx"], dx)
        g["E"] = gE
        return g

    # ------------------------------------------------------------------------------------------- use

    def fisher_diagonal(self, batches) -> dict[str, np.ndarray]:
        """Mean squared minibatch gradient of the task loss, normalized to mean 1 per tensor.

        A minibatch-gradient approximation of the empirical Fisher (Level 1: cheaper and noisier than the
        per-example version; adequate as an EWC importance weight)."""
        acc = {k: np.zeros_like(v) for k, v in self.p.items()}
        count = 0
        for ctx, y in batches:
            cache = self._forward(np.asarray(ctx))
            _, dl = _cross_entropy(cache["logits"], np.asarray(y))
            for k, v in self._backward(cache, dl, {}).items():
                acc[k] += v * v
            count += 1
        return {k: v / max(v.mean(), 1e-12) for k, v in ((k, v / max(count, 1)) for k, v in acc.items())}

    def generate(self, prompt: bytes, n: int = 64, temperature: float = 0.0, seed: float = 0.4123,
                 shift: Optional[Mapping[str, np.ndarray]] = None) -> bytes:
        c = self.cfg.context
        buf = list(prompt[-c:])
        rng = ChaosRNG(seed)
        out = bytearray()
        for _ in range(n):
            ctx = np.array([([0] * c + buf)[-c:]])
            logits = self._forward(ctx, shift)["logits"][0]
            if temperature <= 0:
                b = int(np.argmax(logits))
            else:
                z = (logits - logits.max()) / temperature
                pr = np.exp(z)
                b = int(np.searchsorted(np.cumsum(pr / pr.sum()), rng.uniform(), side="right"))
                b = min(b, self.cfg.vocab - 1)
            out.append(b)
            buf.append(b)
        return bytes(out)


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def kl_to_anchor(logits: np.ndarray, anchor_logp: np.ndarray) -> tuple[float, np.ndarray]:
    """Mean KL(p_anchor ‖ softmax(logits)) over rows, and its gradient (p − p_anchor) / n w.r.t. logits."""
    logp = _log_softmax(logits)
    pa = np.exp(anchor_logp)
    n = len(logits)
    kl = float(np.sum(pa * (anchor_logp - logp)) / n)
    return kl, (np.exp(logp) - pa) / n


def _cross_entropy(logits: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    logp = _log_softmax(logits)
    n = len(y)
    loss = -float(logp[np.arange(n), y].mean())
    d = np.exp(logp)
    d[np.arange(n), y] -= 1.0
    return loss, d / n


def _accumulate(into: dict, more: Mapping[str, np.ndarray]) -> None:
    for k, v in more.items():
        into[k] += v
