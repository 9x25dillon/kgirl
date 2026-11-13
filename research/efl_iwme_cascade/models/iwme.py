"""The attention–memory family, written in the terms of the repaired IWME core.

With H the memory before the latest symbol s is written (s = +1 left, −1 right, 0 before the first symbol):

    memory        H′ = (1 − ρ) H + ρ s                                   the core's M
    next symbol   P(left) = σ( C_E(H) + (ρ s + bias) / T )               the core's F_E
    actuation     C_E(H) = κ (1 − ρ) H / T                               κ = 1 in the family

Three free parameters (ρ, T, bias). κ = 1 reproduces the host's `EMACascade` (h0 = 0): the logit is
(h + bias)/T with h the memory after the latest symbol. κ = 0 is hypothesis H1, C_E ≡ 0: the memory still
updates and no longer acts, and the law collapses to an order-1 Markov chain in scale (`markov_limit`).
So "attention–memory against an order-1 chain at three parameters each" is "memory actuation against its
absence at equal size".

The left probability is the attention weight of the host's two-point memory at T_attention = 2T, and the
conditional mean of the memory update is that memory's map Φ(h). `as_core` builds the host's `IWMECore`
from these maps; `core_rollout` runs it. Proposition P1 applies: on observables this is a hidden-state
model, and only the parametric restriction or an intervention on the memory can separate it from others.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .. import host
from .baseline import MarkovChain, _weights
from .cascade import Array, sigmoid, signs


@dataclass(frozen=True)
class AttentionCascade:
    """gain = 1/T and offset = bias/T. gain = 0 is the binomial limit (T = ∞). `coupling` is κ; fits keep it at 1."""

    rho: float
    gain: float
    offset: float
    coupling: float = 1.0

    @property
    def k(self) -> int:
        return 3

    @property
    def T(self) -> float:
        return math.inf if self.gain == 0.0 else 1.0 / self.gain

    @property
    def bias(self) -> float:
        return 0.0 if self.gain == 0.0 else self.offset / self.gain

    @classmethod
    def from_host(cls, c: host.EMACascade) -> "AttentionCascade":
        if c.h0 != 0.0:
            raise ValueError("the family fixes h0 = 0")
        return cls(rho=c.rho, gain=1.0 / c.T, offset=c.bias / c.T)

    def as_host(self) -> host.EMACascade:
        if self.gain <= 0.0 or self.coupling != 1.0:
            raise ValueError("only κ = 1 with finite temperature is a host EMACascade")
        return host.EMACascade(rho=self.rho, T=self.T, bias=self.bias, h0=0.0)

    def actuation(self, memory: Array) -> Array:
        """C_E(H): what the memory sends to the dynamics, in logit units."""
        return self.coupling * (1.0 - self.rho) * self.gain * np.asarray(memory, dtype=float)

    def cond_left(self, strings: Array) -> Array:
        s = signs(strings)
        out = np.empty(s.shape)
        memory = np.zeros(s.shape[0])            # H: before the latest symbol is written
        latest = np.zeros(s.shape[0])            # s: the latest symbol, 0 before the first
        for pos in range(s.shape[1]):
            out[:, pos] = sigmoid(self.actuation(memory) + self.gain * self.rho * latest + self.offset)
            memory = (1.0 - self.rho) * memory + self.rho * latest
            latest = s[:, pos]
        return out

    # handles used by the registered interventions
    def with_temperature_scaled(self, factor: float) -> "AttentionCascade":
        """T → factor · T with ρ, bias and κ held fixed."""
        return AttentionCascade(self.rho, self.gain / factor, self.offset / factor, self.coupling)

    def with_rho_scaled(self, factor: float, cap: float = 0.98) -> "AttentionCascade":
        return AttentionCascade(min(cap, self.rho * factor), self.gain, self.offset, self.coupling)

    def with_coupling(self, coupling: float) -> "AttentionCascade":
        """κ → coupling: κ = 0 severs the actuation (H1)."""
        return AttentionCascade(self.rho, self.gain, self.offset, coupling)


def memory_trace(strings: Array, rho: float) -> Array:
    """The cascade memory h before each symbol (all earlier symbols written), shape (strings, n)."""
    s = signs(strings)
    out = np.empty(s.shape)
    h = np.zeros(s.shape[0])
    for pos in range(s.shape[1]):
        out[:, pos] = h
        h = (1.0 - rho) * h + rho * s[:, pos]
    return out


def markov_limit(model: AttentionCascade) -> MarkovChain:
    """The order-1 chain equal to the law at κ = 0: only the latest symbol acts."""
    start = float(sigmoid(np.array(model.offset)))
    after_left = float(sigmoid(np.array(model.gain * model.rho + model.offset)))
    after_right = float(sigmoid(np.array(-model.gain * model.rho + model.offset)))
    return MarkovChain(1, (np.array([start]), np.array([after_left, after_right])))


# ----------------------------------------------------------------------------------------------
# The same model as an instance of the host's repaired core
# ----------------------------------------------------------------------------------------------

def as_core(model: AttentionCascade) -> host.IWMECore:
    """The host's IWMECore with this family's maps. State X is the latest symbol (+1, −1, or 0 before the
    first), H the memory before it is written, and the physics noise is one uniform per step."""
    attention = None if model.gain <= 0.0 else host.two_point_memory(2.0 / model.gain, model.rho)

    def weights(H, Y, E):                         # attention over the two branches for the next symbol
        if attention is None:
            return np.array([0.5, 0.5])
        return attention.weights((1.0 - model.rho) * np.atleast_1d(H) + model.rho * np.atleast_1d(Y),
                                 np.array([model.bias]))

    return host.IWMECore(
        Y_E=lambda X, Yext, noise: X,
        A=weights,
        M=lambda H, A, Y, noise: (1.0 - model.rho) * H + model.rho * Y,
        F_E=lambda X, U, noise, act: np.where(noise < sigmoid(act + model.gain * model.rho * X + model.offset), 1.0, -1.0),
        C_E=model.actuation,
        T=lambda X: X,
    )


def core_rollout(model: AttentionCascade, uniforms: Array) -> Array:
    """Run the host core on rows of uniforms; returns the symbol strings it emits (0 left, 1 right)."""
    uniforms = np.atleast_2d(np.asarray(uniforms, dtype=float))
    core = as_core(model)
    n = uniforms.shape[1]
    quiet = np.zeros(1)
    inputs = [{"U": quiet, "E": quiet, "Yext": quiet}] * (n + 1)
    out = np.empty(uniforms.shape, dtype=np.uint8)
    for row, u in enumerate(uniforms):
        noises = [{"x": np.array([u[t]]), "y": quiet, "h": quiet} for t in range(n)] + [{"x": quiet, "y": quiet, "h": quiet}]
        emitted = core.run(np.zeros(1), np.zeros(1), inputs, noises)      # G_t = X_t, starting with X_0 = 0
        out[row] = (emitted[1:, 0] < 0.0).astype(np.uint8)
    return out


def sequential_sample(model, uniforms: Array) -> Array:
    """Draw strings from any family by thresholding one uniform per level against cond_left."""
    uniforms = np.atleast_2d(np.asarray(uniforms, dtype=float))
    out = np.zeros(uniforms.shape, dtype=np.uint8)
    for pos in range(uniforms.shape[1]):
        out[:, pos] = (uniforms[:, pos] >= model.cond_left(out)[:, pos]).astype(np.uint8)
    return out


# ----------------------------------------------------------------------------------------------
# Fit
# ----------------------------------------------------------------------------------------------

RHO_GRID = np.linspace(0.02, 0.98, 49)
RIDGE = 1e-3


def _logistic(x: Array, y: Array, w: Array, ridge: float = RIDGE) -> Tuple[float, float, float]:
    """Maximize Σ w [y ln σ(a x + b) + (1 − y) ln(1 − σ(a x + b))] − (ridge/2)(a² + b²) over a ≥ 0, b.
    Returns (a, b, penalized objective). Newton with step halving; the objective is concave."""

    def objective(a: float, b: float) -> float:
        zed = a * x + b
        return float((w * (y * zed - np.logaddexp(0.0, zed))).sum() - 0.5 * ridge * (a * a + b * b))

    mean = float(np.clip((w * y).sum() / w.sum(), 1e-9, 1.0 - 1e-9))
    a, b = 0.0, math.log(mean / (1.0 - mean))
    best = objective(a, b)
    for _ in range(60):
        prob = sigmoid(a * x + b)
        resid = w * (y - prob)
        curv = w * prob * (1.0 - prob)
        grad = np.array([(resid * x).sum() - ridge * a, resid.sum() - ridge * b])
        hess = np.array([[(curv * x * x).sum() + ridge, (curv * x).sum()],
                         [(curv * x).sum(), curv.sum() + ridge]])
        delta = np.linalg.solve(hess, grad)
        scale, accepted = 1.0, False
        while scale > 1e-8:
            cand = objective(a + scale * delta[0], b + scale * delta[1])
            if cand >= best:
                accepted = True
                break
            scale *= 0.5
        if not accepted:
            break
        improvement = cand - best
        a, b, best = a + scale * delta[0], b + scale * delta[1], cand
        if improvement < 1e-12 * (1.0 + abs(best)):
            break
    if a < 0.0:                                   # outside the family: fall back to its binomial member
        a, b = 0.0, math.log(mean / (1.0 - mean))
        best = objective(a, b)
    return a, b, best


def _attention_profile(strings: Array, w: Array, rho: float) -> Tuple[float, float, float]:
    x = memory_trace(strings, rho).ravel()
    y = (np.asarray(strings) == 0).astype(float).ravel()
    return _logistic(x, y, np.repeat(w, strings.shape[1]))


def fit_attention(strings: Array, weights: Optional[Array] = None) -> AttentionCascade:
    """Maximum likelihood: logistic regression of the next symbol on the memory for each ρ of RHO_GRID,
    then golden-section refinement of ρ between the neighbours of the best grid value."""
    strings = np.asarray(strings)
    w = _weights(strings, weights)
    scores = [_attention_profile(strings, w, float(r))[2] for r in RHO_GRID]
    i = int(np.argmax(scores))
    lo, hi = float(RHO_GRID[max(i - 1, 0)]), float(RHO_GRID[min(i + 1, len(RHO_GRID) - 1)])
    inv_phi = (math.sqrt(5.0) - 1.0) / 2.0
    c, d = hi - inv_phi * (hi - lo), lo + inv_phi * (hi - lo)
    fc, fd = _attention_profile(strings, w, c)[2], _attention_profile(strings, w, d)[2]
    for _ in range(30):
        if fc > fd:
            hi, d, fd = d, c, fc
            c = hi - inv_phi * (hi - lo)
            fc = _attention_profile(strings, w, c)[2]
        else:
            lo, c, fc = c, d, fd
            d = lo + inv_phi * (hi - lo)
            fd = _attention_profile(strings, w, d)[2]
    rho = 0.5 * (lo + hi)
    gain, offset, top = _attention_profile(strings, w, rho)
    if scores[i] > top:                           # keep the grid point if refinement did not improve on it
        rho = float(RHO_GRID[i])
        gain, offset, _ = _attention_profile(strings, w, rho)
    return AttentionCascade(rho=rho, gain=gain, offset=offset)
