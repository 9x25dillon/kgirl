"""Scaling exponents of a series: DFA, MF-DFA and rescaled range. Replaces RVSE's MultiscaleAnalyzer.

    DFA      Peng et al. 1994. Profile Y = cumsum(x − x̄); cut into segments of length s from both ends;
             remove an order-m polynomial per segment; F(s) = RMS residual ~ s^h.
             h ≈ 0.5 white noise, > 0.5 persistent, < 0.5 anti-persistent, ≈ 1.5 for a random walk.
    MF-DFA   Kantelhardt et al. 2002. F_q(s) = (mean_ν F²(ν,s)^{q/2})^{1/q} ~ s^{h(q)}  (q = 0: geometric mean).
             τ(q) = q·h(q) − 1,  α = τ′(q),  f(α) = qα − τ.  The width Δα = α_max − α_min measures how far a
             series is from monofractal. For the binomial cascade the exact answer is
             τ(q) = −log₂(pᵠ + (1−p)ᵠ) (kgirl.iwme.cascade.binomial_tau). Measured on 2¹⁶ points, p = 0.3,
             q ∈ [−4, 4]: with dyadic scales, the estimated Δα is 1.1239 and the exact value is 1.1239.
             h(q) carries a constant offset of −0.042 from linear detrending, which cancels in Δα and in
             h(q) − h(2). Log-spaced scales cut across dyadic structure (Δα 1.176, nonlinear error), so use
             `dyadic=True` for cascade-like data.
    R/S      Hurst 1951 rescaled range, kept from the original with log-spaced scales. It is biased upward on
             short windows (Anis & Lloyd 1976); prefer DFA.

What the original got wrong (kept in research/resonance/originals): it regressed R/S on unit-spaced scales
(weighting the shortest, most biased windows most heavily), and its "multifractal spectrum" took
log mean |Δx|^q / (q log n), which is not a scaling exponent of anything.

Complexity: O(N · |scales|) per q after one QR per scale; vectorized over segments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


def log_scales(n: int, min_scale: int = 10, max_frac: float = 0.25, count: int = 20, dyadic: bool = False) -> np.ndarray:
    hi = max(min_scale + 1, int(n * max_frac))
    if dyadic:
        s = 2 ** np.arange(int(np.ceil(np.log2(min_scale))), int(np.floor(np.log2(hi))) + 1)
        return s.astype(int)
    s = np.unique(np.round(np.exp(np.linspace(np.log(min_scale), np.log(hi), count))).astype(int))
    return s[s >= min_scale]


def segment_variances(x: np.ndarray, scales: Sequence[int], order: int = 1) -> dict[int, np.ndarray]:
    """F²(ν, s) for every segment ν (cut from the start and from the end) at every scale s."""
    x = np.asarray(x, dtype=np.float64)
    y = np.cumsum(x - x.mean())
    n = len(y)
    out = {}
    for s in scales:
        k = n // s
        if k < 1 or s <= order + 1:
            continue
        segs = np.concatenate([y[:k * s].reshape(k, s), y[n - k * s:].reshape(k, s)])
        Q, _ = np.linalg.qr(np.vander(np.arange(s, dtype=np.float64) / s, order + 1))
        resid = segs - (segs @ Q) @ Q.T
        out[int(s)] = np.maximum((resid ** 2).mean(axis=1), 1e-300)
    return out


@dataclass(frozen=True)
class MFDFA:
    q: np.ndarray
    h: np.ndarray
    scales: np.ndarray
    Fq: np.ndarray                     # |q| × |scales|

    @property
    def tau(self) -> np.ndarray:
        return self.q * self.h - 1.0

    @property
    def alpha(self) -> np.ndarray:
        return np.gradient(self.tau, self.q)

    @property
    def f(self) -> np.ndarray:
        return self.q * self.alpha - self.tau

    @property
    def width(self) -> float:
        a = self.alpha
        return float(a.max() - a.min())

    def h_at(self, q: float) -> float:
        i = int(np.argmin(np.abs(self.q - q)))
        return float(self.h[i])

    @property
    def hurst(self) -> float:
        return self.h_at(2.0)

    def summary(self) -> dict:
        return {"hurst": round(self.hurst, 4), "width": round(self.width, 4),
                "h": {float(q): round(float(h), 4) for q, h in zip(self.q, self.h)},
                "scales": [int(self.scales[0]), int(self.scales[-1])]}


def mfdfa(x: np.ndarray, q: Optional[Sequence[float]] = None, *, order: int = 1,
          scales: Optional[Sequence[int]] = None, min_scale: int = 10, dyadic: bool = False) -> MFDFA:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or len(x) < 4 * min_scale:
        raise ValueError(f"MF-DFA needs a 1-D series of at least {4 * min_scale} samples")
    if not np.all(np.isfinite(x)):
        raise ValueError("series contains non-finite values")
    qs = np.asarray(q if q is not None else np.arange(-4.0, 4.01, 0.5), dtype=np.float64)
    sc = np.asarray(scales if scales is not None else log_scales(len(x), max(min_scale, order + 2), dyadic=dyadic))
    F2 = segment_variances(x, sc, order)
    sc = np.array(sorted(F2))
    if len(sc) < 3:
        raise ValueError("too few usable scales")
    Fq = np.empty((len(qs), len(sc)))
    for j, s in enumerate(sc):
        v = F2[s]
        for i, qq in enumerate(qs):
            Fq[i, j] = np.exp(0.5 * np.mean(np.log(v))) if abs(qq) < 1e-12 else np.mean(v ** (qq / 2)) ** (1 / qq)
    ls = np.log(sc)
    h = np.array([np.polyfit(ls, np.log(Fq[i]), 1)[0] for i in range(len(qs))])
    return MFDFA(qs, h, sc, Fq)


def dfa(x: np.ndarray, order: int = 1, min_scale: int = 10) -> float:
    """DFA scaling exponent h(2)."""
    return mfdfa(x, [2.0], order=order, min_scale=min_scale).hurst


def rescaled_range(x: np.ndarray, min_scale: int = 10, count: int = 16) -> float:
    """Classic R/S Hurst exponent on log-spaced, non-overlapping windows (upward-biased on short windows)."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 4 * min_scale:
        raise ValueError(f"R/S needs at least {4 * min_scale} samples")
    sc, rs = [], []
    for s in log_scales(len(x), min_scale, 0.5, count):
        k = len(x) // s
        seg = x[:k * s].reshape(k, s)
        dev = np.cumsum(seg - seg.mean(axis=1, keepdims=True), axis=1)
        r, sd = dev.max(axis=1) - dev.min(axis=1), seg.std(axis=1)
        ok = sd > 0
        if ok.any():
            sc.append(s)
            rs.append(np.mean(r[ok] / sd[ok]))
    if len(sc) < 2:
        return 0.5
    return float(np.polyfit(np.log(sc), np.log(rs), 1)[0])


def trace_dimension(x: np.ndarray) -> float:
    """Box dimension D = 2 − H of the trace of a self-affine series whose increments are `x`.

    Valid only when the DFA exponent of the increments lies in (0, 1); NaN otherwise."""
    h = dfa(x)
    return float(2.0 - h) if 0.0 < h < 1.0 else float("nan")
