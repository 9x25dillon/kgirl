"""The conventional families: no memory, a window over the last m scales, and a generic hidden state.

    Binomial          one probability                                                    k = 1
    MarkovChain(m)    a window over the last m scales, with a full set of start contexts k = 2^(m+1) − 1
    HiddenMarkov      the host's MarkovCascade read as a law on strings: two hidden states k = 7

Every fit takes weighted strings: unit weights fit a sample, probability weights project onto an exact law.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import numpy as np

from .. import host, streams
from .cascade import Array, sigmoid


@dataclass(frozen=True)
class Binomial:
    p: float

    @property
    def k(self) -> int:
        return 1

    def cond_left(self, strings: Array) -> Array:
        return np.full(np.shape(strings), float(self.p))


@dataclass(frozen=True)
class MarkovChain:
    """tables[j][ctx] = P(left given the last j symbols), j = 0 … order. ctx reads those symbols as a binary
    number with the oldest symbol most significant, so index 0 is L…L and the last index is R…R."""

    order: int
    tables: Tuple[Array, ...]

    def __post_init__(self) -> None:
        if len(self.tables) != self.order + 1 or any(len(t) != (1 << j) for j, t in enumerate(self.tables)):
            raise ValueError("need one table of length 2^j for each context length j = 0 … order")

    @property
    def k(self) -> int:
        return (1 << (self.order + 1)) - 1

    def cond_left(self, strings: Array) -> Array:
        strings = np.asarray(strings)
        out = np.empty(strings.shape)
        ctx = np.zeros(strings.shape[0], dtype=np.int64)
        for pos in range(strings.shape[1]):
            out[:, pos] = self.tables[min(pos, self.order)][ctx]
            ctx = ((ctx << 1) + strings[:, pos]) & ((1 << min(pos + 1, self.order)) - 1)
        return out


@dataclass(frozen=True)
class HiddenMarkov:
    """The host's MarkovCascade family read as a law on strings: in state s the left child has probability
    p[s]; after taking child c the state moves to s′ with probability P[s][c][s′]."""

    cascade: host.MarkovCascade

    @property
    def k(self) -> int:
        return 7

    def cond_left(self, strings: Array) -> Array:
        strings = np.asarray(strings)
        p = np.asarray(self.cascade.p, dtype=float)
        trans = np.asarray(self.cascade.P, dtype=float)                    # [s][c][s′]
        out = np.empty(strings.shape)
        f = np.tile(np.asarray(self.cascade.pi0, dtype=float), (strings.shape[0], 1))
        for pos in range(strings.shape[1]):
            out[:, pos] = f @ p
            c = strings[:, pos]
            g = f * np.where(c[:, None] == 0, p[None, :], 1.0 - p[None, :])
            g /= g.sum(axis=1, keepdims=True)
            f = np.einsum("is,isj->ij", g, np.transpose(trans[:, c, :], (1, 0, 2)))
        return out


def markov_from_table(order: int, tables: Sequence[Sequence[float]]) -> MarkovChain:
    return MarkovChain(order, tuple(np.asarray(t, dtype=float) for t in tables))


# ----------------------------------------------------------------------------------------------
# Fits (weighted strings)
# ----------------------------------------------------------------------------------------------

def _weights(strings: Array, weights: Optional[Array]) -> Array:
    return np.ones(strings.shape[0]) if weights is None else np.asarray(weights, dtype=float)


def fit_binomial(strings: Array, weights: Optional[Array] = None) -> Binomial:
    """Krichevsky–Trofimov estimate (add ½)."""
    w = _weights(strings, weights)
    left = float((w[:, None] * (np.asarray(strings) == 0)).sum())
    total = float(w.sum() * strings.shape[1])
    return Binomial((left + 0.5) / (total + 1.0))


def fit_markov(strings: Array, order: int, weights: Optional[Array] = None) -> MarkovChain:
    """Krichevsky–Trofimov estimate of every conditional probability, start contexts included."""
    strings = np.asarray(strings)
    w = _weights(strings, weights)
    left = [np.zeros(1 << j) for j in range(order + 1)]
    total = [np.zeros(1 << j) for j in range(order + 1)]
    ctx = np.zeros(strings.shape[0], dtype=np.int64)
    for pos in range(strings.shape[1]):
        j = min(pos, order)
        total[j] += np.bincount(ctx, weights=w, minlength=1 << j)
        left[j] += np.bincount(ctx, weights=w * (strings[:, pos] == 0), minlength=1 << j)
        ctx = ((ctx << 1) + strings[:, pos]) & ((1 << min(pos + 1, order)) - 1)
    return MarkovChain(order, tuple((l + 0.5) / (t + 1.0) for l, t in zip(left, total)))


def _hmm_expectations(strings: Array, w: Array, p: Array, trans: Array, pi0: Array):
    """Scaled forward–backward for the two-state cascade, vectorized over strings, with the 2 × 2 products
    written out. Returns the weighted log-likelihood (nats) and the expected counts that the M-step and the
    gradient both need: occupancy[s], left emissions[s], transitions[s][c][s′], first-state occupancy[s],
    total weight."""
    n_str, n = strings.shape
    left = strings == 0
    emit0 = np.where(left, p[0], 1.0 - p[0])                 # emission probability in state 0, (strings, n)
    emit1 = np.where(left, p[1], 1.0 - p[1])
    t00, t01 = trans[0, strings, 0], trans[0, strings, 1]    # P[s][c_k][s′] along each string, (strings, n)
    t10, t11 = trans[1, strings, 0], trans[1, strings, 1]

    g0, g1 = np.empty((n_str, n)), np.empty((n_str, n))      # filtered state probabilities
    z = np.empty((n_str, n))                                 # predictive probability of each symbol
    f0, f1 = np.full(n_str, pi0[0]), np.full(n_str, pi0[1])
    for pos in range(n):
        a0, a1 = f0 * emit0[:, pos], f1 * emit1[:, pos]
        zed = a0 + a1
        z[:, pos] = zed
        g0[:, pos], g1[:, pos] = a0 / zed, a1 / zed
        f0 = g0[:, pos] * t00[:, pos] + g1[:, pos] * t10[:, pos]
        f1 = g0[:, pos] * t01[:, pos] + g1[:, pos] * t11[:, pos]

    occ, occ_left, first = np.zeros(2), np.zeros(2), np.zeros(2)
    moves = np.zeros((2, 2, 2))
    b0, b1 = np.ones(n_str), np.ones(n_str)
    for pos in range(n - 1, -1, -1):
        if pos < n - 1:
            e0 = emit0[:, pos + 1] * b0 / z[:, pos + 1]
            e1 = emit1[:, pos + 1] * b1 / z[:, pos + 1]
            x00, x01 = g0[:, pos] * t00[:, pos] * e0, g0[:, pos] * t01[:, pos] * e1
            x10, x11 = g1[:, pos] * t10[:, pos] * e0, g1[:, pos] * t11[:, pos] * e1
            for c in (0, 1):
                wc = w * (strings[:, pos] == c)
                moves[0, c, 0] += wc @ x00
                moves[0, c, 1] += wc @ x01
                moves[1, c, 0] += wc @ x10
                moves[1, c, 1] += wc @ x11
            b0, b1 = t00[:, pos] * e0 + t01[:, pos] * e1, t10[:, pos] * e0 + t11[:, pos] * e1
        gam0, gam1 = g0[:, pos] * b0, g1[:, pos] * b1
        wl = w * left[:, pos]
        occ += (w @ gam0, w @ gam1)
        occ_left += (wl @ gam0, wl @ gam1)
        if pos == 0:
            first = np.array([w @ gam0, w @ gam1])

    loglik = float(w @ np.log(z).sum(axis=1))
    return loglik, occ, occ_left, moves, first, float(w.sum())


def _hmm_em_step(strings: Array, w: Array, p: Array, trans: Array, pi0: Array):
    """One EM step (posterior mode with pseudo-count ½ on every expected count). Returns the new
    (p, trans, pi0) and the log-likelihood of the parameters passed in."""
    loglik, occ, occ_left, moves, first, total = _hmm_expectations(strings, w, p, trans, pi0)
    p_new = (occ_left + 0.5) / (occ + 1.0)
    trans_new = (moves + 0.5) / (moves.sum(axis=2, keepdims=True) + 1.0)
    pi_new = (first + 0.5) / (total + 1.0)
    return p_new, trans_new, pi_new, loglik


LOGIT_CAP = 27.0            # σ(27) = 1 − 1.9e−12: keeps every probability strictly inside (0, 1)


def _hmm_unpack(theta: Array) -> Tuple[Array, Array, Array]:
    q = sigmoid(np.clip(theta, -LOGIT_CAP, LOGIT_CAP))
    to_zero = q[2:6].reshape(2, 2)                                   # P[s][c][0]
    trans = np.stack([to_zero, 1.0 - to_zero], axis=2)
    return q[0:2], trans, np.array([q[6], 1.0 - q[6]])


def _hmm_pack(p: Array, trans: Array, pi0: Array) -> Array:
    q = np.clip(np.concatenate([p, trans[:, :, 0].ravel(), pi0[:1]]), 1e-9, 1.0 - 1e-9)
    return np.log(q / (1.0 - q))


def hmm_objective(theta: Array, strings: Array, w: Array) -> Tuple[float, Array]:
    """The objective EM climbs, as a function of the seven logits, with its exact gradient:
    weighted log-likelihood + ½ Σ [ln q + ln(1 − q)] over the seven probabilities."""
    p, trans, pi0 = _hmm_unpack(theta)
    with np.errstate(all="ignore"):
        loglik, occ, occ_left, moves, first, total = _hmm_expectations(strings, w, p, trans, pi0)
    if not math.isfinite(loglik):                 # a string of probability 0 under these parameters
        return -math.inf, np.zeros(7)
    q = np.concatenate([p, trans[:, :, 0].ravel(), pi0[:1]])
    value = loglik + 0.5 * float((np.log(q) + np.log(1.0 - q)).sum())
    data_grad = np.concatenate([
        occ_left - occ * p,
        (moves[:, :, 0] - moves.sum(axis=2) * trans[:, :, 0]).ravel(),
        [first[0] - total * pi0[0]],
    ])
    return value, data_grad + 0.5 * (1.0 - 2.0 * q)


def _bfgs_maximize(fun: Callable[[Array], Tuple[float, Array]], theta: Array, curvature: float,
                   max_iter: int = 400, tol: float = 1e-12) -> Tuple[Array, float]:
    """Quasi-Newton ascent with Armijo backtracking. `curvature` sets the first step (typical second
    derivative of the objective: the total weight of the data)."""
    dim = theta.size
    eye = np.eye(dim)
    h_inv = eye / curvature
    value, grad = fun(theta)
    for _ in range(max_iter):
        direction = h_inv @ grad
        slope = float(grad @ direction)
        if slope <= 0.0:
            h_inv = eye / curvature
            direction, slope = grad / curvature, float(grad @ grad) / curvature
        step = 1.0
        while True:
            cand = theta + step * direction
            cand_value, cand_grad = fun(cand)
            if np.isfinite(cand_value) and cand_value >= value + 1e-4 * step * slope:
                break
            step *= 0.5
            if step < 1e-12:
                return theta, value
        moved, bent = cand - theta, grad - cand_grad
        done = abs(cand_value - value) <= tol * (1.0 + abs(value))
        theta, value, grad = cand, cand_value, cand_grad
        if done:
            break
        sy = float(moved @ bent)
        if sy > 1e-12 * float(np.linalg.norm(moved) * np.linalg.norm(bent)):
            v = eye - np.outer(moved, bent) / sy
            h_inv = v @ h_inv @ v.T + np.outer(moved, moved) / sy
    return theta, value


def fit_hidden_markov(strings: Array, weights: Optional[Array] = None, restarts: int = 4, warmup: int = 25,
                      kind: str = "chaos") -> HiddenMarkov:
    """Posterior mode with pseudo-count ½ on every expected count (Amendment 1). Each of `restarts`
    deterministic starts (stream purpose `hmm_init`) runs `warmup` EM steps and then quasi-Newton ascent on
    the same objective to convergence; the best objective is kept. Plain EM needs on the order of a thousand
    steps on this family."""
    strings = np.asarray(strings)
    w = _weights(strings, weights)
    curvature = float(w.sum() * strings.shape[1])
    best: Optional[Tuple[float, Array]] = None
    for r in range(restarts):
        rng = streams.stream(kind, "hmm_init", 0, r)
        p = 0.2 + 0.6 * rng.uniform(2)
        trans = 0.2 + rng.uniform((2, 2, 2))
        trans /= trans.sum(axis=2, keepdims=True)
        pi0 = np.array([0.5, 0.5])
        for _ in range(warmup):
            p, trans, pi0, _ = _hmm_em_step(strings, w, p, trans, pi0)
        theta, value = _bfgs_maximize(lambda t: hmm_objective(t, strings, w), _hmm_pack(p, trans, pi0), curvature)
        if best is None or value > best[0]:
            best = (value, theta)
    return hidden_markov_from_arrays(*_hmm_unpack(best[1]))


def hidden_markov_from_arrays(p: Array, trans: Array, pi0: Array) -> HiddenMarkov:
    return HiddenMarkov(host.MarkovCascade(
        p=(float(p[0]), float(p[1])),
        P=tuple(tuple((float(trans[s, c, 0]), float(trans[s, c, 1])) for c in (0, 1)) for s in (0, 1)),
        pi0=(float(pi0[0]), float(pi0[1])),
    ))
