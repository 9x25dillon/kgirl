"""The repaired IWME core (Phase 3, priority 1) and the persistence-after-decoupling criterion (priority 2).

Repaired system (one causal order, loop closed, observation separated from actuation):

    Y_t       = Y_E(X_t, Y_t^ext, N_t^y)                interaction signal, now driven by the state
    A_t       = A(H_t, Y_t, E_t) ∈ Δ^{n−1}              attention
    H_{t+1}   = M(H_t, A_t, Y_t, N_t^h)                 memory
    X_{t+1}   = F_E(X_t, U_t, N_t^x, C_E(H_t))          physics; C_E is the actuation map (H1 ⇔ C_E ≡ 0)
    G_t^phys  = T_{E,λ}(X_t)                            physical observable
    G_t^rep   = T′_{E,λ}(X_t, H_t)                      observer representation (Π_{E,λ} lives here)

P1 (observational equivalence): the pair Z = (X, H) is a single Markov state, so the same observable law is
produced by a model with no separate memory variable (`IWMECore.augmented`). Evidence for IWME must therefore
come from parametric structure or from interventions on the memory substrate.

`persistence_information` computes, exactly, for a linear-Gaussian instance:

    R_persist(L) = I( G_{t_c+1 … t_c+L} ; Y^B_{< t_c} | X_{t_c} [, H_{t_c}] )      in bits

after the A–B channel is severed at t_c. It is 0 when C_E ≡ 0, positive when the memory acts on the
physics, and returns to 0 once H_{t_c} is conditioned on (state incompleteness is all it certifies). Its lag
profile separates an EMA memory (pole 1 − ρ: R → 0) from an accumulator (pole 1, the IRT shadow integrator:
R stays positive).

numpy only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

Array = np.ndarray


# ----------------------------------------------------------------------------------------------
# Generic repaired core
# ----------------------------------------------------------------------------------------------

@dataclass
class IWMECore:
    """Callables for the six maps; noise is passed in explicitly so runs are reproducible."""

    Y_E: Callable[[Array, Array, Array], Array]
    A: Callable[[Array, Array, Array], Array]
    M: Callable[[Array, Array, Array, Array], Array]
    F_E: Callable[[Array, Array, Array, Array], Array]
    C_E: Callable[[Array], Array]
    T: Callable[[Array], Array]
    T_rep: Optional[Callable[[Array, Array], Array]] = None

    def step(self, X: Array, H: Array, U: Array, E: Array, Yext: Array, noise: Dict[str, Array]):
        Y = self.Y_E(X, Yext, noise["y"])
        Att = self.A(H, Y, E)
        H_next = self.M(H, Att, Y, noise["h"])
        X_next = self.F_E(X, U, noise["x"], self.C_E(H))
        return X_next, H_next, {"Y": Y, "A": Att, "G": self.T(X),
                                "G_rep": None if self.T_rep is None else self.T_rep(X, H)}

    def run(self, X0: Array, H0: Array, inputs: Sequence[Dict[str, Array]], noises: Sequence[Dict[str, Array]]):
        X, H = np.asarray(X0, float), np.asarray(H0, float)
        out = []
        for inp, nz in zip(inputs, noises):
            X, H, obs = self.step(X, H, inp["U"], inp["E"], inp["Yext"], nz)
            out.append(obs["G"])
        return np.array(out)

    def augmented(self) -> Callable[[Array, Array, Array, Array, Dict[str, Array]], Tuple[Array, Array]]:
        """P1: one map Z ↦ Z′ on Z = (X, H) with no distinguished memory; outputs G = T(Z[:dx])."""
        def F_prime(Z: Array, dx: int, U: Array, E: Array, Yext: Array, noise: Dict[str, Array]):
            X, H = Z[:dx], Z[dx:]
            X2, H2, obs = self.step(X, H, U, E, Yext, noise)
            return np.concatenate([X2, H2]), obs["G"]
        return F_prime


# ----------------------------------------------------------------------------------------------
# Linear-Gaussian instance: exact conditional mutual information
# ----------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class LinearIWME:
    """Scalar plant and memory.

    t < t_c:  H_{t+1} = μ H_t + ρ Y^B_t        (μ = 1 − ρ for an EMA, μ = 1 for an accumulator)
              X_{t+1} = a X_t + c H_t + s_x ξ_t
    t ≥ t_c:  the same with Y^B ≡ 0 (channel severed)
    G_t = X_t + s_g ζ_t;   Y^B_t ~ N(0, s_y²) i.i.d.;   X_0 = H_0 = 0.
    """

    a: float = 0.8
    c: float = 0.5
    rho: float = 0.3
    memory: str = "ema"           # "ema" or "accumulator"
    s_x: float = 0.1
    s_g: float = 0.1
    s_y: float = 1.0

    @property
    def mu(self) -> float:
        if self.memory == "ema":
            return 1.0 - self.rho
        if self.memory == "accumulator":
            return 1.0
        raise ValueError("memory must be 'ema' or 'accumulator'")


def _linear_rows(m: LinearIWME, t_c: int, horizon: int):
    """Express X_t, H_t (t ≤ t_c + horizon) and G_t (t > t_c) as coefficient rows over independent unit
    Gaussian sources: [Y^B_0..Y^B_{t_c−1} | ξ_0..ξ_{T−1} | ζ_{t_c+1}..ζ_{t_c+horizon}]."""
    T = t_c + horizon
    nY, nX, nZ = t_c, T, horizon
    n = nY + nX + nZ
    X = np.zeros((T + 1, n))
    H = np.zeros((T + 1, n))
    for t in range(T):
        H[t + 1] = m.mu * H[t]
        if t < t_c:
            H[t + 1, t] += m.rho * m.s_y
        X[t + 1] = m.a * X[t] + m.c * H[t]
        X[t + 1, nY + t] += m.s_x
    G = np.zeros((horizon, n))
    for k in range(horizon):
        G[k] = X[t_c + 1 + k]
        G[k, nY + nX + k] += m.s_g
    Yrows = np.zeros((nY, n))
    Yrows[np.arange(nY), np.arange(nY)] = m.s_y
    return X, H, G, Yrows


def _logdet(S: Array) -> float:
    if S.size == 0:
        return 0.0
    sign, val = np.linalg.slogdet(S)
    if sign <= 0:
        raise np.linalg.LinAlgError("covariance not positive definite")
    return float(val)


def _mi_with(A: Array, Z: Array) -> float:
    """I(A; Z) in nats for Gaussian rows; Z may be rank-deficient (pseudo-inverse), A must be nondegenerate."""
    SA = A @ A.T
    if Z.shape[0] == 0:
        return 0.0
    SAZ = A @ Z.T
    SZ = Z @ Z.T
    cond = SA - SAZ @ np.linalg.pinv(SZ, rcond=1e-12, hermitian=True) @ SAZ.T
    return 0.5 * (_logdet(SA) - _logdet(0.5 * (cond + cond.T)))


def conditional_mi_bits(A: Array, B: Array, C: Array) -> float:
    """I(A; B | C) = I(A; B, C) − I(A; C) in bits, for jointly Gaussian variables given as coefficient rows
    over independent unit sources. B and C may be degenerate (e.g. C a function of B); A must carry its own
    noise so that Σ_A and Σ_{A|·} are positive definite."""
    A = np.atleast_2d(A)
    B = np.atleast_2d(B)
    C = np.atleast_2d(C) if len(C) else np.zeros((0, A.shape[1]))
    val = _mi_with(A, np.vstack([B, C])) - _mi_with(A, C)
    return max(0.0, val) / math.log(2.0)


def persistence_information(m: LinearIWME, t_c: int = 40, horizon: int = 30, condition: str = "X",
                            lags: Optional[Sequence[int]] = None) -> Dict[str, object]:
    """R_persist for the block G_{t_c+1..t_c+horizon} and per single lag ℓ (G_{t_c+ℓ} alone)."""
    X, H, G, Y = _linear_rows(m, t_c, horizon)
    cond = [X[t_c]] if condition == "X" else [X[t_c], H[t_c]]
    if condition not in ("X", "XH"):
        raise ValueError("condition must be 'X' or 'XH'")
    C = np.vstack(cond)
    if np.allclose(C, 0):
        C = np.zeros((0, G.shape[1]))
    block = conditional_mi_bits(G, Y, C)
    lags = list(range(1, horizon + 1)) if lags is None else list(lags)
    per_lag = [conditional_mi_bits(G[l - 1:l], Y, C) for l in lags]
    return {"block_bits": block, "lags": lags, "per_lag_bits": per_lag}
