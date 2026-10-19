#!/usr/bin/env python3
"""Minimal reference implementation of the Algorithmic Life Equation.
Companion to Minimal_Algorithmic_Life_Equation.pdf / .tex
18 October 2026
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional

@dataclass
class LifeParams:
    A_max: float = 1.0
    eta_in: float = 0.8
    a: float = 0.15          # organization growth
    b: float = 0.08          # organization decay
    C_min: float = 0.15
    C_max: float = 0.85
    C_star: float = 0.50
    eta_C: float = 0.12
    eta_V: float = 0.05
    eta_M: float = 0.05
    lambda_reg: float = 0.01
    rho: float = 0.2
    eta_R: float = 0.1
    K_F: float = 1.0
    energy_scale: float = 1.0
    w0: float = 0.01
    wA: float = 0.02
    wM: float = 0.001
    wu: float = 0.01
    delta0: float = 0.005
    deltaF: float = 0.01
    delta_eps: float = 0.02
    mem_len: int = 20
    feat_dim: int = 8

@dataclass
class LifeState:
    A: float = 0.5
    C: float = 0.5
    O: float = 0.5
    F: float = 2.0
    R: float = 0.3
    P: float = 1.0
    M: np.ndarray = field(default_factory=lambda: np.zeros((8, 8)))
    H: np.ndarray = field(default_factory=lambda: np.zeros((20, 8)))
    policy: np.ndarray = field(default_factory=lambda: np.ones(4) / 4)

def clip(x, lo, hi):
    return float(np.clip(x, lo, hi))

def feature_map(H: np.ndarray) -> np.ndarray:
    # simple summary features of recent history
    return np.concatenate([H.mean(axis=0)[:4], H.std(axis=0)[:4]])

class MinimalLife:
    def __init__(self, params: Optional[LifeParams] = None):
        self.p = params or LifeParams()
        self.state = LifeState()
        self.eps_history = []

    def predict(self) -> np.ndarray:
        phi = feature_map(self.state.H)
        return self.state.M @ phi

    def life_step(self, observation: np.ndarray, energy_input: float):
        p = self.p
        s = self.state

        # 1. Predict & error
        pred = self.predict()
        eps = observation - pred
        self.eps_history.append(eps)
        if len(self.eps_history) > 10:
            self.eps_history.pop(0)
        var_eps = np.var(self.eps_history) if self.eps_history else 0.0

        # 2. World-model update (gradient + complexity penalty)
        phi = feature_map(s.H)
        grad = -2 * np.outer(eps, phi) + 2 * p.lambda_reg * s.M
        s.M = s.M - p.eta_M * grad

        # 3. Memory update (ring buffer)
        s.H = np.roll(s.H, -1, axis=0)
        s.H[-1] = np.concatenate([observation[:4], eps[:4]]) if observation.size >= 4 else np.pad(observation, (0, 8-observation.size))

        # 4. Self-model strength
        # gain: reward when error is decreasing or energy is stable
        err_mag = float(np.linalg.norm(eps))
        prev_err = float(np.linalg.norm(self.eps_history[-2])) if len(self.eps_history) > 1 else err_mag
        G = 0.3 * np.tanh(prev_err - err_mag) + 0.1 * (1.0 - abs(s.F - 2.0) / 5.0)
        L = 0.05 * err_mag
        s.R = clip(s.R + p.eta_R * (G - L), 0.0, 1.0)

        # 5. Action selection (soft-max over discrete actions)
        logits = s.policy
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        action_idx = np.random.choice(len(probs), p=probs)
        u = np.zeros(len(probs)); u[action_idx] = 1.0

        # 6. Energy accounting
        W = p.w0 + p.wA * s.A + p.wM * np.linalg.norm(s.M) + p.wu * np.linalg.norm(u)
        D = p.delta0 + p.deltaF * s.F + p.delta_eps * np.linalg.norm(eps)**2
        s.F = max(0.0, s.F + p.eta_in * energy_input - W - D)

        # 7. Organization
        s.O = clip(
            s.O + p.a * s.F * s.C * (1 - s.O) - p.b * (D + np.linalg.norm(eps)**2) * s.O,
            0.0, 1.0
        )

        # 8. Coherence
        s.C = clip(
            s.C + p.eta_C * (p.C_star - s.C) - p.eta_V * var_eps,
            p.C_min + 1e-3, p.C_max - 1e-3
        )

        # 9. Potential (log of reachable discrete actions under energy budget)
        n_reach = max(1, int(4 * min(1.0, s.F / 2.0) * s.O * s.C))
        s.P = np.log1p(n_reach)

        # 10. Processing fraction (simple saturation)
        s.A = clip(s.A + 0.05 * (0.7 - s.A) - 0.02 * var_eps, 0.0, p.A_max)

        # Functional index
        C_eff = (s.A / p.A_max) * s.C * s.O * s.R * (s.F / (s.F + p.K_F))

        return s, action_idx, C_eff

# Example usage
if __name__ == "__main__":
    import numpy as np
    life = MinimalLife()
    for t in range(20):
        obs = np.random.randn(8) * 0.1
        E = 0.5 + 0.1 * np.sin(t / 5)
        state, action, Ceff = life.life_step(obs, E)
        print(f"t={t:02d}  F={state.F:.3f}  O={state.O:.3f}  C={state.C:.3f}  R={state.R:.3f}  C_eff={Ceff:.4f}")
