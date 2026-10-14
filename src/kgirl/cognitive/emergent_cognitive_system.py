"""Emergent cognitive system: the LiMp module kgirl imported but never shipped, rebuilt from its call sites.

`cognitive_integration_bridge` and `advanced_cognitive_enhancements` imported

    EmergentCognitiveOrchestrator, QuantumOptimizationStep, SwarmCognitiveStep, NeuromorphicStep, HolographicStep

from a local LiMp checkout that was never pushed. Here each step is a small, measurable dynamical system,
driven by logistic chaos (no PRNG):

    QuantumOptimizationStep(n_qubits)   annealed Gibbs register over 2ⁿ basis states: p_T(s) ∝ exp(−c(s)/T),
                                        T: 1 → T_min geometrically; ψ = √p. Reports entropy per step and the
                                        ground-state probability (exact, no sampling noise).
    SwarmCognitiveStep(n_agents, search_dim, search_bounds)
                                        particle swarm (inertia 0.72, c₁ = c₂ = 1.49; Clerc–Kennedy) on an
                                        objective; reports the best cost curve and swarm diversity.
    NeuromorphicStep(n_neurons, dt)     Izhikevich regular-spiking network with small-world coupling;
                                        reports spike counts, rate entropy and branching ratio σ
                                        (σ ≈ 1 is the critical regime).
    HolographicStep(dim)                HolographicDataEngine (kgirl.cognitive.emergent_cognitive_network):
                                        stores the input, recalls it from its first half; fidelity is the
                                        correlation on the hidden half.

execute_cognitive_cycle(x) runs all four on one input. The target cost of the quantum register, the swarm's
target point and the neurons' drive all come from x. Emergence metrics are measured quantities in [0, 1]:

    quantum_order        1 − H(p_final) / n ln 2        how far annealing concentrated the register
    swarm_convergence    1 − diversity_end / diversity_start
    neural_complexity    normalised entropy of the spike-count distribution
    criticality          exp(−|ln σ|)                   1 at σ = 1
    holographic_fidelity max(0, corr) on the recalled half
    emergence_level      mean of the five

Evidence level: the dynamics are standard (L0). Reading their combination as "emergence" is a modelling
choice (L2).
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from kgirl.evolve.chaos import ChaosGenerator

try:                                     # callers pass torch tensors; torch itself is optional here
    import torch
except ImportError:                      # pragma: no cover
    torch = None


def _as_vector(x: Any) -> np.ndarray:
    if torch is not None and isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    v = np.asarray(x, dtype=np.float64).ravel()
    return v if v.size else np.zeros(1)


class QuantumOptimizationStep:
    def __init__(self, n_qubits: int = 4, steps: int = 32, t_min: float = 0.02):
        self.n_qubits, self.steps, self.t_min = n_qubits, steps, t_min

    def __call__(self, costs: np.ndarray) -> Dict[str, Any]:
        n = 2 ** self.n_qubits
        c = np.resize(np.asarray(costs, dtype=np.float64), n)
        c = (c - c.min()) / (np.ptp(c) or 1.0)                      # costs in [0, 1]
        temps = np.geomspace(1.0, self.t_min, self.steps)
        ent = []
        for T in temps:
            w = np.exp(-(c - c.min()) / T)
            p = w / w.sum()
            ent.append(float(-(p[p > 0] * np.log(p[p > 0])).sum()))
        best = int(np.argmin(c))
        return {"state": np.sqrt(p), "probabilities": p, "entropy_trajectory": ent, "entropy": ent[-1],
                "ground_state": best, "ground_state_probability": float(p[best]),
                "bitstring": format(best, f"0{self.n_qubits}b")}


class SwarmCognitiveStep:
    def __init__(self, n_agents: int = 20, search_dim: int = 4, search_bounds: Tuple[float, float] = (-1.0, 1.0),
                 iterations: int = 60, seed: float = 0.4123):
        self.n, self.d, self.bounds, self.iters = n_agents, search_dim, search_bounds, iterations
        self.rng = ChaosGenerator(seed=seed, lanes=64)

    def __call__(self, objective: Callable[[np.ndarray], np.ndarray]) -> Dict[str, Any]:
        lo, hi = self.bounds
        x = lo + (hi - lo) * self.rng.random((self.n, self.d))
        v = 0.1 * (hi - lo) * (self.rng.random((self.n, self.d)) - 0.5)
        f = objective(x)
        pbest, pf = x.copy(), f.copy()
        g = int(np.argmin(pf))
        curve = [float(pf[g])]
        div0 = float(np.mean(np.linalg.norm(x - x.mean(0), axis=1))) or 1e-12
        for _ in range(self.iters):
            r1, r2 = self.rng.random((self.n, self.d)), self.rng.random((self.n, self.d))
            v = 0.72 * v + 1.49 * r1 * (pbest - x) + 1.49 * r2 * (pbest[g] - x)
            x = np.clip(x + v, lo, hi)
            f = objective(x)
            better = f < pf
            pbest[better], pf[better] = x[better], f[better]
            g = int(np.argmin(pf))
            curve.append(float(pf[g]))
        div = float(np.mean(np.linalg.norm(x - x.mean(0), axis=1)))
        return {"best_position": pbest[g].tolist(), "best_cost": float(pf[g]), "cost_curve": curve,
                "diversity_start": div0, "diversity_end": div, "convergence": float(max(0.0, 1.0 - div / div0))}


class NeuromorphicStep:
    def __init__(self, n_neurons: int = 30, dt: float = 0.5, steps: int = 200, coupling: float = 6.0,
                 seed: float = 0.2357):
        self.n, self.dt, self.steps = n_neurons, dt, steps
        rng = ChaosGenerator(seed=seed, lanes=64)
        w = rng.normal(0.0, 0.05, (n_neurons, n_neurons))
        for i in range(n_neurons):                                   # ring lattice of 4 neighbours
            for k in (-2, -1, 1, 2):
                w[i, (i + k) % n_neurons] += coupling / 4
        np.fill_diagonal(w, 0.0)
        self.w = w

    def __call__(self, drive: np.ndarray) -> Dict[str, Any]:
        I = np.resize(np.asarray(drive, dtype=np.float64), self.n)
        v = np.full(self.n, -65.0)
        u = 0.2 * v
        spikes = np.zeros(self.n, bool)
        counts = np.zeros(self.n)
        per_step = []
        for _ in range(self.steps):
            syn = self.w.T @ spikes.astype(float)
            v = v + self.dt * (0.04 * v * v + 5 * v + 140 - u + I + syn)
            u = u + self.dt * 0.02 * (0.2 * v - u)
            spikes = v >= 30.0
            v[spikes], u[spikes] = -65.0, u[spikes] + 8.0
            counts += spikes
            per_step.append(int(spikes.sum()))
        a = np.asarray(per_step, dtype=np.float64)
        prev, nxt = a[:-1], a[1:]
        sigma = float(nxt[prev > 0].sum() / prev[prev > 0].sum()) if (prev > 0).any() else 0.0
        tot = counts.sum()
        if tot > 0 and self.n > 1:
            p = counts[counts > 0] / tot
            ent = float(-(p * np.log(p)).sum() / math.log(self.n))
        else:
            ent = 0.0
        return {"spike_counts": counts.tolist(), "mean_rate": float(counts.mean() / self.steps),
                "rate_entropy": ent, "branching_ratio": sigma, "spikes_per_step": per_step}


class HolographicStep:
    def __init__(self, dim: int = 8):
        from kgirl.cognitive.emergent_cognitive_network import HolographicDataEngine
        self.engine = HolographicDataEngine(data_dim=dim)
        self.dim = dim

    def __call__(self, data: np.ndarray) -> Dict[str, Any]:
        n = self.dim * self.dim
        x = np.resize(np.asarray(data, dtype=np.float64), n).reshape(self.dim, self.dim)
        self.engine.encode_holographic(x)
        part = x.copy()
        hidden = np.zeros_like(x, bool)
        hidden.flat[n // 2:] = True
        part[hidden] = np.nan
        rec = self.engine.recall_holographic(part, iterations=8)
        a, b = rec[hidden], x[hidden]
        fid = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float(np.allclose(a, b))
        return {"fidelity": fid, "mse_hidden": float(np.mean((a - b) ** 2)), "stored": len(self.engine._patterns)}


class EmergentCognitiveOrchestrator:
    def __init__(self, n_qubits: int = 4, n_agents: int = 20, search_dim: int = 4, n_neurons: int = 30,
                 holo_dim: int = 8):
        self.quantum = QuantumOptimizationStep(n_qubits)
        self.swarm = SwarmCognitiveStep(n_agents, search_dim)
        self.neural = NeuromorphicStep(n_neurons)
        self.holo = HolographicStep(holo_dim)
        self.cycles = 0
        self.history: list = []

    def execute_cognitive_cycle(self, x: Any) -> Dict[str, Any]:
        v = _as_vector(x)
        z = (v - v.mean()) / (v.std() or 1.0)
        q = self.quantum(-z)                                         # high input values = low cost
        target = np.tanh(np.resize(z, self.swarm.d))
        s = self.swarm(lambda P: np.sum((P - target) ** 2, axis=1))
        n = self.neural(5.0 + 5.0 * np.tanh(z))                      # drive around the RS threshold
        h = self.holo(z)
        nq = self.quantum.n_qubits
        metrics = {"quantum_order": float(1.0 - q["entropy"] / (nq * math.log(2))),
                   "swarm_convergence": s["convergence"], "neural_complexity": n["rate_entropy"],
                   "criticality": float(math.exp(-abs(math.log(n["branching_ratio"])))) if n["branching_ratio"] > 0 else 0.0,
                   "holographic_fidelity": max(0.0, h["fidelity"])}
        metrics["emergence_level"] = float(np.mean(list(metrics.values())))
        self.cycles += 1
        self.history.append(metrics)
        return {"quantum_state": q["state"].tolist(), "quantum_results": {k: v for k, v in q.items() if k != "state"},
                "swarm_results": s, "neural_results": n, "holographic_results": h, "emergence_metrics": metrics,
                "cycle": self.cycles}

    def emergence_trend(self) -> Optional[float]:
        if len(self.history) < 2:
            return None
        e = [m["emergence_level"] for m in self.history]
        return float(np.polyfit(np.arange(len(e)), e, 1)[0])
