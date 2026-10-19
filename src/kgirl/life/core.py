"""The Algorithmic Life Equation as one pure discrete map  s_{t+1}, record_t = Φ(p, s_t, I_t, E_t, u_t).

Rebuilt from "Minimal Implementation of the Algorithmic Life Equation" and its mathematical appendix
(research/life/originals/). Every equation of the appendix is kept; where the reference implementation was
measured to degenerate, the fix is stated here and checked in tests/life.

State  s = (A, C, O, F, H, M, R, P, π) plus the bookkeeping the self-model and the variance window need.

    ε_t     = I_t − M_t φ(H_t),            φ(H) = [H; 1]                         prediction error
    W_t     = w0 + wA A + wM ‖M‖_F + wu c(u) + w_ext X_t                         work (X: measured, exogenous)
    D_t     = δ0 + δF F + δε ‖ε‖²                                                dissipation
    F_{t+1} = [F + η_in E − W − D]_+,      deficit_t = [W + D − F − η_in E]_+   free energy
    Ȯ       = a F C (1 − O) − b (D + ‖ε‖²) O                                     organization
    Ċ       = η_C (C_* − C) − η_V (Var ε − v_ref),  C ∈ (C_min, C_max)           coherence window
    M_{t+1} = (1 − 2η_M λ) M + 2η_M ε φᵀ / (κ + ‖φ‖²)  [then prox of l1·‖M‖₁]   complexity-penalized NLMS
    H_{t+1} = (1 − ρ) H + ρ [I; ε]                                               memory
    Ṙ       = η_R (G_self (1 − R) − L_self R)                                    self-model strength
    P       = log(1 + N_reach(F, O, C))                                          reachable potential
    Ȧ       = η_A (A_target · paid − A) − η_AV Var ε                             processing in use
    C_eff   = (A / A_max) · C · O · R · F / (F + K_F)                            functional index

Unifying structure. A, C, O and R each obey ẋ = α − βx with α, β ≥ 0 held fixed over one step, so the
step can be integrated exactly:  x' = x e^{−β} + (α/β)(1 − e^{−β}).  That is `integrator="exact"`
(default). It agrees with the explicit Euler step x + α − βx to first order, is unconditionally stable, and
keeps O and R inside [0, 1] without clipping. `integrator="euler"` is the appendix's literal map.

Fixes to the reference implementation (measured, tests/life/test_core.py):
  1. Organization. With the published defaults F settles near 35, so a F C + b(D + ε²) ≈ 2.6 > 2: the Euler
     map has multiplier < −1 and only the clip at O = 1 hides the oscillation. The exact step removes it.
  2. Self-model. The PDF listing uses G = −tanh‖ε‖ ≤ 0 and the .py uses G = 0.1(1 − |F − 2|/5), which turns
     negative once F > 7. Either way R → 0 within ~50 steps and C_eff ≡ 0 afterwards. Here G_self is the
     appendix's quantity, the predictive usefulness of the internal state for the next error (an online
     R² skill score, mapped to [0, 1]), and L_self is the self-prediction error plus ξ at a discontinuity.
     The logistic form has the interior equilibrium R* = G / (G + L).
  3. Potential. int(4 · min(1, F/2) · O · C) is 0 or 1 for every reachable state, so P ≡ log 2. Here
     N_reach counts the action sequences of length `horizon` whose cost fits the usable budget O κ(C) F.
  4. Variance. np.var over a list of vectors pooled time and channels. Here Var ε is the channel mean of an
     exponentially weighted per-channel variance with span `var_span`.
  5. Memory. The reference used a ring buffer and never used ρ. Here H follows the appendix's EMA.
  6. Coherence. Nothing could push C above C_*, so the "too rigid" side of the window was unreachable.
     v_ref > 0 makes it two-sided: residual variance below v_ref (a world that has become too predictable)
     raises C toward C_max. v_ref = 0 is the appendix.
  7. Policy. π never changed. Here action values follow the appendix's constrained objective
     J = α·(‖ε‖² + λΩ) + β·W + γ·V + δ·R_loss as a bandit: Q_u ← Q_u + η_π(−J − Q_u).
  8. Unpaid work. In the appendix the world model, memory and self-model keep learning at F = 0, which
     is computation without free energy. Measured: the stationary error was identical at every inflow from
     0.6 down to 0.02. Here every work-driven process runs at the paid fraction
     paid = min(1, (F + η_in E) / (W + D)): the learning rates of M, H, S and q, and the processing target
     of A. Passive decay (ridge shrinkage, organization loss) continues unpaid.

Confidence: the map, its invariants (energy ledger, non-negative entropy production, bounds) and the exact
integrator are Level 0 mathematics. Every functional form is a Level 2 modelling choice. C_eff is a
diagnostic of this architecture, not a measure of experience (Level 3 if read that way).

Standard library only (the harness imports it). Cost per step: O(d²) for the world model, O(K^h) for P.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from itertools import product
from typing import Iterable

from .chaos import ChaosStream

Vec = tuple[float, ...]
Mat = tuple[Vec, ...]
ABLATIONS = ("memory", "self", "coherence", "model")
N_INTERNAL = 5                      # z = (A/A_max, C, O, F/(F+K_F), R)


@dataclass(frozen=True)
class LifeParams:
    # processing capacity
    A_max: float = 1.0
    A_target: float = 0.7
    eta_A: float = 0.05
    eta_AV: float = 0.02
    # energy
    eta_in: float = 0.8
    w0: float = 0.01
    wA: float = 0.02
    wM: float = 0.001
    wu: float = 0.01
    w_ext: float = 1.0
    delta0: float = 0.005
    deltaF: float = 0.01
    delta_eps: float = 0.02
    # organization
    a: float = 0.15
    b: float = 0.08
    # coherence window
    C_min: float = 0.15
    C_max: float = 0.85
    eta_C: float = 0.12
    eta_V: float = 0.05
    v_ref: float = 0.0
    margin: float = 1e-3
    # world model and memory
    eta_M: float = 0.05
    lambda_reg: float = 0.01
    l1: float = 0.0
    kappa: float = 1.0
    rho: float = 0.2
    var_span: int = 10
    # self-model
    eta_R: float = 0.1
    eta_S: float = 0.2
    self_tol: float = 0.05
    xi: float = 0.5
    disc_drop: float = 0.3
    skill_span: int = 30
    # reachable potential and action
    action_costs: Vec = (0.5, 1.0, 1.5, 2.0)
    horizon: int = 2
    eta_pi: float = 0.1
    tau_pi: float = 0.5
    # constrained objective J = α·L + β·W + γ·V + δ·R_loss
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 1.0
    delta: float = 1.0
    F_min: float = 0.1
    O_min: float = 0.2
    # functional index and thermodynamics
    K_F: float = 1.0
    kT: float = 1.0
    landauer_bits: float = 0.0
    integrator: str = "exact"

    @property
    def C_star(self) -> float:
        return 0.5 * (self.C_min + self.C_max)

    @property
    def half_width(self) -> float:
        return 0.5 * (self.C_max - self.C_min)

    def violations(self) -> list[str]:
        """Conditions the map needs; an empty list means the parameters are admissible."""
        out = []
        nonneg = ("A_max", "eta_A", "eta_AV", "w0", "wA", "wM", "wu", "w_ext", "delta0", "deltaF", "delta_eps",
                  "a", "b", "eta_C", "eta_V", "v_ref", "eta_M", "lambda_reg", "l1", "eta_R", "eta_S", "xi", "K_F")
        out += [f"{k} < 0" for k in nonneg if getattr(self, k) < 0]
        if not 0 < self.eta_in <= 1:
            out.append("need 0 < eta_in ≤ 1 (conversion cannot create free energy)")
        if not 0 < self.C_min < self.C_max < 1:
            out.append("need 0 < C_min < C_max < 1")
        if not 0 <= self.margin < self.half_width:
            out.append("margin must be smaller than the window half-width")
        if not 0 < self.eta_M < 1:
            out.append("need 0 < eta_M < 1 (NLMS stability: step 2η_M‖φ‖²/(κ+‖φ‖²) < 2)")
        if not 2 * self.eta_M * self.lambda_reg < 1:
            out.append("need 2·eta_M·lambda_reg < 1 (ridge shrinkage must not flip the sign of M)")
        if not 0 < self.rho < 1:
            out.append("need 0 < rho < 1")
        if not 0 <= self.A_target <= self.A_max:
            out.append("need 0 ≤ A_target ≤ A_max")
        if self.var_span < 1 or self.skill_span < 1 or self.horizon < 1 or not self.action_costs:
            out.append("spans and horizon must be ≥ 1 and there must be at least one action")
        if any(c < 0 for c in self.action_costs):
            out.append("action costs must be ≥ 0")
        if self.tau_pi <= 0 or self.kT <= 0 or self.self_tol <= 0 or self.kappa <= 0:
            out.append("tau_pi, kT, self_tol and kappa must be > 0")
        if self.w0 + 1e-15 < self.landauer_bits * self.kT * math.log(2):
            out.append("w0 is below the Landauer floor kT·ln2·landauer_bits for the bits each step erases")
        if self.integrator not in ("exact", "euler"):
            out.append("integrator must be 'exact' or 'euler'")
        return out

    def check(self) -> "LifeParams":
        bad = self.violations()
        if bad:
            raise ValueError("; ".join(bad))
        return self


@dataclass(frozen=True)
class LifeState:
    A: float
    C: float
    O: float
    F: float
    R: float
    P: float
    H: Vec                          # memory: EMA of [I; ε], length 2d
    M: Mat                          # world model: d × (2d + 1)
    Q: Vec                          # action values; policy π = softmax(Q / τ_π)
    eps_mean: Vec
    eps_var: Vec
    S: Mat                          # self-model: 5 × 6, predicts z_{t+1} from [z_t; 1]
    q: Vec                          # error predictor: ‖ε_{t+1}‖² from [z_t; 1]
    z_prev: Vec | None = None
    z_hat: Vec | None = None
    mse_model: float = 0.0
    mse_base: float = 0.0
    e_mean: float = 0.0
    t: int = 0

    @property
    def d(self) -> int:
        return len(self.M)

    @classmethod
    def initial(cls, d: int, p: LifeParams | None = None, *, A: float = 0.5, C: float = 0.5, O: float = 0.5,
                F: float = 2.0, R: float = 0.3) -> "LifeState":
        p = p or LifeParams()
        k = 2 * d + 1
        S = tuple(tuple(1.0 if j == i else 0.0 for j in range(N_INTERNAL + 1)) for i in range(N_INTERNAL))
        s = cls(A, C, O, F, R, 0.0, (0.0,) * (2 * d), tuple((0.0,) * k for _ in range(d)),
                (0.0,) * len(p.action_costs), (0.0,) * d, (0.0,) * d, S, (0.0,) * (N_INTERNAL + 1))
        return replace(s, P=potential(p, s))

    def internal(self, p: LifeParams) -> Vec:
        return (self.A / p.A_max, self.C, self.O, self.F / (self.F + p.K_F), self.R)

    def functional_index(self, p: LifeParams) -> float:
        return (self.A / p.A_max) * self.C * self.O * self.R * (self.F / (self.F + p.K_F))

    def model_norm(self) -> float:
        return math.sqrt(sum(m * m for row in self.M for m in row))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "LifeState":
        def tup(v):
            return tuple(tup(x) for x in v) if isinstance(v, (list, tuple)) else v
        return cls(**{k: tup(v) for k, v in data.items()})


@dataclass(frozen=True)
class StepRecord:
    t: int
    action: int
    err: float                      # ‖ε_t‖
    var: float                      # Var ε after this step
    work: float                     # W_t
    dissipation: float              # D_t
    deficit: float                  # demand the free energy could not cover
    paid: float                     # share of the demanded work that was paid for, in [0, 1]
    heat: float                     # entropy production σ_t ≥ 0 (in units of kT)
    J: float                        # constrained objective
    G: float                        # G_self ∈ [0, 1]
    L: float                        # L_self ≥ 0
    skill: float                    # R² skill of internal state for the next error, in [−1, 1]
    discontinuity: bool
    ceff: float                     # C_eff of the new state
    prediction: Vec = field(default=(), repr=False)


# ---------------------------------------------------------------------- small pure helpers


def _lin(x: float, alpha: float, beta: float, exact: bool) -> float:
    """One step of ẋ = α − βx: exact (exponential) or explicit Euler."""
    if not exact:
        return x + alpha - beta * x
    if beta < 1e-12:
        return x + alpha
    e = math.exp(-beta)
    return x * e + (alpha / beta) * (1.0 - e)


def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _matvec(M: Mat, v: Vec) -> Vec:
    return tuple(sum(m * x for m, x in zip(row, v)) for row in M)


def _nlms(M: Mat, err: Vec, phi: Vec, rate: float, kappa: float, shrink: float = 0.0, l1: float = 0.0) -> Mat:
    """M ← (1 − shrink)·M + rate·err·φᵀ/(κ + ‖φ‖²), then soft-threshold by l1."""
    g = rate / (kappa + sum(x * x for x in phi))
    keep = 1.0 - shrink
    rows = []
    for row, e in zip(M, err):
        ge = g * e
        new = [keep * m + ge * x for m, x in zip(row, phi)]
        if l1 > 0:
            new = [math.copysign(max(abs(m) - l1, 0.0), m) for m in new]
        rows.append(tuple(new))
    return tuple(rows)


def softmax(values: Vec, tau: float) -> Vec:
    top = max(values)
    w = [math.exp((v - top) / tau) for v in values]
    s = sum(w)
    return tuple(x / s for x in w)


def window_quality(p: LifeParams, C: float) -> float:
    """κ(C) = 1 − ((C − C_*)/h)², 1 at the window's centre and 0 at its edges."""
    return max(0.0, 1.0 - ((C - p.C_star) / p.half_width) ** 2)


def overhead(p: LifeParams, s: LifeState) -> float:
    """Per-step cost that does not depend on the action: w0 + wA A + wM ‖M‖ + δ0 + δF F."""
    return p.w0 + p.wA * s.A + p.wM * s.model_norm() + p.delta0 + p.deltaF * s.F


def reach_count(p: LifeParams, budget: float, base: float) -> int:
    """N_reach: action sequences of length `horizon` whose summed cost fits `budget`."""
    costs = [base + p.wu * c for c in p.action_costs]
    if p.horizon * min(costs) > budget:
        return 0
    return sum(1 for seq in product(costs, repeat=p.horizon) if sum(seq) <= budget)


def potential(p: LifeParams, s: LifeState) -> float:
    budget = s.O * window_quality(p, s.C) * s.F
    return math.log1p(reach_count(p, budget, overhead(p, s)))


def viability_penalty(p: LifeParams, s: LifeState, deficit: float = 0.0) -> float:
    v = deficit
    if p.F_min > 0:
        v += max(0.0, p.F_min - s.F) / p.F_min
    if p.O_min > 0:
        v += max(0.0, p.O_min - s.O) / p.O_min
    return v + (1.0 - window_quality(p, s.C))


# ---------------------------------------------------------------------- the map


def step(p: LifeParams, s: LifeState, obs: Iterable[float], energy: float, action: int = 0, *,
         work: float = 0.0, reset: bool = False, ablate: frozenset[str] | set[str] = frozenset()
         ) -> tuple[LifeState, StepRecord]:
    """Advance one step. Pure: same inputs, same outputs; `s` is not modified.

    `work` is measured exogenous work X_t (for example tokens spent, in budget units), charged as w_ext·X_t.
    `reset` marks a forced discontinuity (a restart that lost state). `ablate` ⊆ ABLATIONS freezes a part.
    """
    obs = tuple(float(x) for x in obs)
    d = s.d
    if len(obs) != d:
        raise ValueError(f"observation has {len(obs)} channels, the organism has {d}")
    if not 0 <= action < len(p.action_costs):
        raise ValueError(f"action {action} outside 0..{len(p.action_costs) - 1}")
    if energy < 0 or work < 0:
        raise ValueError("energy inflow and measured work must be ≥ 0")
    exact = p.integrator == "exact"

    # 1. predict and compare
    phi = s.H + (1.0,)
    pred = _matvec(s.M, phi)
    eps = tuple(o - y for o, y in zip(obs, pred))
    e2 = sum(x * x for x in eps)

    # 2. short-horizon residual variance (exponentially weighted, per channel, then averaged)
    av = 2.0 / (p.var_span + 1.0)
    mean, var = [], []
    for x, m, v in zip(eps, s.eps_mean, s.eps_var):
        dlt = x - m
        mean.append(m + av * dlt)
        var.append((1.0 - av) * (v + av * dlt * dlt))
    v_eps = sum(var) / d

    # 3. energy: work, dissipation, ledger
    cost_u = p.action_costs[action]
    W = p.w0 + p.wA * s.A + p.wM * s.model_norm() + p.wu * cost_u + p.w_ext * work
    D = p.delta0 + p.deltaF * s.F + p.delta_eps * e2
    supply = s.F + p.eta_in * energy
    F_next = max(0.0, supply - W - D)
    deficit = max(0.0, W + D - supply)
    paid = 1.0 if W + D <= supply else supply / (W + D)      # work is done only as far as it is paid for
    heat = ((1.0 - p.eta_in) * energy + min(W + D, supply)) / p.kT

    # 4. organization: logistic construction against error- and dissipation-driven decay
    g, loss = p.a * s.F * s.C, p.b * (D + e2)
    O_next = _clip(_lin(s.O, g, g + loss, exact), 0.0, 1.0)

    # 5. coherence window and processing fraction
    lo, hi = p.C_min + p.margin, p.C_max - p.margin
    if "coherence" in ablate:
        C_next = s.C
    else:
        C_next = _clip(_lin(s.C, p.eta_C * p.C_star - p.eta_V * (v_eps - p.v_ref), p.eta_C, exact), lo, hi)
    A_next = _clip(_lin(s.A, p.eta_A * p.A_target * paid - p.eta_AV * v_eps, p.eta_A, exact), 0.0, p.A_max)

    # 6. world model (complexity-penalized NLMS) and memory
    if "model" in ablate:
        M_next = s.M
    else:
        M_next = _nlms(s.M, eps, phi, 2.0 * p.eta_M * paid, p.kappa, 2.0 * p.eta_M * p.lambda_reg,
                       p.eta_M * p.l1)
    rho = p.rho * paid
    H_next = s.H if "memory" in ablate else tuple((1.0 - rho) * h + rho * x for h, x in zip(s.H, obs + eps))

    # 7. self-model: predictive usefulness of the internal state (G) against self-prediction error (L)
    z = s.internal(p)
    aS = 2.0 / (p.skill_span + 1.0)
    q, S, mse_m, mse_b, e_mean = s.q, s.S, s.mse_model, s.mse_base, s.e_mean
    disc = reset
    if s.z_prev is not None:
        zp = s.z_prev + (1.0,)
        r_model = e2 - sum(w * x for w, x in zip(q, zp))
        r_base = e2 - e_mean
        mse_m = (1.0 - aS) * mse_m + aS * r_model * r_model
        mse_b = (1.0 - aS) * mse_b + aS * r_base * r_base
        q = _nlms((q,), (r_model,), zp, p.eta_S * paid, p.kappa)[0]
        disc = disc or (s.z_prev[3] - z[3]) > p.disc_drop
    e_mean = e_mean + aS * (e2 - e_mean) if s.z_prev is not None else e2
    skill = _clip(1.0 - mse_m / mse_b, -1.0, 1.0) if mse_b > 1e-12 else 0.0
    G = 0.5 * (1.0 + skill)
    miss = math.sqrt(sum((a - b) ** 2 for a, b in zip(z, s.z_hat))) if s.z_hat is not None else 0.0
    L = math.tanh(miss / p.self_tol) + (p.xi if disc else 0.0)
    if s.z_prev is not None and s.z_hat is not None:
        S = _nlms(S, tuple(a - b for a, b in zip(z, s.z_hat)), s.z_prev + (1.0,), p.eta_S * paid, p.kappa)
    if "self" in ablate:
        R_next = s.R
    else:
        R_next = _clip(_lin(s.R, p.eta_R * G, p.eta_R * (G + L), exact), 0.0, 1.0)

    nxt = replace(s, A=A_next, C=C_next, O=O_next, F=F_next, R=R_next, H=H_next, M=M_next,
                  eps_mean=tuple(mean), eps_var=tuple(var), S=S, q=q, z_prev=z, mse_model=mse_m,
                  mse_base=mse_b, e_mean=e_mean, t=s.t + 1)
    # the self-model's forecast of z_{t+1} from z_t; it is scored against nxt.internal() at the next step
    nxt = replace(nxt, z_hat=_matvec(S, z + (1.0,)))

    # 8. reachable potential, objective, policy
    P_next = potential(p, nxt)
    omega = sum(m * m for row in M_next for m in row)
    J = (p.alpha * (e2 + p.lambda_reg * omega) + p.beta * W + p.gamma * viability_penalty(p, nxt, deficit)
         + p.delta * max(0.0, s.P - P_next))
    Q = tuple(qv + p.eta_pi * (-J - qv) if i == action else qv for i, qv in enumerate(s.Q))
    nxt = replace(nxt, P=P_next, Q=Q)

    rec = StepRecord(s.t, action, math.sqrt(e2), v_eps, W, D, deficit, paid, heat, J, G, L, skill, disc,
                     nxt.functional_index(p), pred)
    return nxt, rec


# ---------------------------------------------------------------------- stateful wrapper


class Organism:
    """A LifeState plus a chaos-driven policy sampler and a bounded history. Not thread-safe by itself."""

    def __init__(self, params: LifeParams | None = None, d: int = 8, seed: float = 0.4123,
                 state: LifeState | None = None, keep: int | None = 10_000,
                 ablate: Iterable[str] = ()):
        self.p = (params or LifeParams()).check()
        self.state = state or LifeState.initial(d, self.p)
        self.rng = ChaosStream(seed)
        self.history: list[StepRecord] = []
        self.keep = keep
        self.ablate = frozenset(ablate)
        unknown = self.ablate - set(ABLATIONS)
        if unknown:
            raise ValueError(f"unknown ablation(s) {sorted(unknown)}; choose from {ABLATIONS}")

    def policy(self) -> Vec:
        return softmax(self.state.Q, self.p.tau_pi)

    def act(self) -> int:
        return self.rng.choice(self.policy())

    def step(self, obs: Iterable[float], energy: float, action: int | None = None, *, work: float = 0.0,
             reset: bool = False) -> StepRecord:
        u = self.act() if action is None else action
        self.state, rec = step(self.p, self.state, obs, energy, u, work=work, reset=reset, ablate=self.ablate)
        self.history.append(rec)
        if self.keep is not None and len(self.history) > self.keep:
            del self.history[: len(self.history) - self.keep]
        return rec

    def vitals(self) -> dict:
        s, p = self.state, self.p
        return {"t": s.t, "A": s.A, "C": s.C, "O": s.O, "F": s.F, "R": s.R, "P": s.P,
                "C_eff": s.functional_index(p), "model_norm": s.model_norm(),
                "var_eps": sum(s.eps_var) / s.d, "policy": self.policy()}
