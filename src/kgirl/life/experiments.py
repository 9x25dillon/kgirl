"""The five experimental axes of the parent manuscript, as runnable, multi-seed protocols.

    ablation      zero or freeze H, R, C, or the update of M; which capacities collapse
    stress        sweep the mean energy inflow down; the order in which O, C, R, A and C_eff degrade, and the
                  measured collapse against the analytic critical inflow E_crit (analysis.critical_energy)
    perturbation  distribution shifts of growing size; adaptation time, overshoot, self-model continuity
    compression   sweep the complexity penalty λ; where compression starts to destroy what recovery needs
    markers       C_eff and its factors against externally measured markers, beside simpler baselines. It
                  needs real marker data, so nothing here fabricates any.

The world: d channels  I_t,i = amp · sin(2π t / (20 + 7i) + φ_i) + offset_i + σ ξ_t,i  (ξ from ChaosStream);
inflow E_t = Ē (1 + 0.2 sin(t / 5)). A shock at t_s adds `shock` to every offset and reverses the phases.

Every number these protocols report is a property of this model and this world (Level 1 measurements of a
Level 2 model). They show what the architecture does, not what consciousness is.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from typing import Sequence

from .analysis import critical_energy, steady_state
from .chaos import ChaosStream
from .core import ABLATIONS, LifeParams, Organism

SEEDS = (0.4123, 0.5123, 0.6123)


@dataclass(frozen=True)
class World:
    d: int = 8
    amp: float = 0.5
    noise: float = 0.1
    energy: float = 0.5
    shock_at: int | None = 600
    shock: float = 1.0
    steps: int = 1200

    def observe(self, t: int, rng: ChaosStream) -> list[float]:
        shocked = self.shock_at is not None and t >= self.shock_at
        sign, off = (-1.0, self.shock) if shocked else (1.0, 0.0)
        return [sign * self.amp * math.sin(2 * math.pi * t / (20 + 7 * i) + 0.7 * i) + off + 0.1 * i
                + self.noise * rng.normal() for i in range(self.d)]

    def inflow(self, t: int) -> float:
        return self.energy * (1.0 + 0.2 * math.sin(t / 5.0))


@dataclass
class Run:
    err: list[float] = field(default_factory=list)
    ceff: list[float] = field(default_factory=list)
    A: list[float] = field(default_factory=list)
    C: list[float] = field(default_factory=list)
    O: list[float] = field(default_factory=list)
    F: list[float] = field(default_factory=list)
    R: list[float] = field(default_factory=list)
    P: list[float] = field(default_factory=list)
    J: list[float] = field(default_factory=list)
    heat: list[float] = field(default_factory=list)
    model_norm: float = 0.0
    var: float = 0.0


def run(params: LifeParams | None = None, world: World | None = None, seed: float = SEEDS[0],
        ablate: Sequence[str] = ()) -> Run:
    p, w = params or LifeParams(), world or World()
    org = Organism(p, d=w.d, seed=seed, keep=1, ablate=ablate)
    noise = ChaosStream((seed * 3.7 + 0.29) % 1.0)
    out = Run()
    for t in range(w.steps):
        rec = org.step(w.observe(t, noise), w.inflow(t))
        s = org.state
        out.err.append(rec.err)
        out.ceff.append(rec.ceff)
        out.heat.append(rec.heat)
        out.J.append(rec.J)
        for k in ("A", "C", "O", "F", "R", "P"):
            getattr(out, k).append(getattr(s, k))
    out.model_norm = org.state.model_norm()
    out.var = sum(org.state.eps_var) / w.d
    return out


# ---------------------------------------------------------------------- summary statistics


def _mean(x: Sequence[float]) -> float:
    return sum(x) / len(x) if x else float("nan")


def _sd(x: Sequence[float]) -> float:
    if len(x) < 2:
        return 0.0
    m = _mean(x)
    return math.sqrt(sum((v - m) ** 2 for v in x) / (len(x) - 1))


def smooth(x: Sequence[float], span: int = 10) -> list[float]:
    a, m, out = 2.0 / (span + 1.0), x[0], []
    for v in x:
        m += a * (v - m)
        out.append(m)
    return out


def adaptation(r: Run, t_shock: int, pre: int = 150, factor: float = 1.5) -> dict:
    """Steps until the smoothed error is back within `factor` × its pre-shock mean; overshoot; continuity of R."""
    e = smooth(r.err)
    base = _mean(e[t_shock - pre:t_shock])
    post = e[t_shock:]
    k = next((i for i, v in enumerate(post) if i > 0 and v <= factor * base), None)
    r_before = r.R[t_shock - 1]
    return {"adapt_steps": k, "overshoot": max(post) / base if base > 0 else math.inf,
            "R_continuity": min(r.R[t_shock:t_shock + 150]) / r_before if r_before > 0 else 0.0}


def _agg(rows: list[dict]) -> dict:
    out = {}
    for key in rows[0]:
        vals = [r[key] for r in rows if r[key] is not None and math.isfinite(r[key])]
        missing = len(rows) - len(vals)
        out[key] = {"mean": _mean(vals) if vals else None, "sd": _sd(vals) if vals else None,
                    **({"missing": missing} if missing else {})}
    return out


def _late(r: Run, frac: float = 1 / 3, end: int | None = None) -> slice:
    end = end or len(r.err)
    return slice(int(end * (1 - frac)), end)


# ---------------------------------------------------------------------- the five axes


def ablation(params: LifeParams | None = None, world: World | None = None, seeds: Sequence[float] = SEEDS) -> dict:
    p, w = params or LifeParams(), world or World()
    out = {}
    for cond in ("none",) + ABLATIONS:
        rows = []
        for sd in seeds:
            r = run(p, w, sd, () if cond == "none" else (cond,))
            pre = slice(w.shock_at - 200, w.shock_at) if w.shock_at else _late(r)
            ad = adaptation(r, w.shock_at) if w.shock_at else {}
            rows.append({"err": _mean(r.err[pre]), "C_eff": _mean(r.ceff[pre]), "C_eff_sd": _sd(r.ceff[pre]),
                         "R": _mean(r.R[pre]), **ad})
        out[cond] = _agg(rows)
    return out


def stress(params: LifeParams | None = None, world: World | None = None, seeds: Sequence[float] = SEEDS,
           levels: Sequence[float] = (0.6, 0.4, 0.2, 0.1, 0.07, 0.05, 0.04, 0.03, 0.02)) -> dict:
    p = params or LifeParams()
    w = replace(world or World(), shock_at=None)
    table, e2s, vs, ms = [], [], [], []
    for E in levels:
        rows = []
        for sd in seeds:
            r = run(p, replace(w, energy=E), sd)
            late = _late(r)
            rows.append({k: _mean(getattr(r, k)[late]) for k in ("A", "C", "O", "F", "R")}
                        | {"C_eff": _mean(r.ceff[late]), "err2": _mean([e * e for e in r.err[late]])})
            if E == levels[0]:                  # the analyst's prior: statistics of the unstressed system
                e2s.append(rows[-1]["err2"])
                vs.append(r.var)
                ms.append(r.model_norm)
        table.append({"E": E, **{k: v["mean"] for k, v in _agg(rows).items()}})
    top = table[0]
    order = {}
    for k in ("A", "C", "O", "R", "C_eff"):
        hit = next((row["E"] for row in table if row[k] is not None and top[k] and row[k] < 0.5 * top[k]), None)
        order[k] = hit
    e2, var, m = _mean(e2s), _mean(vs), _mean(ms)
    e_crit = critical_energy(p, e2, var, m)
    collapse = next((row["E"] for row in table if row["O"] is not None and row["O"] < p.O_min), None)
    return {"levels": table, "half_loss_at_E": order, "E_crit_analytic": e_crit,
            "first_level_below_O_min": collapse,
            "steady_state_at_top": steady_state(p, levels[0], e2, var, m).to_dict()}


def perturbation(params: LifeParams | None = None, world: World | None = None, seeds: Sequence[float] = SEEDS,
                 magnitudes: Sequence[float] = (0.25, 0.5, 1.0, 2.0, 4.0)) -> dict:
    p, w = params or LifeParams(), world or World()
    out = {}
    for mag in magnitudes:
        rows = []
        for sd in seeds:
            r = run(p, replace(w, shock=mag), sd)
            rows.append(adaptation(r, w.shock_at) | {"O_min_after": min(r.O[w.shock_at:]),
                                                     "C_eff_min_after": min(r.ceff[w.shock_at:])})
        out[str(mag)] = _agg(rows)
    return out


def compression(params: LifeParams | None = None, world: World | None = None, seeds: Sequence[float] = SEEDS,
                lambdas: Sequence[float] = (0.0, 0.01, 0.1, 0.3, 1.0, 3.0, 9.0)) -> dict:
    p, w = params or LifeParams(), world or World()
    table = []
    for lam in lambdas:
        q = replace(p, lambda_reg=lam).check()
        rows = []
        for sd in seeds:
            r = run(q, w, sd)
            pre = slice(w.shock_at - 200, w.shock_at)
            rows.append({"model_norm": r.model_norm, "err": _mean(r.err[pre])} | adaptation(r, w.shock_at))
        table.append({"lambda": lam, **{k: v["mean"] for k, v in _agg(rows).items()}})
    base = table[0]
    knee = next((row["lambda"] for row in table[1:]
                 if (row["err"] or 0) > 1.25 * (base["err"] or 0)
                 or (row["adapt_steps"] or math.inf) > 2 * max(1.0, base["adapt_steps"] or 1.0)), None)
    return {"levels": table, "knee_lambda": knee}


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    n = min(len(x), len(y))
    if n < 3:
        return float("nan")
    mx, my = _mean(x[:n]), _mean(y[:n])
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sx = math.sqrt(sum((a - mx) ** 2 for a in x[:n]))
    sy = math.sqrt(sum((b - my) ** 2 for b in y[:n]))
    return sxy / (sx * sy) if sx > 0 and sy > 0 else float("nan")


def markers(r: Run, observed: Sequence[float]) -> dict:
    """Correlate C_eff and each factor with external markers, beside the manuscript's simpler baselines."""
    n = min(len(observed), len(r.ceff))
    F_norm = [f / (f + 1.0) for f in r.F]
    cand = {"C_eff": r.ceff, "A": r.A, "C": r.C, "O": r.O, "R": r.R, "baseline_energy": F_norm,
            "baseline_error": [-e for e in r.err]}
    scores = {k: pearson(v[:n], observed[:n]) for k, v in cand.items()}
    best_baseline = max(abs(scores["baseline_energy"]), abs(scores["baseline_error"]))
    return {"pearson": scores, "C_eff_beats_baselines": abs(scores["C_eff"]) > best_baseline}


AXES = {"ablation": ablation, "stress": stress, "perturbation": perturbation, "compression": compression}


def as_json(obj):
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    raise TypeError(type(obj))
