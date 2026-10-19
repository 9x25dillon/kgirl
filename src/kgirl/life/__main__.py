"""`python -m kgirl.life <command>`: run, analyze and test the Algorithmic Life Equation.

    demo          one organism through a shock; prints its vitals every 100 steps
    steady        closed-form fixed point, time constants, Euler multiplier and critical inflow at --energy
    ablation | stress | perturbation | compression
                  the manuscript's experimental axes, multi-seed (`--seeds`)
    stagger       replica schedules (single, sync, half-cycle, uniform, random): exact and jittered availability,
                  the replication criterion, and swarm plans for a range of agent success rates
    triage        the uploaded reference simulator (research/life/originals/life_simulator.py, needs numpy)
                  against this rebuild on the same input
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

from . import experiments as X
from .analysis import critical_energy, critical_free_energy, steady_state
from .chaos import ChaosStream
from .core import LifeParams, Organism
from .replication import (SCHEMES, availability, best_replication, best_stagger, free_running_availability,
                          offsets, simulate, swarm_size, uniform_availability)

ORIGINAL = Path(__file__).resolve().parents[3] / "research" / "life" / "originals" / "life_simulator.py"


def steady(energy: float = 0.5, params: LifeParams | None = None) -> dict:
    p = params or LifeParams()
    r = X.run(p, X.World(energy=energy, shock_at=None, steps=1500))
    late = slice(1000, 1500)
    e2 = X._mean([e * e for e in r.err[late]])
    ss = steady_state(p, energy, e2, r.var, r.model_norm)
    return {"energy": energy, "measured": {k: X._mean(getattr(r, k)[late]) for k in ("A", "C", "O", "F", "R")},
            "analytic": ss.to_dict(), "e2": e2, "var": r.var, "model_norm": r.model_norm,
            "F_crit": critical_free_energy(p, ss.C, e2), "E_crit": critical_energy(p, e2, r.var, r.model_norm)}


def stagger(n: int = 3, period: float = 1.0, reset: float = 0.4) -> dict:
    rows = {}
    for sch in SCHEMES:
        offs = offsets(sch, 1 if sch == "single" else n, period)
        rows[sch] = {"offsets": [round(o, 4) for o in offs], "availability": availability(offs, period, reset),
                     "jittered_phase_locked": simulate(sch, len(offs), period, reset).to_dict(),
                     "jittered_free_running": simulate(sch, len(offs), period, reset, phase_lock=False).to_dict()}
    plans = {}
    for p in (0.1, 0.3, 0.5, 0.7, 0.9):
        size = swarm_size(p, 10.0, 8)
        sc = best_stagger(p, size, 60.0, 2.0)
        plans[str(p)] = {"N*": size, "stagger_s_for_60s_agents": sc.delta, "E_cost": sc.cost,
                         "E_latency": sc.latency, "P_success": sc.success}
    return {"n": n, "period": period, "reset": reset, "schedules": rows,
            "uniform_closed_form": uniform_availability(n, period, reset),
            "free_running_closed_form": free_running_availability(n, period, reset),
            "best_replication": best_replication(period, reset, value=10.0, c_rep=0.5, c_coord=0.1).to_dict(),
            "swarm_plans_V_over_c_10": plans}


def triage(steps: int = 400, starve: float = 0.03) -> dict:
    """`steps` at the reference's own input, then `steps` more at a starving inflow `starve`."""
    try:
        import numpy as np
    except ImportError as exc:                           # pragma: no cover - numpy is present in kgirl envs
        raise RuntimeError("triage needs numpy") from exc
    if not ORIGINAL.exists():
        raise FileNotFoundError(f"reference not found at {ORIGINAL}")
    spec = importlib.util.spec_from_file_location("_life_reference", ORIGINAL)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    np.random.seed(0)
    life = ref.MinimalLife()
    rows = []
    for t in range(2 * steps):
        obs = np.random.randn(8) * 0.1
        s, _, ceff = life.life_step(obs, 0.5 + 0.1 * np.sin(t / 5) if t < steps else starve)
        eps2 = float(np.linalg.norm(life.eps_history[-1]) ** 2)
        D = life.p.delta0 + life.p.deltaF * s.F + life.p.delta_eps * eps2
        rows.append((s.F, s.O, s.R, float(s.P), float(ceff), life.p.a * s.F * s.C + life.p.b * (D + eps2)))
    org, noise = Organism(d=8), ChaosStream(0.77)
    mine = []
    for t in range(2 * steps):
        rec = org.step([0.1 * noise.normal() for _ in range(8)], 0.5 + 0.1 * math.sin(t / 5) if t < steps else starve)
        mine.append((org.state.R, org.state.P, rec.ceff, org.state.F))
    fed, rf = slice(0, steps), rows[:steps]
    return {
        "reference": {
            "P_values_fed_and_starving": sorted({round(r[3], 6) for r in rows}),
            "R_first_zero_step": next((t for t, r in enumerate(rows) if r[2] == 0.0), None),
            "share_fed_steps_with_C_eff_zero": sum(r[4] == 0.0 for r in rf) / steps,
            "F_end_of_fed_phase": rf[-1][0],
            "organization_rate_g_plus_l": rf[-1][5],
            "euler_multiplier_O": 1.0 - rf[-1][5],
        },
        "rebuild": {
            "P_fed": mine[steps - 1][1], "P_starving_end": mine[-1][1],
            "R_min_fed_after_step_50": min(m[0] for m in mine[50:steps]),
            "share_fed_steps_with_C_eff_zero": sum(m[2] == 0.0 for m in mine[fed]) / steps,
            "C_eff_end_of_fed_phase": mine[steps - 1][2], "F_starving_end": mine[-1][3],
        },
    }


def experiment(axis: str, seeds: int = 3, energy: float = 0.5) -> dict:
    sd = X.SEEDS[:seeds] if seeds <= len(X.SEEDS) else tuple(0.4123 + 0.1 * k for k in range(seeds))
    if axis == "steady":
        return steady(energy)
    if axis == "stagger":
        return stagger()
    if axis == "triage":
        return triage()
    if axis in X.AXES:
        return X.AXES[axis](seeds=sd)
    raise ValueError(f"axis must be steady, stagger, triage or one of {sorted(X.AXES)}")


def demo(steps: int = 1200) -> list[str]:
    w = X.World(steps=steps)
    org, noise = Organism(d=w.d), ChaosStream(0.29)
    lines = []
    for t in range(steps):
        rec = org.step(w.observe(t, noise), w.inflow(t))
        if t % 100 == 0 or t == w.shock_at:
            v = org.vitals()
            lines.append(f"t={t:4d}{'*' if t == w.shock_at else ' '} |ε|={rec.err:.3f} F={v['F']:6.2f} "
                         f"O={v['O']:.3f} C={v['C']:.3f} R={v['R']:.3f} A={v['A']:.3f} P={v['P']:.2f} "
                         f"C_eff={v['C_eff']:.3f} σ={rec.heat:.3f}")
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="kgirl.life", description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["demo", "steady", "stagger", "triage", *X.AXES])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--energy", type=float, default=0.5)
    a = ap.parse_args(argv)
    if a.cmd == "demo":
        print("\n".join(demo()))
        return 0
    print(json.dumps(experiment(a.cmd, a.seeds, a.energy), indent=2, default=float))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
