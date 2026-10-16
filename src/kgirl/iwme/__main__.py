"""`python -m kgirl.iwme <command>` — the IWME verification bench from the command line."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def _dump(obj) -> str:
    def default(o):
        if hasattr(o, "tolist"):
            return o.tolist()
        if isinstance(o, complex):
            return [o.real, o.imag]
        return str(o)
    return json.dumps(obj, indent=2, default=default, ensure_ascii=False)


def irt_point(phi: float, psi: float, kernel: str = "damped", R: float = 3.0, r: float = 1.0,
              alpha: float = 0.3, beta: float = 0.5, k0: float = 1.0) -> dict:
    from . import irt
    p = irt.IRTParams(R, r, alpha, beta)
    kern = irt.DAMPED_KERNEL if kernel == "damped" else irt.constant_kernel(k0)
    g = irt.metric(phi, psi, p, kern)
    return {
        "kernel": kern.name,
        "metric (theta, phi, psi)": g.round(10).tolist(),
        "published_metric": irt.published_metric(phi, psi, p, kern).round(10).tolist(),
        "det": float(irt.det_metric(phi, psi, p, kern)),
        "jacobian_rank": irt.jacobian_rank(0.0, phi, psi, p, kern),
        "scalar_curvature": float(irt.scalar_curvature_gauss(0.0, phi, psi, p, kern)),
        "principal_curvatures": irt.principal_curvatures(0.0, phi, psi, p, kern).tolist(),
        "published_eq29": float(irt.published_scalar_curvature(phi, psi, p, kern)),
    }


def attention_summary(T: float, rho: float = 0.5) -> dict:
    from . import attention as at
    fps = at.fixed_points_tanh(T)
    mult = [at.multiplier_tanh(h, T, rho) for h in fps]
    mu0 = at.multiplier_tanh(0.0, T, rho)
    if abs(mu0) < 1.0:
        ret0 = at.effective_time_constant(mu0)
    elif abs(mu0) == 1.0:
        ret0 = "critical: algebraic decay H_t ~ (3/(2 rho t))^(1/2)"
    else:
        ret0 = "unstable: perturbations of H = 0 grow (lock-in to ±H*)"
    return {
        "T": T, "rho": rho,
        "fixed_points": fps,
        "multipliers": mult,
        "stable": [abs(m) < 1 for m in mult],
        "retention_time_at_0": ret0,
        "open_loop_retention_time": at.effective_time_constant(1 - rho),
        "static_hysteresis_area": at.static_hysteresis_area(T),
        "flip_threshold_if_values_antialigned": at.flip_threshold(rho),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="kgirl.iwme", description="IWME Phase 3–4 verification bench")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("verify", help="run the claim registry")
    p.add_argument("--json", action="store_true")
    p.add_argument("--fast", action="store_true", help="skip slow checks")
    p.add_argument("--latex", help="write LaTeX table rows to this path")
    p.add_argument("--only", nargs="*", help="check ids, e.g. V22 V26")
    p = sub.add_parser("irt", help="IRT metric and curvature at one point")
    p.add_argument("--phi", type=float, default=2.5)
    p.add_argument("--psi", type=float, default=1.7)
    p.add_argument("--kernel", choices=("damped", "constant"), default="damped")
    p.add_argument("--k0", type=float, default=1.0)
    p = sub.add_parser("attention", help="fixed points, stability, retention and hysteresis at temperature T")
    p.add_argument("--T", type=float, default=0.8)
    p.add_argument("--rho", type=float, default=0.5)
    p = sub.add_parser("cascade", help="D_q of the binomial cascade")
    p.add_argument("--p", type=float, default=0.7)
    p = sub.add_parser("persist", help="R_persist of the linear-Gaussian IWME after decoupling")
    p.add_argument("--c", type=float, default=0.5)
    p.add_argument("--rho", type=float, default=0.3)
    p.add_argument("--memory", choices=("ema", "accumulator"), default="ema")
    p = sub.add_parser("plot", help="write the paper's figures (matplotlib)")
    p.add_argument("-o", "--outdir", default="outputs/iwme")
    a = ap.parse_args(argv)

    if a.cmd == "verify":
        from .verify import render, run_all, to_json, to_latex
        res = run_all(fast=a.fast, only=a.only)
        if a.latex:
            Path(a.latex).parent.mkdir(parents=True, exist_ok=True)
            Path(a.latex).write_text(to_latex(res), encoding="utf-8")
        print(to_json(res) if a.json else render(res))
        return 0 if all(r.passed for r in res) else 1
    if a.cmd == "irt":
        print(_dump(irt_point(a.phi, a.psi, a.kernel, k0=a.k0)))
    elif a.cmd == "attention":
        print(_dump(attention_summary(a.T, a.rho)))
    elif a.cmd == "cascade":
        from .cascade import binomial_Dq
        qs = [-math.inf, -1, 0, 0.5, 1, 2, 3, math.inf]
        print(_dump({str(q): binomial_Dq(a.p, q) for q in qs}))
    elif a.cmd == "persist":
        from .core import LinearIWME, persistence_information
        m = LinearIWME(c=a.c, rho=a.rho, memory=a.memory)
        out = {"given_X": persistence_information(m), "given_X_and_H": persistence_information(m, condition="XH")}
        print(_dump(out))
    elif a.cmd == "plot":
        from .plots import ALL
        for name, fn in ALL.items():
            print(fn(Path(a.outdir) / f"{name}.pdf"))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
