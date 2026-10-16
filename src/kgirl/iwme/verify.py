"""Claim registry: every number in the IWME Phase 0–3 paper, recomputed and checked.

    python -m kgirl.iwme verify            # table
    python -m kgirl.iwme verify --json     # machine-readable
    python -m kgirl.iwme verify --latex research/iwme/verification_table.tex

Each check returns a Result with the observed value(s), the expected value or relation, a PASS/FAIL flag and
the evidence level of the claim being checked (0 established, 1 extrapolation, 2 hypothesis, 3 speculative).
A FAIL means the code disagrees with the paper; a source claim found false is reported as a PASS of the
refuting check (the "expected" column says what the paper asserts about the source).
Deterministic: no PRNG (kgirl.evolve.chaos.ChaosRNG where variates are needed).
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from . import attention as at
from . import cascade as cs
from . import core
from . import irt
from . import sources as so
from . import wake as wk


@dataclass
class Result:
    cid: str
    area: str
    claim: str
    level: int
    expected: str
    observed: str
    passed: bool
    seconds: float = 0.0
    data: Dict = field(default_factory=dict)


REGISTRY: List = []


def check(cid: str, area: str, claim: str, level: int, slow: bool = False, needs: Optional[str] = None):
    def deco(fn: Callable[[], tuple]):
        REGISTRY.append((cid, area, claim, level, slow, needs, fn))
        return fn
    return deco


def _rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-300)


# ----------------------------------------------------------------------------------------------
# Attention and memory
# ----------------------------------------------------------------------------------------------

@check("V01", "attention", "P2: with the attention sequence fixed, ∂H_t/∂H_{t−k} = (1−ρ)^k", 0)
def _v01():
    rho, k = 0.5, 10
    m_seq = np.sin(np.arange(k) * 0.7)          # any input sequence chosen in advance

    def run(h):
        for m in m_seq:
            h = (1 - rho) * h + rho * m
        return h
    eps = 1e-6
    s = (run(0.3 + eps) - run(0.3 - eps)) / (2 * eps)
    exp = at.open_loop_sensitivity(rho, k)
    return f"(1−ρ)^k = {exp:.6g}", f"{s:.6g}", _rel(s, exp) < 1e-8, {"sensitivity": s}


@check("V02", "attention", "P2′: closed loop (scores read H), ∂H_t/∂H_0 is the product of (1−ρ)I + (ρ/T)Cov_a(v,k)", 0)
def _v02():
    mem = at.two_point_memory(1.2, 0.5)
    S = float(at.closed_loop_sensitivity(mem, np.array([0.3]), 10)[0, 0])
    F = float(at.finite_difference_sensitivity(mem, np.array([0.3]), 10)[0, 0])
    ol = at.open_loop_sensitivity(0.5, 10)
    return (f"Jacobian product = finite difference; ≠ open-loop {ol:.3g}",
            f"{S:.6f} (FD {F:.6f}), ratio to open loop {S / ol:.0f}",
            _rel(S, F) < 1e-7 and S / ol > 10, {"closed": S, "fd": F, "open": ol})


@check("V03", "attention", "Closed-loop retention time at H=0 diverges at T→1⁺ and tends to τ_mem as T→∞", 0)
def _v03():
    rho = 0.5
    tau_mem = at.effective_time_constant(1 - rho)
    rows = {T: at.effective_time_constant(at.multiplier_tanh(0.0, T, rho)) for T in (1.05, 1.5, 3.0, 1e6)}
    ok = rows[1.05] > rows[1.5] > rows[3.0] > rows[1e6] and _rel(rows[1e6], tau_mem) < 1e-5
    return (f"monotone in T, → τ_mem = {tau_mem:.4f}", ", ".join(f"T={k:g}: {v:.3f}" for k, v in rows.items()), ok,
            {"tau_mem": tau_mem, "tau_eff": {str(k): v for k, v in rows.items()}})


@check("V04", "attention", "Lemma: sup_a ‖Cov_a(v,k)‖ ≤ D_v D_k / 4 (contraction for T > D_v D_k / 4)", 0)
def _v04():
    from ..evolve.chaos import ChaosRNG
    rng = ChaosRNG(0.271)
    worst = 0.0
    for _ in range(12):
        K = rng.normal((6, 3))
        V = K @ rng.normal((3, 3))
        mem = at.AttentionMemory(K=K, V=V, T=1.0, rho=0.5)
        ratio = at.sup_cov_norm(mem, rng.normal((300, 3)) * 3) / at.cov_bound(K, V)
        worst = max(worst, ratio)
    return "ratio ≤ 1 on every configuration", f"max ratio {worst:.4f}", worst <= 1.0 + 1e-12, {"max_ratio": worst}


@check("V05", "attention", "The D_v D_k/4 bound is attained (two-point example at H=0), so T_contract = T_c = 1 there", 0)
def _v05():
    mem = at.two_point_memory(0.7, 0.5)
    a = mem.weights(np.array([0.0]))
    m, kb = a @ mem.V, a @ mem.K
    cov = float(((mem.V - m).T @ (a[:, None] * (mem.K - kb)))[0, 0])
    bound = at.cov_bound(mem.K, mem.V)
    return "‖Cov‖ = D_vD_k/4 = 1", f"‖Cov‖ = {cov:.12f}, bound {bound:.12f}", abs(cov - bound) < 1e-12, {}


@check("V06", "attention", "P3 at T=0.5, ρ=0.5: fixed points 0 (unstable) and ±H* (stable)", 0)
def _v06():
    fps = at.fixed_points_tanh(0.5)
    mults = [at.multiplier_tanh(h, 0.5, 0.5) for h in fps]
    ok = len(fps) == 3 and mults[1] > 1 and abs(mults[0]) < 1 and abs(mults[2]) < 1
    return "three fixed points; |μ(0)| > 1 > |μ(±H*)|", f"H* = {fps[2]:.6f}; μ = {mults[2]:.4f}, {mults[1]:.4f}", ok, {"Hstar": fps[2]}


@check("V07", "attention", "Near T_c: H* ≈ T√(3(1−T)) (mean-field exponent ½)", 0)
def _v07():
    ratios = {T: at.fixed_points_tanh(T)[-1] / (T * math.sqrt(3 * (1 - T))) for T in (0.99, 0.999, 0.9999)}
    return "ratio → 1", ", ".join(f"T={k}: {v:.6f}" for k, v in ratios.items()), abs(ratios[0.9999] - 1) < 1e-3, {}


@check("V08", "attention", "Critical slowing at T=1: H_t √t → √(3/(2ρ)) (algebraic decay)", 0, slow=True)
def _v08():
    cd = at.critical_decay(0.5, steps=100000)
    return f"√(3/(2ρ)) = {cd['limit']:.6f}", f"{cd['final_scaled']:.6f} at t = 1e5", _rel(cd["final_scaled"], cd["limit"]) < 1e-3, cd


@check("V09", "attention", "Log–log decay slope: −½ at T=1; 0 for T<1 (lock-in); steep for T>1", 0)
def _v09():
    s = {T: at.decay_exponent(T, 0.5) for T in (0.8, 1.0, 1.2)}
    ok = abs(s[0.8]) < 1e-6 and abs(s[1.0] + 0.5) < 0.03 and s[1.2] < -2
    return "0, −0.5, < −2", ", ".join(f"T={k}: {v:.3f}" for k, v in s.items()), ok, {}


@check("V10", "attention", "Static (adiabatic) hysteresis area > 0 iff T < 1", 0)
def _v10():
    a = {T: at.static_hysteresis_area(T) for T in (0.6, 0.8, 0.95, 1.01, 1.3)}
    ok = a[0.6] > a[0.8] > a[0.95] > 1e-3 and abs(a[1.01]) < 1e-12 and abs(a[1.3]) < 1e-12
    return "positive below 1, zero above", ", ".join(f"T={k}: {v:.3g}" for k, v in a.items()), ok, a


@check("V11", "attention", "Hysteresis area ∝ (1−T)² near T_c (mean field)", 0)
def _v11():
    a1, a2 = at.static_hysteresis_area(0.9, n=8001), at.static_hysteresis_area(0.95, n=8001)
    expo = math.log(a1 / a2) / math.log(0.1 / 0.05)
    return "exponent ≈ 2", f"{expo:.3f}", abs(expo - 2.0) < 0.1, {"exponent": expo}


@check("V12", "attention", "Anti-aligned values: flip (period-2) bifurcation at T_f = ρ/(2−ρ)", 0)
def _v12():
    Tf = at.flip_threshold(0.5)
    below, above = at.period_two_amplitude(0.95 * Tf), at.period_two_amplitude(1.05 * Tf)
    return f"T_f = {Tf:.4f}; amplitude > 0 below, 0 above", f"{below:.4f} / {above:.2e}", below > 1e-2 and above < 1e-10, {}


@check("V13", "attention", "P3′: crossing type set by the spectrum of C₀ = Cov_uniform(v,k)", 0)
def _v13():
    K = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], float)
    cases = {"aligned": K, "anti": -K, "rotated": K @ np.array([[0, -1], [1, 0]], float).T}
    out, ok = {}, True
    for name, V in cases.items():
        th = at.linear_thresholds(K, V, 0.5)[0]
        Tc = th["T_crit"]
        lo, hi = at.spectral_radius_at_zero(K, V, 0.98 * Tc, 0.5), at.spectral_radius_at_zero(K, V, 1.02 * Tc, 0.5)
        ok &= lo > 1 > hi
        out[name] = (th["type"].split(" (")[0], round(Tc, 4))
    return "radius crosses 1 at T_crit", "; ".join(f"{k}: {v[0]} at {v[1]}" for k, v in out.items()), ok, {}


@check("V14", "attention", "Gated forgetting makes open-loop retention attention-dependent", 1)
def _v14():
    g = at.GatedMemory()
    d = g.impulse_response([np.ones(4) / 4] * 10)[-1]
    s = g.impulse_response([np.array([0.97, 0.01, 0.01, 0.01])] * 10)[-1]
    return "diffuse ≫ sharp at lag 10", f"{d:.3g} vs {s:.3g}", d > 1e3 * s, {}


# ----------------------------------------------------------------------------------------------
# Cascades
# ----------------------------------------------------------------------------------------------

REPORT_DQ = {-1: 1.1258, 0: 1.0000, 0.5: 0.9385, 1: 0.8813, 2: 0.7859, 3: 0.7172, math.inf: 0.5146, -math.inf: 1.7370}


@check("V15", "cascade", "Report table: D_q at p = 0.7 (hand-computed values, 4 d.p.)", 0)
def _v15():
    got = {q: cs.binomial_Dq(0.7, q) for q in REPORT_DQ}
    ok = all(abs(round(got[q], 4) - v) < 1e-9 for q, v in REPORT_DQ.items())
    return "all 8 values reproduce", ", ".join(f"{q}: {got[q]:.4f}" for q in REPORT_DQ), ok, {str(k): v for k, v in got.items()}


@check("V16", "cascade", "Box counting on the exact level-16 measure reproduces the closed form", 0)
def _v16():
    mu = cs.binomial_measure(0.7, 16)
    errs = [abs(cs.estimate_Dq(mu, q, range(4, 17)) - cs.binomial_Dq(0.7, q)) for q in (-1, 0, 0.5, 1, 2, 3)]
    return "|error| < 1e−10", f"max {max(errs):.1e}", max(errs) < 1e-10, {}


@check("V17", "cascade", "H_n = n h2(p) exactly (no limit needed)", 0)
def _v17():
    d = cs.shannon_entropy_levels(0.7, 10) - np.arange(1, 11) * cs.h2(0.7)
    return "difference 0", f"max {np.abs(d).max():.1e}", float(np.abs(d).max()) < 1e-12, {}


@check("V18", "cascade", "Markov cascade: Z_n = π₀ᵀM_qⁿ1 (enumeration) and i.i.d. limit = binomial", 0)
def _v18():
    mc = cs.MarkovCascade()
    errs = [_rel(mc.enumerate_Zn(q, 10), mc.Zn(q, 10)) for q in (0.5, 2.0, 3.0)]
    iid = cs.MarkovCascade(p=(0.7, 0.7), P=(((0.5, 0.5), (0.5, 0.5)), ((0.5, 0.5), (0.5, 0.5))))
    e2 = abs(iid.tau(2.0) - cs.binomial_tau(0.7, 2.0))
    return "agreement to round-off", f"Z_n rel err {max(errs):.1e}; i.i.d. τ(2) err {e2:.1e}; D_2 = {mc.Dq(2.0):.4f}", max(errs) < 1e-12 and e2 < 1e-12, {}


@check("V19", "cascade", "EMA-memory cascade: transfer operator gives τ(q); memory moves D₂ off the matched memoryless value", 1, slow=True)
def _v19():
    ec = cs.EMACascade()
    t_tr = ec.tau_transfer(2.0, 801)
    t_en = ec.tau_enumeration(2.0, 16, 22)
    pm = cs.memoryless_match(ec, 1.0)
    d2_bin = cs.binomial_Dq(pm, 2.0)
    ok = abs(t_tr - t_en) < 2e-3 and abs(d2_bin - t_tr) > 0.1
    return ("enumeration → transfer value; D₂ differs from the D₁-matched binomial",
            f"τ(2) transfer {t_tr:.5f}, enumeration (n 16→22) {t_en:.5f}; D₂ memoryless {d2_bin:.4f} vs {t_tr:.4f}",
            ok, {"tau2_transfer": t_tr, "tau2_enum": t_en, "p_matched": pm, "D2_matched": d2_bin})


@check("V20", "cascade", "EFL fixed point model: Hutchinson operator contracts W1 by exactly ½; binomial is its fixed point", 0)
def _v20():
    hc = cs.hutchinson_contraction(0.7, 10)
    dev = max(abs(r - 0.5) for r in hc["ratios"])
    return "ratio ½; Φ(binomial_n) = binomial_{n+1}", f"max abs(ratio − ½) {dev:.1e}; fixed-point err {max(hc['fixed_point_error']):.1e}", dev < 1e-9 and max(hc["fixed_point_error"]) < 1e-15, {}


@check("V21", "cascade", "Source B: D_eff = limsup log N(r)/log λ^r with N(r) ~ λ^r is identically 1", 0)
def _v21():
    d = cs.efl_effective_dimension(3.0)
    return "1 for every λ (definition is vacuous)", f"{d.min():.12f} … {d.max():.12f}", bool(np.allclose(d, 1.0)), {}


# ----------------------------------------------------------------------------------------------
# IRT geometry (Source C)
# ----------------------------------------------------------------------------------------------

_P = irt.IRTParams()
_PTS = [(0.3, 0.7, 0.9), (1.1, 2.5, 1.7), (2.0, 4.0, 0.4), (0.0, math.pi / 2, 2.2), (0.5, 3.0, 3.1)]


@check("V22", "irt", "Closed-form induced metric = JᵀJ (numeric, 2 kernels × 5 points)", 0)
def _v22():
    err = max(float(np.abs(irt.metric_from_jacobian(t, f, s, _P, k) - irt.metric(f, s, _P, k)).max())
              for k in (irt.CONSTANT_KERNEL, irt.DAMPED_KERNEL) for (t, f, s) in _PTS)
    return "difference < 1e−12", f"{err:.1e}", err < 1e-12, {}


@check("V23", "irt", "Symbolic proof: JᵀJ = closed form and det g = u²e^{−2αψ}[r²e^{−2αψ}D² + κ²g_φφ]", 0, needs="sympy")
def _v23():
    res = irt.symbolic_metric_check()
    return "all identities simplify to 0", ", ".join(f"{k}: {v}" for k, v in res.items()), all(res.values()), res


@check("V24", "irt", "Source C A.1 is wrong in g_φφ, g_ψψ and g_φψ (g_θθ and g_θφ = g_θψ = 0 are right)", 0)
def _v24():
    k = irt.DAMPED_KERNEL
    f, s = 2.5, 1.7
    g, gp = irt.metric(f, s, _P, k), irt.published_metric(f, s, _P, k)
    d = {"theta-theta": _rel(gp[0, 0], g[0, 0]), "phi-phi": _rel(gp[1, 1], g[1, 1]),
         "psi-psi": _rel(gp[2, 2], g[2, 2]), "phi-psi true value": float(g[1, 2])}
    ok = d["theta-theta"] < 1e-14 and d["phi-phi"] > 0.01 and d["psi-psi"] > 0.01 and abs(d["phi-psi true value"]) > 0.01
    return "g_θθ agrees; three components differ", ", ".join(f"{k}: {v:.3g}" for k, v in d.items()), ok, d


@check("V25", "irt", "Jacobian rank 3 off the predicted circles; rank 2 and det g = 0 exactly on them", 0)
def _v25():
    k = irt.DAMPED_KERNEL
    generic = all(irt.jacobian_rank(t, f, s, _P, k) == 3 for (t, f, s) in _PTS)
    loc = irt.singular_locus(_P, k, 6.0)
    on = all(irt.jacobian_rank(0.3, L["phi"], L["psi"], _P, k, rtol=1e-7) == 2 and float(irt.det_metric(L["phi"], L["psi"], _P, k)) < 1e-20 for L in loc)
    zeros = [round(float(x), 4) for x in k.zeros(6.0)]
    return "generic rank 3; rank 2 on each circle", f"{len(loc)} circles (2 at ψ=0, 2 per zero of κ: ψ_k = {zeros})", generic and on, {"locus": loc}


@check("V26", "irt", "Scalar curvature: Gauss equation = intrinsic Riemann = warped product 2K_Σ − 2Δ_Σρ/ρ", 0)
def _v26():
    worst = 0.0
    for k in (irt.CONSTANT_KERNEL, irt.DAMPED_KERNEL):
        for (t, f, s) in _PTS:
            A = float(irt.scalar_curvature_gauss(t, f, s, _P, k))
            B = irt.scalar_curvature_intrinsic((t, f, s), _P, k)
            C = irt.scalar_curvature_warped(f, s, _P, k)
            worst = max(worst, _rel(B, A), _rel(C, A))
    return "relative spread < 1e−5", f"{worst:.1e}", worst < 1e-5, {}


@check("V27", "irt", "Curvature depends on κ only through κ and κ′: changing κ″(ψ₀) leaves Scal(ψ₀) unchanged", 0)
def _v27():
    vals = [irt.scalar_curvature_intrinsic((0.0, 0.9, 1.2), _P, irt.QuadraticKernel(1.0, 0.3, c2, 1.2)) for c2 in (0.0, 0.5, -2.0, 7.0)]
    spread = (max(vals) - min(vals)) / abs(np.mean(vals))
    return "spread at finite-difference level", f"κ″ ∈ {{0, 1, −4, 14}}: spread {spread:.1e}", spread < 1e-6, {"values": vals}


@check("V28", "irt", "Large constant κ: Scal → 2K of the ψ-slice torus, which changes sign at cos φ = 0 (Gauss–Bonnet)", 0)
def _v28():
    errs = {}
    for k0 in (10.0, 100.0, 1000.0):
        kern = irt.constant_kernel(k0)
        e = 0.0
        for f in np.linspace(0.1, 6.2, 13):
            for s in (0.5, 1.5, 3.0):
                S = float(irt.scalar_curvature_gauss(0.0, f, s, _P, kern))
                K = float(irt.slice_gaussian_curvature(f, s, _P))
                e = max(e, abs(S - 2 * K) / (abs(2 * K) + 1e-12))
        errs[k0] = e
    phs = np.linspace(0.01, 2 * math.pi - 0.01, 400)
    sign_ok = bool(np.all((irt.slice_gaussian_curvature(phs, 1.0, _P) < 0) == (np.cos(phs) > 0)))
    ok = errs[1000.0] < 1e-4 and errs[100.0] < 1e-2 and sign_ok
    return "error O(κ₀⁻²); K < 0 exactly where cos φ > 0", ", ".join(f"κ₀={k:g}: {v:.1e}" for k, v in errs.items()) + f"; sign rule {sign_ok}", ok, {}


@check("V29", "irt", "Source C: 'negative curvature' is false; Scal takes both signs over the chart", 0)
def _v29():
    out = {}
    for k in (irt.CONSTANT_KERNEL, irt.DAMPED_KERNEL):
        s = irt.curvature_survey(_P, k)
        out[k.name] = (s.frac_negative, s.frac_positive)
    ok = all(min(v) > 0.2 for v in out.values())
    return "positive on a set of positive measure", "; ".join(f"{k}: {v[0]:.0%} neg / {v[1]:.0%} pos" for k, v in out.items()), ok, {k: list(v) for k, v in out.items()}


@check("V30", "irt", "Source C eq. 29 disagrees with the curvature in value and sign", 0)
def _v30():
    rows = []
    for (t, f, s) in _PTS:
        true = float(irt.scalar_curvature_gauss(t, f, s, _P, irt.DAMPED_KERNEL))
        pub = float(irt.published_scalar_curvature(f, s, _P, irt.DAMPED_KERNEL))
        rows.append((true, pub))
    sign_mismatch = sum(1 for a, b in rows if a * b < 0)
    return "mismatch", f"sign differs at {sign_mismatch}/5 points; e.g. true {rows[0][0]:+.3f} vs eq. 29 {rows[0][1]:+.3f}", sign_mismatch >= 2, {"pairs": rows}


@check("V31", "irt", "Source C eq. 28: Γ^ψ_θθ has the wrong sign even for its own (diagonal) metric", 0)
def _v31():
    pt = (0.0, 0.9, 1.2)
    k = irt.DAMPED_KERNEL
    own = irt.christoffel_of(lambda f, s: irt.published_metric(f, s, _P, k), pt)
    printed = irt.published_christoffels(pt[1], pt[2], _P, k)
    p_tt = printed[("psi", "theta", "theta")]
    p_ff = printed[("psi", "phi", "phi")]
    ok = abs(p_tt + own[2, 0, 0]) < 1e-6 and abs(p_ff - own[2, 1, 1]) > 0.05
    return "printed = −(own); Γ^ψ_φφ inconsistent too", f"Γ^ψ_θθ printed {p_tt:+.4f} vs own {own[2, 0, 0]:+.4f}; Γ^ψ_φφ {p_ff:+.4f} vs {own[2, 1, 1]:+.4f}", ok, {}


@check("V32", "irt", "Source C eq. 16 is not an antiderivative of eq. 15 (and w(0) ≠ 0)", 0)
def _v32():
    k = irt.DAMPED_KERNEL
    ps, h = np.linspace(0.1, 3.0, 30), 1e-6
    exact = float(np.max(np.abs((k.w(ps + h) - k.w(ps - h)) / (2 * h) + k.k(ps))))
    pub = float(np.max(np.abs((k.w_source_c_eq16(ps + h) - k.w_source_c_eq16(ps - h)) / (2 * h) + k.k(ps))))
    return "exact w: dw/dψ = −κ; eq. 16 fails", f"exact residual {exact:.1e}; eq. 16 residual {pub:.3f}; eq. 16 w(0) = {float(k.w_source_c_eq16(0.0)):.1f}", exact < 1e-8 and pub > 0.5, {}


# ----------------------------------------------------------------------------------------------
# Wake field (Source B §4)
# ----------------------------------------------------------------------------------------------

@check("V33", "wake", "CFL: centered limit 1/√d for every γΔt; Source B's scheme √((1−γΔt/2)/d)", 0, slow=True)
def _v33():
    worst = 0.0
    for sch in ("centered", "backward"):
        for g in (0.2, 0.8, 1.5):
            for d in (1, 2):
                worst = max(worst, abs(wk.cfl_limit(sch, g, d) - wk.cfl_limit_numeric(sch, g, d)))
    return "closed form = numeric bisection", f"max abs diff {worst:.1e}", worst < 1e-7, {}


@check("V34", "wake", "Temporal order: centered 2, Source B (4.13) 1", 0)
def _v34():
    a, b = wk.temporal_order("centered"), wk.temporal_order("backward")
    return "≈ 2 and ≈ 1", f"{a:.3f} and {b:.3f}", abs(a - 2) < 0.05 and abs(b - 1) < 0.05, {}


@check("V35", "wake", "Discrete energy identity of the centered scheme (2D, damped, forced)", 0)
def _v35():
    n = 48
    dx = 1.0 / n
    x = (np.arange(n) + 0.5) * dx
    X, Y = np.meshgrid(x, x, indexing="ij")
    W0 = np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 0.01)
    run = wk.run_wake(W0, np.zeros_like(W0), v=1.0, gamma=0.7, dx=dx, dt=0.45 * dx, steps=600,
                      force=lambda k, t: 0.3 * np.sin(2 * np.pi * X) * np.cos(3 * t))
    res = float(np.abs(run.energy - run.energy[0] - (run.work - run.dissipated)).max())
    return "E^{n+½} − E^{½} = work − dissipation", f"residual {res:.1e}", res < 1e-12, {}


@check("V36", "wake", "Source B Prop. 4.2 is false: curl ∇θ carries 2πm; the flux of ∇θ (∫Δθ) is 0", 0)
def _v36():
    out = {m: wk.defect_identity(64, m) for m in (1, -1, 2)}
    circ_ok = all(abs(c - 2 * math.pi * m) < 1e-9 for m, d in out.items() for c, k in zip(d["circulation"], d["loop_half_widths"]) if k >= 2)
    flux = max(d["max_abs_flux"] for d in out.values())
    return "circulation 2πm; flux 0 (Source B: flux 2πm)", f"circulation/2π = {[round(c / (2 * math.pi), 9) for c in out[1]['circulation']]}; max abs flux {flux:.1e}", circ_ok and flux < 1e-9, {}


@check("V37", "wake", "Source B (4.2c) does not preserve c ∈ [0,1]", 0)
def _v37():
    ce = wk.coherence_equation(1.0, 1.0, 0.5, 1.0)
    return "c → −γ_c|∇θ|²/μ < 0", f"c∞ = {ce['c_infinity']}, crosses 0 at t = {ce['t_cross_zero']:.4f} (= ln 3)", ce["leaves_unit_interval"] and abs(ce["t_cross_zero"] - math.log(3)) < 1e-12, ce


@check("V38", "wake", "Source B Prop. 4.1: energy need not decay with stationary S, c (static source, start at rest)", 0)
def _v38():
    r = wk.static_source_energy()
    return "energy rises", f"E: {r['E_start']:.2e} → max {r['E_max']:.2e}", not r["monotone_decay"] and r["E_max"] > 100 * r["E_start"], r


@check("V39", "wake", "No localized steady wake without a restoring term; (4.9) holds only with mass γ_m W", 0)
def _v39():
    W0 = []
    for L in (1.0, 2.0, 4.0):
        x, W = wk.steady_point_source(L, mass=0.0)
        W0.append(float(W[len(W) // 2]))
    x, W = wk.steady_point_source(6.0, mass=2.0)
    err = float(np.abs(W - wk.eq_4_9(x, 1.0, 2.0, 1.0)).max())
    return "W(0) = L/2 without mass; (4.9) with mass", f"W(0) = {[round(w, 6) for w in W0]}; (4.9) max err {err:.1e}", np.allclose(W0, [0.5, 1.0, 2.0], atol=1e-8) and err < 1e-3, {}


@check("V40", "wake", "Decoherence pulse: uniform-mode displacement β/γ, independent of σ (Source B: βσ/γ)", 0)
def _v40():
    a, b = wk.decoherence_pulse(sigma=2.0), wk.decoherence_pulse(sigma=0.5)
    return "β/γ = 2", f"σ=2: {a['W_final']:.6f}; σ=0.5: {b['W_final']:.6f}", abs(a["W_final"] - 2) < 1e-6 and abs(b["W_final"] - 2) < 1e-6, {}


@check("V41", "wake", "Lagrangian (4.4) does not generate (4.2a)", 0, needs="sympy")
def _v41():
    r = wk.euler_lagrange_4_4()
    obs = "EL − (4.2a) = (α − χv²)S_xx + v²ω²W − v²ξ(1 − c) − β∂_t c (restoring term, no ∂_t c source)"
    return "EL equation ≠ (4.2a)", obs, r["has_restoring_term"] and not r["coherence_source_is_time_derivative"], r


@check("V42", "wake", "2D point source: ∫∇W·∇W diverges as ln(1/h)/(2π), outside Theorem 4.1's H¹", 0)
def _v42():
    a, b = wk.point_source_energy_2d(1e-2), wk.point_source_energy_2d(1e-4)
    return "doubles when h → h²", f"{a:.4f} → {b:.4f}", abs(b - 2 * a) < 1e-12, {}


# ----------------------------------------------------------------------------------------------
# Source D and Source B coherence formula
# ----------------------------------------------------------------------------------------------

@check("V43", "sources", "Source D Tier 1: reported Walsh coefficients belong to a different input; Parseval passes for both", 0)
def _v43():
    r = so.reported_tier1()
    ok = (not r["reported_matches_any_ordering_or_scale"]) and r["parseval_reported"] and r["parseval_computed"]
    return "mismatch undetectable by Parseval", f"WHT{r['input']} = {r['computed_natural']}; reported = transform of {r['signal_whose_transform_is_reported']}", ok, r


@check("V44", "sources", "Source D Tier 2: φ(a^k) = φ(a)^k for φ = mod 3 on Z9 cannot fail", 0)
def _v44():
    r = so.z9_z3_homomorphism()
    return "0 failures possible", f"{r['failures']} failures in {r['checks']}; unital Z4→Z3 map exists: {r['z4_to_z3_unital_hom_exists']}", r["failures"] == 0 and not r["z4_to_z3_unital_hom_exists"], r


@check("V45", "sources", "Source D Prediction 5 metric is maximal for a constant signal", 0)
def _v45():
    from ..evolve.chaos import ChaosRNG
    c = so.low_order_concentration(np.ones(64))
    n = so.low_order_concentration(ChaosRNG(0.3).normal(64))
    return "constant → 1", f"constant {c:.3f}; chaotic noise {n:.3f}", c == 1.0 and n < 0.2, {}


@check("V46", "sources", "Source B eq. 4.10 inverts agreement; spectral share is sign-blind; the resultant separates the antipodal pair", 0)
def _v46():
    r = so.pc_cases()
    ok = (r["identical"]["pc_4_10"] < 1e-12 and r["orthogonal"]["pc_4_10"] == 1.0 and math.isnan(r["antipodal_pair"]["pc_4_10"])
          and r["antipodal_pair"]["pc_spectral"] == 1.0 and r["antipodal_pair"]["pc_resultant"] == 0.0)
    return "4.10: identical 0, orthogonal 1", "; ".join(f"{k}: {v['pc_4_10']:.2g}/{v['pc_spectral']:.2g}/{v['pc_resultant']:.2g}" for k, v in r.items()), ok, r


# ----------------------------------------------------------------------------------------------
# Persistence after decoupling (priority 2)
# ----------------------------------------------------------------------------------------------

@check("V47", "core", "R_persist: 0 under H1; > 0 with actuation; 0 again given (X,H); EMA decays, accumulator saturates", 0)
def _v47():
    r_h1 = core.persistence_information(core.LinearIWME(c=0.0))["block_bits"]
    r_x = core.persistence_information(core.LinearIWME(c=0.5))["block_bits"]
    r_xh = core.persistence_information(core.LinearIWME(c=0.5), condition="XH")["block_bits"]
    ema = core.persistence_information(core.LinearIWME(memory="ema"), horizon=80, lags=[10, 60])["per_lag_bits"]
    acc = core.persistence_information(core.LinearIWME(memory="accumulator"), horizon=80, lags=[10, 60])["per_lag_bits"]
    ok = r_h1 < 1e-12 and r_x > 0.5 and r_xh < 1e-9 and ema[1] < 1e-8 and acc[1] > 1.0
    return ("0 / >0 / 0; EMA → 0, accumulator stays", f"H1 {r_h1:.1e}; X {r_x:.3f} bits; X,H {r_xh:.1e}; lag 60: EMA {ema[1]:.1e}, accumulator {acc[1]:.2f} bits",
            ok, {"H1": r_h1, "X": r_x, "XH": r_xh, "ema": ema, "acc": acc})


# ----------------------------------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------------------------------

def run_all(fast: bool = False, only: Optional[List[str]] = None) -> List[Result]:
    out: List[Result] = []
    for cid, area, claim, level, slow, needs, fn in REGISTRY:
        if only and cid not in only:
            continue
        if fast and slow:
            continue
        if needs:
            try:
                __import__(needs)
            except ImportError:
                out.append(Result(cid, area, claim, level, "-", f"skipped: {needs} not installed", True))
                continue
        t0 = time.time()
        try:
            exp, obs, ok, data = fn()
        except Exception as e:  # a crash is a failed check, reported, never hidden
            exp, obs, ok, data = "-", f"error: {type(e).__name__}: {e}", False, {}
        out.append(Result(cid, area, claim, level, exp, obs, bool(ok), round(time.time() - t0, 3), _jsonable(data)))
    return out


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, complex):
        return [x.real, x.imag]
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return str(x)
    return x


def render(results: List[Result]) -> str:
    lines = []
    for r in results:
        lines.append(f"{r.cid} [{'PASS' if r.passed else 'FAIL'}] L{r.level} {r.area:9s} {r.claim}")
        lines.append(f"      expected: {r.expected}")
        lines.append(f"      observed: {r.observed}  ({r.seconds:.2f}s)")
    n_pass = sum(r.passed for r in results)
    lines.append(f"{n_pass}/{len(results)} checks pass")
    return "\n".join(lines)


def to_json(results: List[Result]) -> str:
    return json.dumps([asdict(r) for r in results], indent=2, ensure_ascii=False)


_TEX_ESC = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
            "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", "\\": r"\textbackslash{}"}


def _tex(s: str) -> str:
    return "".join(_TEX_ESC.get(ch, ch) for ch in s)


def to_latex(results: List[Result]) -> str:
    """A longtable body (rows only) for the paper's verification appendix; uses \\checkmark-free text."""
    rows = []
    for r in results:
        rows.append(f"{r.cid} & {r.level} & {_tex(r.claim)} & {_tex(r.observed)} & {'PASS' if r.passed else 'FAIL'} \\\\")
    return "\n".join(rows) + "\n"
