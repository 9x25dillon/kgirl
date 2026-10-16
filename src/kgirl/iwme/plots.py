"""Figures for the IWME Phase 0–3 paper (matplotlib, vector PDF).

Palette: the validated reference categorical order (blue, orange, aqua, yellow) for series identity, a
blue ↔ gray ↔ red diverging map for curvature sign, recessive hairline grids, ink-colored text. Every
figure is a function of the package's own computations; nothing is hand-entered.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
RED = "#e34948"
INK, INK2, MUTED, GRID, AXIS, MID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#f0efec"


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.5,
        "lines.solid_capstyle": "round", "legend.frameon": False, "pdf.fonttype": 42,
        "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })
    return plt


def _save(fig, path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return str(path)


def _label(ax, x, y, text, color=INK2, **kw):
    ax.annotate(text, (x, y), xytext=(4, 0), textcoords="offset points", va="center", color=color, fontsize=7, **kw)


# ----------------------------------------------------------------------------------------------

def fig_bifurcation(path):
    from . import attention as at
    plt = _style()
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 2.5))
    Ts = np.linspace(0.3, 1.6, 400)
    hs = np.array([at.fixed_points_tanh(T)[-1] if T < 1 else 0.0 for T in Ts])
    lo = Ts < 1
    a.plot(Ts[lo], hs[lo], color=C1)
    a.plot(Ts[lo], -hs[lo], color=C1)
    a.plot(Ts[~lo], hs[~lo], color=C1, label="stable")
    a.plot(Ts[lo], 0 * Ts[lo], color=C2, linestyle=(0, (4, 2)), label="unstable")
    a.axvline(1.0, color=AXIS, linewidth=0.8)
    a.set_xlabel("attention temperature T")
    a.set_ylabel("memory equilibria H*")
    a.set_title("(a) pitchfork at T$_c$ = 1", loc="left")
    a.legend(loc="upper right")
    rho = 0.5
    for T, col, name in ((0.8, C1, "T = 0.8: lock-in"), (1.0, C2, "T = 1: t$^{-1/2}$"), (1.2, C3, "T = 1.2: exponential")):
        traj = np.abs(at.two_point_memory(T, rho).run(np.array([0.5]), 3000)[1:, 0])
        t = np.arange(1, len(traj) + 1)
        keep = traj > 1e-12
        b.loglog(t[keep], traj[keep], color=col, label=name)
    t = np.array([10, 3000])
    b.loglog(t, math.sqrt(3 / (2 * rho)) / np.sqrt(t), color=MUTED, linewidth=0.8, linestyle=(0, (2, 2)))
    _label(b, 3000, math.sqrt(3) / math.sqrt(3000) * 0.6, "√(3/(2ρt))", color=MUTED)
    b.set_xlabel("step t")
    b.set_ylabel("|H$_t$|")
    b.set_title("(b) relaxation from H$_0$ = 0.5, ρ = 0.5", loc="left")
    b.set_ylim(1e-6, 2)
    b.legend(loc="lower left")
    fig.tight_layout()
    return _save(fig, path)


def fig_retention(path):
    from . import attention as at
    plt = _style()
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 2.5))
    rho = 0.5
    Ts = np.concatenate([np.linspace(0.3, 0.995, 200), np.linspace(1.005, 4.0, 200)])
    tau = []
    for T in Ts:
        h = at.fixed_points_tanh(T)[-1] if T < 1 else 0.0
        tau.append(at.effective_time_constant(at.multiplier_tanh(h, T, rho)))
    tau = np.array(tau)
    lo = Ts < 1
    a.semilogy(Ts[lo], tau[lo], color=C1, label="closed loop, at the stable state")
    a.semilogy(Ts[~lo], tau[~lo], color=C1)
    a.axhline(at.effective_time_constant(1 - rho), color=C2, linestyle=(0, (4, 2)), label="open loop: −1/ln(1−ρ)")
    a.axvline(1.0, color=AXIS, linewidth=0.8)
    a.set_xlabel("attention temperature T")
    a.set_ylabel("retention time (steps)")
    a.set_title("(a) attention sets retention in closed loop", loc="left")
    a.legend(loc="upper right")
    bs = np.linspace(-1, 1, 1201)
    for T, col in ((0.6, C1), (0.8, C2), (1.2, C3)):
        up = at._equilibrium_branch(T, bs, -1.0)
        down = at._equilibrium_branch(T, bs[::-1], 1.0)[::-1]
        b.plot(bs, up, color=col, label=f"T = {T}")
        b.plot(bs, down, color=col)
    b.set_xlabel("input bias b")
    b.set_ylabel("adiabatic memory state H")
    b.set_title("(b) hysteresis only below T$_c$", loc="left")
    b.legend(loc="lower right")
    fig.tight_layout()
    return _save(fig, path)


def fig_cascades(path):
    from . import cascade as cs
    plt = _style()
    fig, ax = plt.subplots(figsize=(4.4, 2.7))
    qs = np.linspace(-4, 6, 81)
    qs = qs[np.abs(qs - 1) > 1e-9]
    ax.plot(qs, [cs.binomial_Dq(0.7, q) for q in qs], color=C1, label="binomial p = 0.7 (closed form)")
    mu = cs.binomial_measure(0.7, 16)
    qb = np.array([-4, -2, 0, 2, 4, 6], float)
    ax.plot(qb, [cs.estimate_Dq(mu, q, range(4, 17)) for q in qb], linestyle="none", marker="o", markersize=4,
            color=C1, markeredgecolor="white", markeredgewidth=0.8, label="box counting, level 16")
    mc = cs.MarkovCascade()
    ax.plot(qs, [mc.Dq(q) for q in qs], color=C2, label="Markov-modulated (transfer matrix)")
    ec = cs.EMACascade()
    qe = np.linspace(-4, 6, 41)
    qe = qe[np.abs(qe - 1) > 1e-9]
    ax.plot(qe, [ec.tau_transfer(q, 401) / (q - 1) for q in qe], color=C3, label="EMA memory (transfer operator)")
    pm = cs.memoryless_match(ec, 1.0)
    ax.plot(qs, [cs.binomial_Dq(pm, q) for q in qs], color=C4, linestyle=(0, (4, 2)), label=f"memoryless, same D$_1$ (p = {pm:.3f})")
    ax.set_xlabel("moment order q")
    ax.set_ylabel("Rényi dimension D$_q$")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return _save(fig, path)


def fig_irt(path):
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    from . import irt
    plt = _style()
    cmap = LinearSegmentedColormap.from_list("div", [C1, MID, RED])
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7), sharey=True)
    p = irt.IRTParams()
    for ax, kern, title in ((axes[0], irt.CONSTANT_KERNEL, "(a) κ ≡ 1"), (axes[1], irt.DAMPED_KERNEL, "(b) κ = e$^{-0.2ψ}$cos 2ψ")):
        s = irt.curvature_survey(p, kern, n_phi=241, n_psi=201, psi_range=(0.05, 4.0))
        z = np.sign(s.scal) * np.log10(1 + np.abs(s.scal))
        lim = float(np.nanmax(np.abs(z)))
        im = ax.pcolormesh(s.phi, s.psi, z.T, cmap=cmap, norm=TwoSlopeNorm(0.0, -lim, lim), shading="auto", rasterized=True)
        ax.contour(s.phi, s.psi, np.nan_to_num(s.scal.T), levels=[0.0], colors=INK2, linewidths=0.6)
        if kern is irt.DAMPED_KERNEL:
            for pk in kern.zeros(4.0):
                ax.axhline(pk, color=MUTED, linewidth=0.5, linestyle=(0, (2, 2)))
            loc = [L for L in irt.singular_locus(p, kern, 4.0) if L["psi"] > 0]
            ax.plot([L["phi"] for L in loc], [L["psi"] for L in loc], linestyle="none", marker="x", color=INK, markersize=5)
        ax.set_xlabel("φ")
        ax.set_title(title, loc="left")
        ax.set_xticks([0, math.pi / 2, math.pi, 3 * math.pi / 2, 2 * math.pi])
        ax.set_xticklabels(["0", "π/2", "π", "3π/2", "2π"])
        ax.grid(False)
    axes[0].set_ylabel("ψ")
    cb = fig.colorbar(im, ax=axes, shrink=0.9, pad=0.02)
    cb.set_label("sign(Scal)·log$_{10}$(1+|Scal|)")
    cb.outline.set_edgecolor(AXIS)
    return _save(fig, path)


def fig_wake_persist(path):
    from . import core, wake as wk
    plt = _style()
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 2.5))
    g = np.linspace(0, 1.99, 300)
    cen = np.array([wk.cfl_limit("centered", x, 2) for x in g])
    bwd = np.array([wk.cfl_limit("backward", x, 2) for x in g])
    a.plot(g, cen, color=C1, label="centered damping")
    a.plot(g, bwd, color=C2, label="Source B eq. 4.13")
    a.fill_between(g, bwd, cen, color=C2, alpha=0.10, linewidth=0)
    a.text(1.05, 0.52, "stable only\nwith centering", color=INK2, fontsize=7)
    a.set_xlabel("γΔt")
    a.set_ylabel("largest stable vΔt/Δx (d = 2)")
    a.set_title("(a) damping tightens Source B's CFL limit", loc="left")
    a.set_ylim(0, 0.8)
    a.legend(loc="lower left")
    lags = list(range(1, 81))
    for mem, col, name in (("ema", C1, "EMA memory (pole 1−ρ)"), ("accumulator", C2, "accumulator (pole 1, IRT w)")):
        r = core.persistence_information(core.LinearIWME(memory=mem), horizon=80, lags=lags)["per_lag_bits"]
        r = np.array(r)
        keep = r > 1e-14
        b.semilogy(np.array(lags)[keep], r[keep], color=col, label=name)
    b.set_xlabel("lag ℓ after decoupling")
    b.set_ylabel("R$_{persist}$(ℓ) (bits)")
    b.set_title("(b) persistence after decoupling", loc="left")
    b.legend(loc="lower left")
    fig.tight_layout()
    return _save(fig, path)


ALL = {
    "fig_bifurcation": fig_bifurcation,
    "fig_retention": fig_retention,
    "fig_cascades": fig_cascades,
    "fig_irt": fig_irt,
    "fig_wake_persist": fig_wake_persist,
}
