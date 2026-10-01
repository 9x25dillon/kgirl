"""Figures in the lab scripts' dark style. Every function writes one PNG and returns its path."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BG, EDGE, TICK, LABEL, TITLE = "#0a0a0f", "#6a6a8a", "#b8b8d0", "#d8d8ee", "#e8e8ff"
PINK, CYAN, GOLD, GREEN, VIOLET = "#ff5aa0", "#5ad8ff", "#ffd35a", "#5affc0", "#a060ff"
LEGEND = dict(facecolor="#15151f", edgecolor="#5a5a7a", labelcolor=LABEL, fontsize=8)


def style(ax):
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_color(EDGE)
    ax.tick_params(colors=TICK)
    ax.xaxis.label.set_color(LABEL)
    ax.yaxis.label.set_color(LABEL)
    ax.title.set_color(TITLE)
    ax.grid(True, alpha=0.15, color="#5a5a7a")


def _save(fig, path, title) -> str:
    fig.suptitle(title, color=TITLE, fontsize=12, y=0.995)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_calibration(path: str | Path) -> str:
    from . import calibration as cal
    from .patent import MARKERS
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.patch.set_facecolor(BG)
    for ax, mode in zip(axes, ("conc", "cells")):
        for (name, m), col in zip(MARKERS.items(), (PINK, CYAN)):
            c = m.conc_curve if mode == "conc" else m.cell_curve
            x = np.logspace(math.log10(c.lod) - 1, math.log10(c.hi), 200)
            ax.semilogx(x, [cal.signal(c, v) for v in x], color=col, lw=1.4, label=f"{name}")
            ax.axhline(cal.scar(c), color=col, lw=0.7, ls=":", alpha=0.8)
            ax.axvline(c.lod, color=col, lw=0.7, ls="--", alpha=0.6)
        ax.set_xlabel(f"{'antigen (ng/mL)' if mode == 'conc' else 'MCF-7 (cells/mL)'}")
        ax.set_ylabel("ECL intensity (a.u.)")
        ax.set_title(f"{'Example 4' if mode == 'conc' else 'Example 6'} curves — dotted: scar on zero")
        ax.legend(**LEGEND)
        style(ax)
    return _save(fig, path, "CN121933729A calibration — every LOD reveals the blank's residue")


def plot_gap(path: str | Path) -> str:
    from .amplification import chain
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor(BG)
    for name, col in (("CEA", PINK), ("AFP", CYAN)):
        ch = chain(name)
        ax.plot(range(len(ch.stages)), [math.log10(max(s.value, 1e-30)) for s in ch.stages], "o-", color=col,
                lw=1.4, label=f"{name}: LOD {ch.lod_cells_per_ml:.3g} vs reported {ch.reported_lod_cells_per_ml:.3g} cells/mL")
        names = [s.name for s in ch.stages]
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("log10(quantity per cell)")
    ax.set_title("Amplification chain — antigen → probes → MB → e⁻ → photons → counts")
    ax.legend(**LEGEND)
    style(ax)
    return _save(fig, path, "Criticality bridge, re-aimed: can one MCF-7 cell light the anode?")


def plot_pulse(path: str | Path) -> str:
    from .pulse import control_nihil_sweep, leverage_sweep, ratio_sweep
    lev, joint, rat = leverage_sweep(), control_nihil_sweep(), ratio_sweep()
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.patch.set_facecolor(BG)
    ax = axes[0, 0]
    ax.plot(lev["f_nihil"], lev["product"], color=GOLD, lw=1.6, marker="o", ms=4)
    ax.axvline(lev["f_opt"], color="#ff4060", ls="--", lw=1, label=f"optimal nihil f* = {lev['f_opt']:.2f}")
    ax.set_xlabel("nihil fraction of each 5 V pulse")
    ax.set_ylabel("yield × reproducibility × viability")
    ax.set_title("Master curve — the dose of nothing")
    ax.legend(**LEGEND)
    style(ax)
    ax = axes[0, 1]
    for key, col, lbl in (("yield_", CYAN, "yield"), ("reproducibility", PINK, "reproducibility"),
                          ("viability", GREEN, "viability")):
        ax.plot(lev["f_nihil"], lev[key], color=col, lw=1.2, label=lbl)
    ax.set_xlabel("nihil fraction")
    ax.set_title("What nothing buys")
    ax.legend(**LEGEND)
    style(ax)
    ax = axes[1, 0]
    im = ax.imshow(joint["product"], aspect="auto", origin="lower", cmap="magma",
                   extent=[joint["f"].min(), joint["f"].max(), joint["C"].min(), joint["C"].max()])
    ax.plot(joint["f_opt_by_C"], joint["C"], color=GREEN, marker="o", ms=4, lw=1.4, label=joint["verdict"])
    ax.set_xlabel("nihil fraction f")
    ax.set_ylabel("control C (TPA 20→70 mM)")
    ax.set_title("Control × nihil landscape")
    ax.legend(**LEGEND)
    plt.colorbar(im, ax=ax)
    style(ax)
    ax = axes[1, 1]
    ax.plot(rat["ratio"], rat["pattern"], color=GREEN, lw=1.2, label="pattern (1 − CV)")
    ax.plot(rat["ratio"], rat["yield"] / max(rat["yield"].max(), 1e-12), color=CYAN, lw=1.2, label="yield (norm.)")
    ax.set_xlabel("pulse rate / ECL relaxation rate")
    ax.set_title("Pattern vs recursion")
    ax.legend(**LEGEND)
    style(ax)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return _save(fig, path, "Nihil leverage of the c-BPE drive")


def plot_bloom(path: str | Path) -> str:
    from .bloom import run
    r = run()
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.patch.set_facecolor(BG)
    ax = axes[0]
    sig2 = 4.0
    for k, col in (("Q", GREEN), ("M", GOLD), ("A", PINK)):
        ax.plot(r.t, r.kappa[k] * sig2, color=col, lw=0.6, label=k)
    ax.set_ylabel("κ(t) σ²")
    ax.set_title("MCF-7 states on the cathode — quiescent / mitotic / apoptotic")
    ax.legend(**LEGEND)
    style(ax)
    ax = axes[1]
    ax2 = ax.twinx()
    ax.plot(r.t, r.phase_variance, color=PINK, lw=1.1, label="σ²_φ phase variance")
    ax2.plot(r.t, r.mean_intensity, color=CYAN, lw=1.1, label="⟨|B̃|²⟩")
    if r.delta_t is not None:
        ax.axvspan(r.t_phase_breach, r.t_amp_collapse, color=GOLD, alpha=0.12)
        ax.set_title(f"Viability sentinel — phase warns {r.delta_t:.0f} s before intensity collapses")
    ax.set_xlabel("time in the drive (s)")
    style(ax)
    for s in ax2.spines.values():
        s.set_color(EDGE)
    ax2.tick_params(colors=CYAN)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return _save(fig, path, "Bloom bridge, re-aimed: is the read killing the cells?")


def plot_array(path: str | Path) -> str:
    from .array import ArrayParams, characterize, lyapunov, phase_diagram, wrapped_charge
    from dataclasses import replace
    pd = phase_diagram()
    vortex = replace(ArrayParams(), J_ang=0.42, J_ax=0.5, seam_phase=np.pi * 0.98, sigma_noise=0.18,
                     A_drive=0.42, g_cubic=0.04)
    v = characterize(vortex)
    fig = plt.figure(figsize=(13, 9))
    fig.patch.set_facecolor(BG)
    gs = fig.add_gridspec(2, 4, hspace=0.45, wspace=0.4)
    ext = [pd["sigma"].min(), pd["sigma"].max(), pd["A"].min(), pd["A"].max()]
    for j, (key, lbl, cmap) in enumerate((("R", "synchrony R", "magma"), ("S", "spectral entropy", "viridis"),
                                          ("xi", "crosstalk length ξ", "plasma"), ("defects", "dark/hot spots", "inferno"))):
        ax = fig.add_subplot(gs[0, j])
        im = ax.imshow(pd[key], aspect="auto", origin="lower", extent=ext, cmap=cmap)
        ax.set_xlabel("noise σ")
        ax.set_ylabel("drive A")
        ax.set_title(lbl, fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.05)
        style(ax)
    ax = fig.add_subplot(gs[1, :2])
    psi = v["final"]
    ax.imshow(np.abs(psi).T, aspect="auto", origin="lower", cmap="magma")
    q = wrapped_charge(psi, vortex.seam_phase)
    ys, xs = np.nonzero(q[:-1])
    ax.scatter(ys + 0.5, xs + 0.5, c=[GREEN if q[a, b] > 0 else PINK for a, b in zip(ys, xs)], s=40,
               edgecolor="white", linewidth=0.6)
    half = psi.shape[0] / 2 - 0.5
    ax.axvline(half, color=GREEN, ls="--", lw=1)
    ax.set_xlabel("lane (CEA | AFP)")
    ax.set_ylabel("position along channel")
    ax.set_title(f"vortex regime: {v['defects']:.1f} defects (time mean), lane RSD CEA {v['lane_rsd']['CEA']:.3f} "
                 f"AFP {v['lane_rsd']['AFP']:.3f}", fontsize=9)
    style(ax)
    ax = fig.add_subplot(gs[1, 2:])
    for name, params, col in (("standard", ArrayParams(), CYAN), ("vortex", vortex, PINK)):
        ly = lyapunov(params)
        d = ly["divergence"]
        ok = d > 1e-14
        ax.semilogy(np.where(ok)[0], d[ok], color=col, lw=1, label=f"{name}: λ ≈ {ly['lambda']:+.4f} ({ly['regime']})")
    ax.set_xlabel("iteration")
    ax.set_ylabel("‖ψa − ψb‖")
    ax.set_title("read robustness — twin trajectories")
    ax.legend(**LEGEND)
    style(ax)
    return _save(fig, path, "Phase-topology lab, re-aimed: the c-BPE array chip")


ALL = {"calibration": plot_calibration, "gap": plot_gap, "pulse": plot_pulse, "bloom": plot_bloom,
       "array": plot_array}
