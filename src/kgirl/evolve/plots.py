"""Figures for evolution runs (dark lab style shared with kgirl.nihiline.plots)."""

from __future__ import annotations

from ..nihiline.plots import BG, CYAN, GOLD, GREEN, LEGEND, PINK, _save, plt, style


def plot_assay_run(me, baseline, path) -> str:
    fig = plt.figure(figsize=(13, 9))
    fig.patch.set_facecolor(BG)
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.28)
    d0, d1 = me.descriptors
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(me.grid().T, origin="lower", aspect="auto", cmap="magma",
                   extent=[d0.lo, d0.hi, d1.lo, d1.hi])
    c = me.champion()
    ax.scatter([c.behavior[0]], [c.behavior[1]], s=120, marker="*", color=GOLD, edgecolor="white", label="champion")
    ax.scatter([baseline.behavior[0]], [baseline.behavior[1]], s=70, marker="X", color=PINK, edgecolor="white",
               label="patent default, continuous drive")
    ax.set_xlabel(d0.name)
    ax.set_ylabel(d1.name)
    ax.set_title("Archive — best design per (viability, sensitivity) niche")
    ax.legend(**LEGEND)
    plt.colorbar(im, ax=ax)
    style(ax)
    ax = fig.add_subplot(gs[0, 1])
    gens = [h.generation for h in me.history]
    ax.plot(gens, [h.coverage for h in me.history], color=GREEN, lw=1.4, label="coverage")
    ax2 = ax.twinx()
    ax2.plot(gens, [h.qd_score for h in me.history], color=CYAN, lw=1.4, label="QD-score")
    ax.set_xlabel("generation")
    ax.set_ylabel("coverage", color=GREEN)
    ax2.set_ylabel("QD-score", color=CYAN)
    ax2.tick_params(colors=CYAN)
    ax.set_title("Emergence — niches filled and summed quality")
    style(ax)
    ax = fig.add_subplot(gs[1, 0])
    ax.bar(gens, [h.new_cells for h in me.history], color=GREEN, label="new niche")
    ax.bar(gens, [h.improvements for h in me.history], bottom=[h.new_cells for h in me.history], color=PINK,
           label="elite improved")
    ax.set_xlabel("generation")
    ax.set_ylabel("innovations")
    ax.set_title("Innovation events per generation")
    ax.legend(**LEGEND)
    style(ax)
    ax = fig.add_subplot(gs[1, 1])
    names = ["tpa_mM", "ru_mM", "probe_ug_ml", "drive_V", "f_nihil", "period_s"]
    top = me.elites()[:12]
    for rank, e in enumerate(top):
        u = [me.space.params[i].encode(e.info[n]) for i, n in enumerate(names)]
        ax.plot(range(len(names)), u, color=GOLD if rank == 0 else CYAN, alpha=1.0 if rank == 0 else 0.35,
                lw=2.0 if rank == 0 else 1.0)
    ub = [me.space.params[i].encode(baseline.info[n]) for i, n in enumerate(names)]
    ax.plot(range(len(names)), ub, color=PINK, ls="--", lw=1.4, label="patent default")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, fontsize=8)
    ax.set_ylabel("position in claim range (0 = min, 1 = max)")
    ax.set_title("Top-12 genomes (gold = champion)")
    ax.legend(**LEGEND)
    style(ax)
    return _save(fig, path, "Evolving c-BPE-ECL designs inside the CN121933729A claims")
