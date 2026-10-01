"""`python -m kgirl.evolve assay` — evolve c-BPE-ECL designs; print elites, emergence stats, optional figure."""

from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="kgirl.evolve")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("assay", help="MAP-Elites over the CN121933729A claim ranges")
    p.add_argument("-g", "--generations", type=int, default=40)
    p.add_argument("-b", "--batch", type=int, default=48)
    p.add_argument("--seed", type=float, default=0.4123)
    p.add_argument("-k", "--top", type=int, default=5)
    p.add_argument("--plot", help="write a figure to this PNG path")
    p.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    from .pareto import front
    from .problems import assay as A
    import numpy as np

    me = A.evolve(a.generations, a.batch, a.seed)
    base = A.baseline()
    champ = me.champion()
    elites = me.elites()
    F = np.array([[e.info["yield"], e.info["viability"], e.info["sensitivity"]] for e in elites])
    pf = [elites[i] for i in front(F)]
    last = me.history[-1]
    result = {
        "generations": a.generations, "coverage": last.coverage, "qd_score": last.qd_score,
        "stepping_stones": me.stepping_stones(), "ancestry_size": len(me.ancestry(champ.id)),
        "baseline_patent_default": {"fitness": base.fitness, **base.info},
        "champion": {"fitness": champ.fitness, "generation": champ.generation, **champ.info},
        "top": [{"fitness": e.fitness, **e.info} for e in elites[:a.top]],
        "pareto_front_size": len(pf),
    }
    if a.plot:
        from .plots import plot_assay_run
        result["figure"] = plot_assay_run(me, base, a.plot)
    if a.json:
        print(json.dumps(result, indent=2, default=float))
        return 0
    print(f"MAP-Elites, {a.generations} generations: coverage {last.coverage:.0%}, QD-score {last.qd_score:.2f}, "
          f"champion fitness {champ.fitness:.3f} (patent default under continuous drive: {base.fitness:.3f})")
    print(f"champion found in generation {champ.generation}; its ancestry ({result['ancestry_size']} elites) "
          f"crossed {result['stepping_stones']} niches; Pareto front (yield, viability, sensitivity): {len(pf)} designs")
    keys = ["tpa_mM", "ru_mM", "probe_ug_ml", "drive_V", "f_nihil", "period_s", "viability", "sensitivity",
            "lod_CEA", "lod_AFP"]
    print("  fitness  " + "  ".join(f"{k:>11}" for k in keys))
    for e in elites[:a.top]:
        print(f"  {e.fitness:7.4f}  " + "  ".join(f"{e.info[k]:>11.3g}" for k in keys))
    if a.plot:
        print(f"figure: {result['figure']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
