"""`python -m kgirl.evolve assay|fuel` — evolve designs; print elites, emergence stats, optional figure.

    assay   c-BPE-ECL assay designs inside the CN121933729A claim ranges
    fuel    PWR fuel-rod enrichment patterns on the NuRea_sim lattice-physics surrogate
"""

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
    p = sub.add_parser("fuel", help="MAP-Elites over 39 rod enrichments (NuRea_sim surrogate)")
    p.add_argument("-g", "--generations", type=int, default=300)
    p.add_argument("-b", "--batch", type=int, default=128)
    p.add_argument("--bins", type=int, default=10)
    p.add_argument("--seed", type=float, default=0.4123)
    p.add_argument("-k", "--top", type=int, default=5)
    p.add_argument("--data", help="lattice dataset directory (default: NuRea_sim via kgirl.paths)")
    p.add_argument("--out", help="write every elite (with its 39 enrichments) to this JSON file")
    p.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "fuel":
        return _fuel(a)

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


def _fuel(a) -> int:
    from .problems import fuel as Fu
    me, model = Fu.evolve(a.generations, a.batch, a.seed, bins=a.bins, data_dir=a.data)
    last, champ, elites = me.history[-1], me.champion(), me.elites()
    real_best = min(model.data_best.values(), key=lambda r: r["pppf"])
    result = {"generations": a.generations, "coverage": last.coverage, "qd_score": last.qd_score,
              "stepping_stones": me.stepping_stones(),
              "best_simulated_design": real_best,
              "champion": {"fitness": champ.fitness, "generation": champ.generation, **champ.info},
              "versus_simulated_per_cell": Fu.compare_to_data(me, model),
              "top": [{"fitness": e.fitness, **{k: v for k, v in e.info.items() if k != "enrichments"}}
                      for e in elites[:a.top]]}
    if a.out:
        with open(a.out, "w") as fh:
            json.dump({**result, "elites": [{"fitness": e.fitness, "cell": list(e.cell), **e.info} for e in elites]},
                      fh, indent=1, default=float)
    if a.json:
        print(json.dumps(result, indent=2, default=float))
        return 0
    cmp = result["versus_simulated_per_cell"]
    print(f"MAP-Elites, {a.generations} generations: coverage {last.coverage:.0%}, "
          f"{me.stepping_stones()} niches crossed by the champion's ancestry")
    print(f"champion: predicted PPPF {champ.info['pppf']:.4f}, k_inf {champ.info['k_inf']:.4f}, mean enrichment "
          f"{champ.info['mean_enrichment']:.3f} wt% (trust {champ.info['trust_sd']:.2f} SD); best simulated design: "
          f"PPPF {real_best['pppf']:.4f}, k_inf {real_best['k_inf']:.4f}")
    print(f"versus the best simulated design in the same cell: elite predicted flatter in "
          f"{cmp['cells_where_elite_flatter']} of {cmp['cells_compared']} cells (median ΔPPPF {cmp['median_pppf_delta']:+.4f})")
    print("predictions are Level 2 until a lattice code simulates them; PPPF surrogate RMSE ≈ 0.063")
    print("  fitness    k_inf    PPPF  mean_e  max_e  trust")
    for e in elites[:a.top]:
        i = e.info
        print(f"  {e.fitness:7.4f}  {i['k_inf']:.4f}  {i['pppf']:.4f}  {i['mean_enrichment']:.3f}  "
              f"{i['max_enrichment']:.2f}  {i['trust_sd']:.2f}")
    if a.out:
        print(f"elites written to {a.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
