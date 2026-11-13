"""Command line for the package. Run from the `research/` directory: python -m efl_iwme_cascade <command>."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional, Sequence


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="efl_iwme_cascade", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("checks", help="recompute the Level 0 claims the package relies on")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("part1", help="Part 1, partial observability (registered Experiment A)")
    p.add_argument("--config", default="part1_registered")
    p.add_argument("--jobs", type=int, default=1, help="worker processes; results do not depend on it")
    p.add_argument("--resume", action="store_true", help="reuse stages cached by an interrupted run")
    p.add_argument("--no-save", action="store_true")

    p = sub.add_parser("part2", help="Part 2, layered coupling (the EFL arm)")
    p.add_argument("--config", default="part2_registered")
    p.add_argument("--no-save", action="store_true")

    p = sub.add_parser("data", help="measured points of [0, 1), one per line: the Part 1 comparison")
    p.add_argument("file")
    p.add_argument("--depth", type=int, default=16)
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("layers", help="measured scale signals, one column per scale: the Part 2 analysis")
    p.add_argument("file")
    p.add_argument("--json", action="store_true")

    a = ap.parse_args(argv)
    log = lambda m: print(m, file=sys.stderr, flush=True)  # noqa: E731

    if a.cmd == "checks":
        from . import checks
        results = checks.run_all()
        print(json.dumps([r.__dict__ for r in results], indent=1, ensure_ascii=False) if a.json else checks.render(results))
        return 0 if all(r.passed for r in results) else 1

    if a.cmd == "part1":
        from .experiments import partial_observability as part1
        res = part1.run(part1.load_config(a.config), log=log, jobs=a.jobs, use_cache=a.resume)
        if not a.no_save:
            part1.save(res)
        print(part1.render(res))
        return 0 if all(v["met"] for v in res["verdicts"].values()) else 1

    if a.cmd == "part2":
        from .experiments import layered_coupling as part2
        res = part2.run(part2.load_config(a.config), log=log)
        if not a.no_save:
            part2.save(res)
        print(part2.render(res))
        return 0 if all(v["met"] for v in res["verdicts"].values()) else 1

    import numpy as np

    from .experiments import real_data
    from .experiments.partial_observability import plain
    if a.cmd == "data":
        rep = real_data.analyze_points(np.loadtxt(a.file, dtype=float).ravel(), a.depth)
    else:
        rep = real_data.analyze_layers(np.loadtxt(a.file, dtype=float))
    print(json.dumps(plain(rep), indent=1) if a.json else real_data.render(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
