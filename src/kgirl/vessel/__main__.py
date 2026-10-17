"""CLI: python -m kgirl.vessel {demo, status, verify, invariance, soulpack}.

The default database is $KGIRL_HOME/vessel/matrix.db (KGIRL_HOME defaults to ~/.kgirl).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def default_db() -> str:
    return str(Path(os.environ.get("KGIRL_HOME", Path.home() / ".kgirl")) / "vessel" / "matrix.db")


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    from ._glyph import knocks, reveal
    if knocks(argv):
        print(reveal() + "hello, world.")
        return 0
    ap = argparse.ArgumentParser(prog="python -m kgirl.vessel", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo", help="whole lifecycle on the reference substrate")
    d.add_argument("--scale", choices=("quick", "full"), default="quick")
    d.add_argument("--db", default=":memory:", help="persist the demo vessels here (must not hold them already)")
    d.add_argument("--json", action="store_true")
    s = sub.add_parser("status", help="ledger and last readings from the database alone")
    s.add_argument("name", nargs="?")
    s.add_argument("--db", default=default_db())
    v = sub.add_parser("verify", help="recompute a vessel's hash chain; exit 1 if broken")
    v.add_argument("name")
    v.add_argument("--db", default=default_db())
    a = sub.add_parser("autopilot", help="run the vessel unattended at a recursive-self-improvement level (RSI slider)")
    a.add_argument("--rsi", type=int, choices=range(5), default=1,
                   help="0 observe · 1 steer · 2 learn · 3 consolidate · 4 self-tune (cumulative)")
    a.add_argument("--cycles", type=int, default=8)
    a.add_argument("--push", type=int, default=60, help="world disturbance: unanchored foreign steps per cycle")
    a.add_argument("--sweep", action="store_true", help="run every level 0–4 on the same world")
    a.add_argument("--db", default=":memory:", help="persist the run's ledger here (single level only)")
    a.add_argument("--json", action="store_true")
    i = sub.add_parser("invariance", help="verify the signature's invariance group on a trained ReferenceLM")
    i.add_argument("--samples", type=int, default=8)
    p = sub.add_parser("soulpack", help="export a vessel's soulpack (CarryOn 0.1.x) as JSON")
    p.add_argument("name")
    p.add_argument("--db", default=default_db())
    args = ap.parse_args(argv)

    if args.cmd == "demo":
        from .demo import render, run
        r = run(args.scale, args.db, log=lambda m: print(f"· {m}", file=sys.stderr))
        print(json.dumps(r, indent=2, default=str) if args.json else render(r))
        return 0

    if args.cmd == "autopilot":
        from .rsi_demo import render, run, sweep
        if args.cycles < 1:
            ap.error("--cycles must be at least 1")
        say = lambda m: print(f"· {m}", file=sys.stderr)  # noqa: E731
        results = sweep(cycles=args.cycles, push=args.push, log=say) if args.sweep else \
            [run(args.rsi, args.cycles, push=args.push, db_path=args.db)]
        print(json.dumps(results if args.sweep else results[0], indent=2, default=str) if args.json
              else "\n\n".join(render(r) for r in results))
        return 0

    from .matrixdb import MatrixDB
    if args.cmd in ("status", "verify", "soulpack"):
        if not Path(args.db).exists():
            print(f"no vessel database at {args.db}", file=sys.stderr)
            return 2
        with MatrixDB(args.db) as db:
            if args.cmd == "status":
                from .demo import db_status
                print(json.dumps(db_status(db, args.name), indent=2, default=str))
                return 0
            if args.cmd == "verify":
                r = db.verify(args.name)
                print(json.dumps(r, indent=2))
                return 0 if r["ok"] else 1
            from .persona import Genome, genome_for, soulpack
            g = db.genome(args.name)
            print(json.dumps(soulpack(args.name, Genome.from_dict(g) if g else genome_for(args.name),
                                      db_path=args.db, ledger_head=db.head(args.name)), indent=2))
            return 0

    if args.cmd == "invariance":
        from . import corpus as C
        from .invariance import verify
        from .substrate import Objective, ReferenceLM, RefConfig
        m = ReferenceLM(RefConfig())
        X, y = C.windows(C.prose(0.11, 300), 8)
        for b in C.batches(X, y, 128, 0.3, 300):
            m.step(b, Objective(lr=3e-3))
        r = verify(m, C.probe_set(8, 192, 0.1414), samples=args.samples)
        print(r.render())
        return 0 if r.verified else 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
