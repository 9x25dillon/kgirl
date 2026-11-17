"""python -m kgirl.harness.helix run | state | history | diff | verify | ingest | floor | audit"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..hermes.router import Router
from ..llm import BackendError
from .evaluate import evaluate
from .harness import Helix, critic_chain
from .store import HelixError, StateStore


def render(r: dict) -> str:
    """A run for a person: the answer first, then the flight recorder in a few lines."""
    led = r["ledger"]
    out = [r["answer"], "", "-" * 72]
    if r["bypass"]:
        out.append(f"BYPASS  {r['bypass']}")
    else:
        m = r["models"]
        out.append(f"experts {' + '.join(r['experts'])} -> CRITIC -> SYNTHESIST   state v{r['versions'][0]} -> "
                   f"v{r['versions'][1]}")
        out.append(f"models  generator {m['generator']}, critic {m['critic']} on {m['critic_backend']}"
                   + ("" if m["decorrelated"] else "  (same family)"))
        out.extend(f"        passed over {s['critic']}: {s['why']}" for s in m["critics_skipped"])
    out.append(f"{'node':<11}{'kind':<11}{'target':>7}{'ext':>7}{'est':>7}{'in':>7}  stop")
    for n in led["nodes"]:
        ext = "-" if n["ext"] is None else n["ext"]
        out.append(f"{n['node']:<11}{n['kind']:<11}{n['target']:>7}{ext:>7}{n['est']:>7}"
                   f"{'-' if n['in_ext'] is None else n['in_ext']:>7}  {n['stop']}")
    spent = led["spent_ext"] if led["spent_ext"] is not None else f"{led['spent_est']} EST"
    out.append(f"budget  B {led['B']}  spent {spent}{'  OVERRUN' if led['overrun'] else ''}  "
               f"steering {led['steer_spent']}/{led['pools']['steer']}  answer share {led['efficiency']}  "
               f"est/ext {led['est_over_ext']}")
    for t in r["tunes"]:
        out.append(f"tune    {t['axis']} {t['old']:.2f} -> {t['new']:.2f} (v{t['v']}): \"{t['quote']}\""
                   if t["accepted"] else f"tune    rejected: {t['reason']}")
    if not r["bypass"]:
        kept = "" if r["unresolved_stated"] else "  (synthesis did not say, so all are kept)"
        out.append(f"open    {len(r['open_conflicts'])} conflict(s){kept}")
        out.extend(f"  - {c}" for c in r["open_conflicts"])
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m kgirl.harness.helix", description=__doc__)
    ap.add_argument("--home", help="harness home (default $KGIRL_HOME, else ~/.kgirl)")
    ap.add_argument("--session", default="main", help="whose STATE to use (default main)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run", help="answer a task within a budget of generated tokens")
    p.add_argument("task")
    p.add_argument("-B", "--budget", type=int, default=4000)
    p.add_argument("--expert", action="append", help="name an expert (repeatable) instead of routing by intent")
    p.add_argument("--lattice", action="store_true", help="two experts instead of one")
    p.add_argument("--expect", type=int, help="expected size of the answer in tokens, for the BYPASS predicate")
    p.add_argument("--critic-model", help="Ollama model for the critic (default $KGIRL_HELIX_CRITIC_MODEL)")
    p.add_argument("--critic-host", help="try the critic on this Ollama host first, e.g. a phone "
                                         "(default $KGIRL_HELIX_CRITIC_HOST); this machine is the fallback")
    p.add_argument("--json", action="store_true", help="the whole result, log included")
    p = sub.add_parser("state", help="the current STATE, or one version of it")
    p.add_argument("-v", type=int, dest="version")
    sub.add_parser("history", help="every version with the diff that made it")
    p = sub.add_parser("diff", help="what changed between two versions")
    p.add_argument("a", type=int)
    p.add_argument("b", type=int)
    sub.add_parser("verify", help="recompute the hash chains")
    p = sub.add_parser("ingest", help="take a handed-off STATE from a file (- for stdin); validated, fail-closed")
    p.add_argument("file")
    sub.add_parser("floor", help="the scaffold floor the BYPASS predicate uses, and where it came from")
    p = sub.add_parser("audit", help="audit a HELIX-9 transcript a model wrote on its own (- for stdin)")
    p.add_argument("file")
    p.add_argument("--external", type=int, help="the run's token count from the provider or a tokenizer")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "audit":                              # reads a file, touches no store
            text = sys.stdin.read() if args.file == "-" else Path(args.file).read_text(encoding="utf-8")
            report = evaluate(text, args.external)
            print(json.dumps(report, indent=1, ensure_ascii=False))
            return 0 if report["verdict"] == "PASS" else 1
        with StateStore(args.home) as store:
            if args.cmd == "run":
                critic = critic_chain(args.critic_model, args.critic_host) if args.critic_model or args.critic_host else None
                r = Helix(Router.from_env(), store=store, session=args.session, critic=critic).run(
                    args.task, args.budget, experts=args.expert, lattice=args.lattice, expect=args.expect)
                print(json.dumps(r, indent=1, ensure_ascii=False) if args.json else render(r))
                return 0
            if args.cmd == "state":
                s = store.get(args.session, args.version) if args.version is not None else store.head(args.session)
                if s is None:
                    raise HelixError(f"session {args.session!r} has no such state")
                out = s.to_dict()
            elif args.cmd == "history":
                out = store.history(args.session)
            elif args.cmd == "diff":
                out = store.diff(args.session, args.a, args.b)
            elif args.cmd == "verify":
                out = store.verify()
            elif args.cmd == "floor":
                out = store.floor()
            else:
                raw = sys.stdin.read() if args.file == "-" else Path(args.file).read_text(encoding="utf-8")
                v = store.ingest(args.session, raw, source=args.file)
                out = {"accepted": v.ok, "violations": list(v.violations), "clamped": list(v.clamped),
                       "state": v.state.to_dict()}
                print(json.dumps(out, indent=1, ensure_ascii=False))
                return 0 if v.ok else 1
    except (HelixError, BackendError, OSError, ValueError) as exc:
        print(f"helix: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(out, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
