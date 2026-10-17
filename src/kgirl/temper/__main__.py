"""`python -m kgirl.temper gate|plan` — the model-swap gate and its sample-size planner.

    gate   --eval runs.jsonl [--suite suite.jsonl] [--train sft.jsonl ...] [--no-training]
           [--shadow shadow.jsonl] [--artifact SHA] [--out report.json]
    plan   --delta 0.05 [--discordance 0.2]    |    --items 200 [--discordance 0.2]

File formats (JSON lines):
    runs     {"item", "family", "arm": "base"|"tuned", "seed", "temperature", "valid", "success", "artifact"}
    suite    {"item", "family", "text"}            the held-out items, checked for leakage into training
    train    rows from `python -m kgirl.harness soup export-sft --role ...`
    shadow   {"item", "base", "tuned"}             parsed decisions of both models on unscored traffic

Exit status: 0 ACCEPT, 1 REJECT, 2 INCOMPLETE.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .gate import (GateConfig, evaluate, load_jsonl, minimum_detectable, required_items, train_text)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m kgirl.temper")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gate", help="may the tuned model replace its base? ACCEPT / REJECT / INCOMPLETE")
    g.add_argument("--eval", required=True)
    g.add_argument("--suite")
    g.add_argument("--train", nargs="*", default=None)
    g.add_argument("--no-training", action="store_true", help="the swap involved no fine-tuning (no leakage check)")
    g.add_argument("--shadow")
    g.add_argument("--artifact", help="sha of the served artifact (the GGUF Ollama will serve)")
    g.add_argument("--min-items", type=int, default=GateConfig.min_items)
    g.add_argument("--out")
    p = sub.add_parser("plan", help="paired sample size, or the minimum detectable gain")
    p.add_argument("--delta", type=float)
    p.add_argument("--items", type=int)
    p.add_argument("--discordance", type=float, default=0.2)
    a = ap.parse_args(argv)

    if a.cmd == "plan":
        if (a.delta is None) == (a.items is None):
            ap.error("plan needs exactly one of --delta or --items")
        if a.delta is not None:
            print(f"{required_items(a.delta, a.discordance)} paired items detect a {a.delta:.3f} gain "
                  f"(discordance {a.discordance}, α 0.05 one-sided, 80 % power)")
        else:
            print(f"{a.items} paired items detect a gain of {minimum_detectable(a.items, a.discordance):.3f} "
                  f"(discordance {a.discordance}, α 0.05 one-sided, 80 % power)")
        return 0

    suite = {str(r["item"]): r.get("text", "") for r in load_jsonl(a.suite)} if a.suite else None
    train = [train_text(r) for f in a.train for r in load_jsonl(f)] if a.train else None
    shadow = load_jsonl(a.shadow) if a.shadow else None
    rep = evaluate(load_jsonl(a.eval), GateConfig(min_items=a.min_items), a.artifact, suite, train,
                   trained=not a.no_training, shadow=shadow)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rep, indent=1))
    print(f"verdict: {rep['verdict']}")
    for name, gate in rep["gates"].items():
        state = {True: "pass", False: "FAIL", None: "missing"}[gate["pass"]]
        detail = "; ".join(gate.get("problems", [])) or ", ".join(
            f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in gate.items()
            if k in ("delta", "lower_bound", "p", "method", "rho", "worst_family", "worst_delta", "paired_items"))
        print(f"  {name} {state:<7} {detail}")
    return {"ACCEPT": 0, "REJECT": 1, "INCOMPLETE": 2}[rep["verdict"]]


if __name__ == "__main__":
    sys.exit(main())
