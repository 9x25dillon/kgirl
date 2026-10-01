"""Command line: `python -m kgirl.harness <command>` (PYTHONPATH=src from the repo root)."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

from .assistant import Assistant
from .atlas import architecture_card, blast_radius, coupling, render_card
from .jev import SwarmConfig
from .report import atlas_report


def _assistant(args) -> Assistant:
    return Assistant(home=args.home, swarm_config=SwarmConfig(keep_workspaces=getattr(args, "keep", False)))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="kgirl.harness", description="repo-aware personal coding assistant")
    ap.add_argument("--home", help="state dir (default $KGIRL_HOME or ~/.kgirl)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("index", help="index local paths, owner/repo or git URLs into the atlas")
    p.add_argument("sources", nargs="+")
    p.add_argument("--refresh", action="store_true", help="re-fetch cached clones")
    sub.add_parser("repos", help="list indexed repos")
    p = sub.add_parser("search", help="symbol search across repos")
    p.add_argument("query")
    p.add_argument("--repo")
    p.add_argument("-n", type=int, default=15)
    p = sub.add_parser("card", help="architecture card of a repo")
    p.add_argument("repo")
    p.add_argument("--summarize", action="store_true", help="ask the scout model for a prose summary")
    p = sub.add_parser("blast", help="blast radius of a file or symbol")
    p.add_argument("target")
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("coupling", help="rank repos by coupling into a hub repo")
    p.add_argument("--hub", default="kgirl")
    p = sub.add_parser("report", help="write the markdown atlas report")
    p.add_argument("--hub", default="kgirl")
    p.add_argument("-o", "--out", default="-")
    p = sub.add_parser("ask", help="ask about your repos")
    p.add_argument("question")
    p.add_argument("--repo")
    p = sub.add_parser("task", help="run a Jev swarm on a coding goal")
    p.add_argument("goal")
    p.add_argument("--repo", required=True, help="path of the repo to work on")
    p.add_argument("--verify", help="verification command, e.g. 'python -m pytest -q tests/test_x.py'")
    p.add_argument("-n", "--size", type=int, default=3)
    p.add_argument("--apply", action="store_true", help="write the accepted diff into the repo")
    p.add_argument("--keep", action="store_true", help="keep agent workspaces for inspection")
    p = sub.add_parser("soup", help="memory pool")
    ss = p.add_subparsers(dest="soup_cmd", required=True)
    q = ss.add_parser("add")
    q.add_argument("text")
    q.add_argument("--kind", default="note")
    q.add_argument("--tags", default="")
    q.add_argument("--scope", default="*")
    q = ss.add_parser("recall")
    q.add_argument("query")
    q.add_argument("--budget", type=int, default=600)
    q.add_argument("--scope")
    ss.add_parser("stats")
    ss.add_parser("curate")
    q = ss.add_parser("promote", help="make a staged fragment (for example one written over MCP) recallable")
    q.add_argument("id", type=int)
    q = ss.add_parser("ledger")
    q.add_argument("-n", type=int, default=30)
    q = ss.add_parser("export-sft", help="accepted trajectories as JSONL for local fine-tuning")
    q.add_argument("out")
    sub.add_parser("route", help="show which backend serves each role")
    p = sub.add_parser("trace", help="tail the Hermes trace")
    p.add_argument("-n", type=int, default=40)
    sub.add_parser("mcp", help="serve the harness over MCP (stdio)")

    args = ap.parse_args(argv)
    a = _assistant(args)
    try:
        return _run(a, args)
    finally:
        a.close()


def _run(a: Assistant, args) -> int:
    cmd = args.cmd
    if cmd == "index":
        for st in a.index(args.sources, refresh=args.refresh):
            print(f"{st.repo:<40} files={st.scanned:<5} changed={st.changed:<5} removed={st.removed:<4} "
                  f"symbols+={st.symbols}")
    elif cmd == "repos":
        for r in a.atlas.repos():
            print(f"{r['name']:<40} files={r['nfiles']:<5} symbols={r['nsymbols']:<6} {r['root']}")
    elif cmd == "search":
        for h in a.atlas.search(args.query, args.n, args.repo):
            print(f"{h.cite():<70} {h.kind:<8} {h.signature[:90]}")
    elif cmd == "card":
        if args.summarize:
            print(a.summarize_repo(args.repo), end="\n\n")
        print(render_card(architecture_card(a.atlas, args.repo)))
    elif cmd == "blast":
        rep = blast_radius(a.atlas, args.target, args.depth)
        print(json.dumps(rep.to_dict(), indent=2) if args.json else rep.render(80))
    elif cmd == "coupling":
        for c in coupling(a.atlas, args.hub):
            print(f"{c.repo:<40} score={c.score:>7.1f} imp->={c.imports_from:<3} imp<-={c.imports_into:<3} "
                  f"clones={c.clones:<5} names={c.shared_names:<4} exposed={c.hub_files_exposed}")
    elif cmd == "report":
        text = atlas_report(a.atlas, args.hub)
        if args.out == "-":
            print(text)
        else:
            Path(args.out).write_text(text + "\n", encoding="utf-8")
            print(f"wrote {args.out}")
    elif cmd == "ask":
        ans = a.ask(args.question, args.repo)
        print(ans.text)
        if ans.citations:
            print(f"\n[{ans.intent} via {ans.backend}; ~{ans.context_tokens} ctx tokens] "
                  + ", ".join(ans.citations[:10]))
    elif cmd == "task":
        verify = shlex.split(args.verify) if args.verify else None
        res = a.task(args.goal, args.repo, verify, size=args.size, apply=args.apply)
        print(res.render())
        return 0 if res.winner else 1
    elif cmd == "soup":
        return _soup(a, args)
    elif cmd == "route":
        for role, chain in a.router.describe().items():
            print(f"{role:<6} -> " + "  |  ".join(chain))
    elif cmd == "trace":
        path = a.home / "trace.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines()[-args.n:] if path.exists() else []
        for ln in lines:
            e = json.loads(ln)
            print(f"{e['topic']:<22} {e['sender']:<10} corr={e['correlation_id'] or e['id']} "
                  f"{json.dumps(e['payload'], ensure_ascii=False)[:120]}")
    elif cmd == "mcp":
        from .mcp_server import MCPServer
        MCPServer(a).serve()
    return 0


def _soup(a: Assistant, args) -> int:
    c = args.soup_cmd
    if c == "add":
        print(f"#{a.soup.add(args.text, args.kind, args.tags, 'user', args.scope)}")
    elif c == "recall":
        rec = a.soup.recall(args.query, args.budget, scope=args.scope)
        print(rec.render() or "(nothing)")
        print(f"[{rec.tokens}/{rec.budget} tokens]")
    elif c == "stats":
        print(json.dumps(a.soup.stats(), indent=2))
    elif c == "promote":
        a.soup.set_status(args.id, "active", "promoted by user")
        print(f"#{args.id} active")
    elif c == "curate":
        rep = a.curator.step()
        print(f"promoted={rep.promoted} retired={rep.retired} expired={rep.expired}")
    elif c == "ledger":
        for r in a.soup.ledger(limit=args.n):
            print(f"{r['seq']:>6} {r['op']:<10} #{r['fragment_id'] or '-':<5} {r['before'] or ''}->{r['after'] or ''} "
                  f"{r['detail'] or ''}")
    elif c == "export-sft":
        n = 0
        with open(args.out, "w", encoding="utf-8") as fh:
            for f in a.soup.all(kind="trajectory"):
                if f.status == "retired" or not f.data:
                    continue
                rec = json.loads(f.data)
                if not rec.get("diff"):
                    continue
                fh.write(json.dumps({"instruction": rec["goal"], "context": ", ".join(rec.get("files", [])),
                                     "output": rec["diff"], "meta": {"ops": rec.get("ops"), "fragment": f.id}},
                                    ensure_ascii=False) + "\n")
                n += 1
        print(f"wrote {n} examples to {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
