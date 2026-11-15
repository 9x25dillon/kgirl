"""CLI verbs for kgirl's mind: chat, build, write, evolve, convo. Wired into cli.py."""

from __future__ import annotations

import sys
from pathlib import Path

CHAT_HELP = """commands:
  /new [title]      start a new conversation        /recap          what this conversation is about
  /list             recent conversations            /open ID        switch to conversation ID
  /recall QUERY     search memory and old chats     /remember TEXT  store a fact or preference now
  /build            spec -> tested code (paste the spec, end with a line '.')
  /accept [note]    keep the last build as a learned exemplar
  /write BRIEF      long-form text                  /reflect        reflect on this conversation now
  /evolve           consolidate memory now          /quit           reflect and leave (also Ctrl-D)"""


def add_parsers(sub) -> None:
    p = sub.add_parser("chat", help="talk with kgirl: remembers, uses tools, learns from your reactions")
    p.add_argument("message", nargs="*", help="one message, then exit (omit for an interactive session)")
    p.add_argument("--new", action="store_true", help="start a new conversation instead of resuming the latest")
    p.add_argument("--conv", type=int, help="resume conversation ID")
    p.add_argument("--quiet", action="store_true", help="hide tool and memory events")
    p = sub.add_parser("build", help="spec -> tested Python module (tests first, samples, agreement, repair)")
    p.add_argument("spec", help="spec file, or - for stdin")
    p.add_argument("-o", "--out", help="write the module here (and its tests beside it as test_<name>.py)")
    p.add_argument("-n", type=int, default=6, help="implementations to sample (default 6)")
    p.add_argument("-k", type=int, default=3, help="independent test suites (default 3)")
    p.add_argument("--accept", action="store_true", help="store the result as an active exemplar right away")
    p = sub.add_parser("write", help="long-form writing: plan, draft by section, critique, revise")
    p.add_argument("brief", help="what to write, or @file to read the brief from a file")
    p.add_argument("-w", "--words", type=int)
    p.add_argument("-o", "--out")
    p.add_argument("--no-revise", action="store_true")
    p = sub.add_parser("evolve", help="consolidate: reflect, curate habits, rewrite the user profile")
    p.add_argument("--dataset", help="also export approved exchanges as chat JSONL for fine-tuning")
    p = sub.add_parser("convo", help="conversation history")
    cs = p.add_subparsers(dest="convo_cmd", required=True)
    cs.add_parser("list")
    q = cs.add_parser("show")
    q.add_argument("id", type=int)
    q = cs.add_parser("search")
    q.add_argument("query")
    q = cs.add_parser("delete")
    q.add_argument("id", type=int)


def run(a, args) -> int | None:
    """Handle a mind verb; None means `args.cmd` isn't one."""
    if args.cmd not in ("chat", "build", "write", "evolve", "convo"):
        return None
    from .mind import Mind
    mind = Mind(a)
    try:
        if args.cmd == "chat":
            return _chat(mind, args)
        if args.cmd == "build":
            return _build(mind, args)
        if args.cmd == "write":
            return _write(mind, args)
        if args.cmd == "evolve":
            from .mind import evolve
            rep = evolve(mind._llm_text, a.soup, a.curator, mind.store, args.dataset)
            print(rep.render(a.soup))
            return 0
        return _convo(mind, args)
    finally:
        mind.store.close()


# ---------------------------------------------------------------------- chat

def _chat(mind, args) -> int:
    conv = args.conv or (mind.start() if args.new else mind.resume())
    event = (lambda m: None) if args.quiet else (lambda m: print(f"  · {m}", file=sys.stderr, flush=True))
    out = lambda piece: print(piece, end="", flush=True)  # noqa: E731
    if args.message:
        mind.chat(conv, " ".join(args.message), on_text=out, on_event=event)
        return 0
    c = mind.store.get(conv)
    print(f"kgirl · conversation #{conv}{' · ' + c.title if c and c.title else ''} · /help for commands",
          file=sys.stderr)
    if c and c.summary:
        print(f"  (so far: {c.summary[:300]})", file=sys.stderr)
    while True:
        try:
            line = input("\n› ").strip()
        except (EOFError, KeyboardInterrupt):
            print(file=sys.stderr)
            break
        if not line:
            continue
        if line.startswith("/"):
            cmd, _, rest = line[1:].partition(" ")
            if cmd in ("quit", "exit", "q"):
                break
            conv = _slash(mind, conv, cmd, rest.strip(), event)
            continue
        try:
            print()
            mind.chat(conv, line, on_text=out, on_event=event)
        except Exception as exc:  # noqa: BLE001 - keep the session alive
            print(f"\n(error: {exc})", file=sys.stderr)
    print("  · reflecting before exit…", file=sys.stderr)
    mind.end(conv)
    return 0


def _slash(mind, conv: int, cmd: str, rest: str, event) -> int:
    a = mind.a
    if cmd == "help":
        print(CHAT_HELP)
    elif cmd == "new":
        mind.end(conv)
        conv = mind.start(rest)
        print(f"new conversation #{conv}")
    elif cmd == "list":
        for c in mind.store.list(15):
            print(f"#{c.id:<5} {c.title or '(untitled)':<40} {c.summary[:80]}")
    elif cmd == "open" and rest.isdigit() and mind.store.get(int(rest)):
        conv = int(rest)
        print(mind.recap(conv))
    elif cmd == "recap":
        print(mind.recap(conv))
    elif cmd == "recall":
        print(mind._tool("recall", {"query": rest}, event))
    elif cmd == "remember" and rest:
        fid = a.remember(rest, kind="preference" if rest.lower().startswith(("i ", "always", "never", "don")) else "fact",
                         tags="conversation")
        print(f"remembered #{fid}")
    elif cmd == "reflect":
        ref = mind.reflector.reflect(conv, force=True)
        print(f"summary: {ref.summary}\nproposed: {ref.proposed}" if ref else "nothing new to reflect on")
    elif cmd == "evolve":
        from .mind import evolve
        print(evolve(mind._llm_text, a.soup, a.curator, mind.store).render(a.soup))
    elif cmd == "build":
        print("paste the spec; end with a line containing only '.'")
        lines = []
        while True:
            try:
                ln = input()
            except EOFError:
                break
            if ln.strip() == ".":
                break
            lines.append(ln)
        res = mind.builder(workers=4).build("\n".join(lines))
        mind.last_build = res
        print(res.render())
        if res.code:
            print(f"\n```python\n{res.code}```\n(/accept to keep it as an exemplar kgirl learns from)")
        mind.store.append(conv, "tool", f"[build] {res.render()}\n{res.code}")
    elif cmd == "accept":
        if mind.last_build is None:
            print("no build in this session")
        else:
            fid = mind.builder().learn(mind.last_build, accepted=True, note=rest)
            print(f"stored exemplar #{fid}; future builds with similar specs will see it")
    elif cmd == "write" and rest:
        from .mind import Writer
        ctx = a.soup.recall(rest, budget_tokens=500, kinds=("preference", "fact", "episode")).render()
        d = Writer(mind._llm_text).write(rest, ctx, progress=event)
        print(d.markdown())
        mind.store.append(conv, "tool", f"[write] {d.markdown()}")
    else:
        print(CHAT_HELP)
    return conv


# ---------------------------------------------------------------------- build / write / convo

def _build(mind, args) -> int:
    spec = sys.stdin.read() if args.spec == "-" else Path(args.spec).read_text(encoding="utf-8")
    res = mind.builder(n_impl=args.n, n_suites=args.k).build(spec)
    print(res.render(), file=sys.stderr)
    if not res.code:
        return 1
    if args.out:
        out = Path(args.out)
        out.write_text(res.code, encoding="utf-8")
        tests = out.with_name(f"test_{out.stem}.py")
        tests.write_text(res.tests.replace("from solution import", f"from {out.stem} import")
                         .replace("import solution", f"import {out.stem} as solution"), encoding="utf-8")
        print(f"wrote {out} and {tests}", file=sys.stderr)
    else:
        print(res.code)
    fid = mind.builder().learn(res, accepted=args.accept)
    print(f"exemplar #{fid} {'active' if args.accept else 'staged (kgirl soup promote ' + str(fid) + ' to keep it)'}",
          file=sys.stderr)
    return 0


def _write(mind, args) -> int:
    from .mind import Writer
    brief = Path(args.brief[1:]).read_text(encoding="utf-8") if args.brief.startswith("@") else args.brief
    ctx = mind.a.soup.recall(brief, budget_tokens=500, kinds=("preference", "fact", "episode")).render()
    d = Writer(mind._llm_text).write(brief, ctx, args.words, revise=not args.no_revise,
                                     progress=lambda m: print(f"  · {m}", file=sys.stderr, flush=True))
    if args.out:
        Path(args.out).write_text(d.markdown(), encoding="utf-8")
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(d.markdown())
    return 0


def _convo(mind, args) -> int:
    s = mind.store
    if args.convo_cmd == "list":
        for c in s.list(50):
            print(f"#{c.id:<5} turns={s.count(c.id):<4} {c.title or '(untitled)':<40} {c.summary[:90]}")
    elif args.convo_cmd == "show":
        print(mind.recap(args.id), end="\n\n")
        for t in s.turns(args.id):
            print(f"[{t.role}] {t.content}\n")
    elif args.convo_cmd == "search":
        for t in s.search(args.query, limit=20):
            print(f"#{t.conv_id} {t.role}: {' '.join(t.content.split())[:160]}")
    elif args.convo_cmd == "delete":
        s.delete(args.id)
        print(f"deleted conversation #{args.id}")
    return 0
