"""CLI verbs for kgirl's mind: chat, build, write, evolve, convo, console, ui. Wired into cli.py."""

from __future__ import annotations

import shlex
import sys
from pathlib import Path

VERBS = ("chat", "build", "write", "evolve", "convo", "console", "ui")

CHAT_HELP = """conversation
  /new [title]      start a new conversation        /recap          what this conversation is about
  /list             recent conversations            /open ID        switch to conversation ID
  /paste            a multi-line message (end with a line '.')
  /export [FILE]    this conversation as Markdown   /quit           reflect and leave (also Ctrl-D)
memory
  /recall QUERY     search memory and old chats     /remember TEXT  store a fact or preference now
  /mem QUERY        matching memories with their ids
  /forget ID        retire a memory                 /good  /bad     your verdict on the last answer
  /reflect          reflect on this conversation    /evolve         consolidate memory now
project (the directory chat was started in, or --dir)
  /diff             what kgirl changed this session /undo           take back its last change
  /auto on|off      apply edits without asking      /allow CMD      let kgirl run CMD without asking
  /verify CMD       the command that proves a swarm's change (for the task tool)
  /apply            write the last swarm's verified diff into the project
making things
  /build            spec -> tested code (paste the spec, end with a line '.')
  /accept [note]    keep the last build as a learned exemplar
  /write BRIEF      long-form text                  /tools          what kgirl can call"""


def add_parsers(sub) -> None:
    p = sub.add_parser("chat", help="talk with kgirl: remembers, uses tools, works in your project, learns from "
                                    "your reactions")
    p.add_argument("message", nargs="*", help="one message, then exit (omit for an interactive session)")
    p.add_argument("--new", action="store_true", help="start a new conversation instead of resuming the latest")
    p.add_argument("--conv", type=int, help="resume conversation ID")
    p.add_argument("--quiet", action="store_true", help="hide tool and memory events")
    p.add_argument("--dir", help="project directory kgirl may read and (with your approval) change; "
                                 "default: the current directory")
    p.add_argument("--no-workspace", action="store_true", help="no file or command tools, only indexed repos and memory")
    p.add_argument("--yes", action="store_true", help="apply kgirl's file edits without asking (they stay undoable)")
    p.add_argument("--verify", help="command that proves a code change, e.g. 'python3 -m pytest -q' (for the task tool)")
    p = sub.add_parser("build", help="spec -> tested Python module (tests first, samples, agreement, repair)")
    p.add_argument("spec", help="spec file, or - for stdin")
    p.add_argument("-o", "--out", help="write the module here (and its agreed tests beside it as test_<name>*.py)")
    p.add_argument("-n", type=int, default=6, help="implementations to sample (default 6)")
    p.add_argument("-k", type=int, default=3, help="independent test suites (default 3)")
    p.add_argument("-r", "--repair-rounds", type=int, default=2, help="targeted repair rounds (default 2)")
    p.add_argument("--explain", action="store_true",
                   help="show generated suites, candidate votes, repair steps, and why the winner won")
    p.add_argument("--repo", help="indexed Atlas repository name when building against existing repo files")
    p.add_argument("--context", action="append", default=[],
                   help="existing repo path (with --repo) or local file to build against (may be repeated)")
    p.add_argument("--accept", action="store_true", help="store the result as an active exemplar right away")
    p.add_argument("--no-exemplars", action="store_true", help="build without recalling Soup exemplars")
    p.add_argument("--allow-io", action="store_true", help="the spec needs file access: don't drop candidates for it")
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
    q = cs.add_parser("export", help="one conversation as Markdown")
    q.add_argument("id", type=int)
    q.add_argument("-o", "--out")
    q = cs.add_parser("delete", help="delete a conversation, its summary and the proposals only it supported")
    q.add_argument("id", type=int)
    p = sub.add_parser("console", help="local web console: chat, specialists, memory, atlas, forge, vitals, matrix, "
                                       "services — one live surface over the whole harness")
    p.add_argument("--port", type=int, default=8765, help="localhost port (default 8765)")
    p.add_argument("--dir", help="project directory kgirl may read and (with your approval) change")
    p.add_argument("--no-workspace", action="store_true", help="no file or command tools, only indexed repos and memory")
    p.add_argument("--yes", action="store_true", help="apply kgirl's file edits without asking (they stay undoable)")
    p = sub.add_parser("ui", help="terminal front end: chat | status, jobs and code search (needs textual)")
    p.add_argument("--dir", help="project directory kgirl may read and (with your approval) change")
    p.add_argument("--no-workspace", action="store_true", help="no file or command tools, only indexed repos and memory")
    p.add_argument("--yes", action="store_true", help="apply kgirl's file edits without asking (they stay undoable)")
    p.add_argument("--new", action="store_true", help="start a new conversation instead of resuming the latest")
    p.add_argument("--conv", type=int, help="resume conversation ID")


def run(a, args) -> int | None:
    """Handle a mind verb; None means `args.cmd` isn't one."""
    if args.cmd not in VERBS:
        return None
    if args.cmd == "ui":
        from .ui import run as run_ui
        return run_ui(a, args)
    if args.cmd == "console":
        from .console import serve
        root = None
        if not args.no_workspace:
            cand = Path(args.dir).expanduser() if args.dir else Path.cwd()
            if cand.resolve() not in (Path.home().resolve(), Path(cand.anchor)):
                root = cand
        return serve(a, port=args.port, workspace_root=root, auto_edit=args.yes)
    from .mind import Mind
    try:
        mind = Mind(a, workspace=_workspace(args), verify_cmd=_argv(getattr(args, "verify", None)))
    except ValueError as exc:
        print(f"kgirl: {exc}", file=sys.stderr)
        return 2
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


def _argv(command: str | None) -> list[str] | None:
    return shlex.split(command) if command and command.strip() else None


def _workspace(args):
    """The project directory for a chat: --dir, else the current directory unless that is the home folder
    or the filesystem root (too broad to hand to an agent)."""
    if args.cmd != "chat" or args.no_workspace:
        return None
    from .mind import Workspace
    root = Path(args.dir).expanduser() if args.dir else Path.cwd()
    if not args.dir and root.resolve() in (Path.home().resolve(), Path(root.anchor)):
        return None
    interactive = sys.stdin.isatty() and not args.message
    return Workspace(root, confirm=_confirm if interactive else None, auto_edit=args.yes,
                     notify=None if args.quiet else lambda text: print(f"  · {text}", file=sys.stderr, flush=True))


def _confirm(question: str, detail: str) -> bool:
    """Ask the person at the terminal. Anything but y/yes is a no."""
    print(f"\n{detail.rstrip()}", file=sys.stderr)
    try:
        return input(f"{question} [y/N] ").strip().lower() in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        print(file=sys.stderr)
        return False


def _history(home: Path):
    """Line editing and a persistent input history when readline exists. Returns the function that saves it."""
    try:
        import readline
    except ImportError:
        return lambda: None
    path = str(home / "chat_history")
    try:
        readline.read_history_file(path)
    except OSError:
        pass
    readline.set_history_length(2000)

    def save() -> None:
        try:
            readline.write_history_file(path)
        except OSError:
            pass
    return save


def _read_block(hint: str) -> str:
    print(f"{hint}; end with a line containing only '.'", file=sys.stderr)
    lines = []
    while True:
        try:
            ln = input()
        except (EOFError, KeyboardInterrupt):
            break
        if ln.strip() == ".":
            break
        lines.append(ln)
    return "\n".join(lines)


# ---------------------------------------------------------------------- chat

def _chat(mind, args) -> int:
    if args.conv and not mind.store.get(args.conv):
        print(f"no conversation #{args.conv} (see `convo list`)", file=sys.stderr)
        return 1
    conv = args.conv or (mind.start() if args.new else mind.resume())
    event = (lambda m: None) if args.quiet else (lambda m: print(f"  · {m}", file=sys.stderr, flush=True))
    out = lambda piece: print(piece, end="", flush=True)  # noqa: E731
    if args.message:
        mind.background = False      # a reflection that is due runs before the process exits, not in a thread
        mind.chat(conv, " ".join(args.message), on_text=out, on_event=event)
        return 0
    save_history = _history(mind.a.home)
    c = mind.store.get(conv)
    where = f" · {mind.workspace.root}" if mind.workspace else ""
    print(f"kgirl · conversation #{conv}{' · ' + c.title if c and c.title else ''}{where} · /help for commands",
          file=sys.stderr)
    if c and c.summary:
        print(f"  (so far: {c.summary[:300]})", file=sys.stderr)
    while True:
        try:
            line = input("\n› ").strip()
        except EOFError:
            print(file=sys.stderr)
            break
        except KeyboardInterrupt:
            print("\n  (Ctrl-D or /quit to leave)", file=sys.stderr)
            continue
        if not line:
            continue
        if line.startswith("/"):
            cmd, _, rest = line[1:].partition(" ")
            if cmd in ("quit", "exit", "q"):
                break
            if cmd == "paste":
                line = _read_block("paste your message")
                if not line.strip():
                    continue
            else:
                try:
                    conv = _slash(mind, conv, cmd, rest.strip(), event)
                except Exception as exc:  # noqa: BLE001 - keep the session alive
                    print(f"(error: {exc})", file=sys.stderr)
                continue
        try:
            print()
            mind.chat(conv, line, on_text=out, on_event=event)
        except KeyboardInterrupt:
            print("\n  (interrupted; your message is kept, the answer is not)", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001 - keep the session alive
            print(f"\n(error: {exc})", file=sys.stderr)
    save_history()
    print("  · reflecting before exit…", file=sys.stderr)
    try:
        mind.end(conv)
    except KeyboardInterrupt:
        print("  · reflection skipped", file=sys.stderr)
    return 0


def _slash(mind, conv: int, cmd: str, rest: str, event) -> int:
    a, ws = mind.a, mind.workspace
    no_ws = "no project directory is attached (start chat inside one, or pass --dir)"
    if cmd == "new":
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
    elif cmd == "export":
        text = mind.store.transcript(conv)
        if rest:
            Path(rest).expanduser().write_text(text, encoding="utf-8")
            print(f"wrote {rest}")
        else:
            print(text)
    elif cmd == "recall":
        print(mind.call_tool("recall", {"query": rest}, event))
    elif cmd == "remember" and rest:
        fid = a.remember(rest, kind="preference" if rest.lower().startswith(("i ", "always", "never", "don")) else "fact",
                         tags="conversation")
        print(f"remembered #{fid}")
    elif cmd == "mem" and rest:
        hits = a.soup.similar(rest, limit=10)
        for f, _ in hits:
            print(f"#{f.id:<5} {f.status:<7} {f.kind:<11} u={f.utility:.2f}  {f.text[:110]}")
        if not hits:
            print("no memory matches")
    elif cmd == "forget" and rest.isdigit():
        frag = a.soup.get(int(rest))
        if frag is None:
            print(f"no memory #{rest}")
        else:
            a.soup.set_status(frag.id, "retired", "forgotten by the user")
            print(f"retired #{frag.id}: {frag.text[:100]}")
    elif cmd in ("good", "bad"):
        print(mind.feedback(conv, "positive" if cmd == "good" else "negative"))
    elif cmd == "reflect":
        ref = mind.reflector.reflect(conv, force=True)
        print(f"summary: {ref.summary}\nproposed: {ref.proposed}" if ref else "nothing new to reflect on")
    elif cmd == "evolve":
        from .mind import evolve
        print(evolve(mind._llm_text, a.soup, a.curator, mind.store).render(a.soup))
    elif cmd == "tools":
        for t in mind.tools().values():
            print(f"{t.name:<12} {t.help}")
    elif cmd == "diff":
        print((ws.diff() or "kgirl has changed nothing in this session") if ws else no_ws)
    elif cmd == "undo":
        print(ws.undo() if ws else no_ws)
    elif cmd == "auto" and rest in ("on", "off"):
        if ws:
            ws.auto_edit = rest == "on"
        print((f"edits now apply {'without asking (still undoable with /undo)' if rest == 'on' else 'only after you approve them'}")
              if ws else no_ws)
    elif cmd == "allow" and rest:
        print(f"kgirl may now run: {shlex.join(ws.allow(rest))}" if ws else no_ws)
    elif cmd == "verify":
        if rest:
            mind.verify_cmd = shlex.split(rest)
        print(f"verifier: {shlex.join(mind.verify_cmd)}" if mind.verify_cmd else
              "no verifier set; /verify python3 -m pytest -q (or your project's test command)")
    elif cmd == "apply":
        print(mind.apply_task())
    elif cmd == "build":
        explain = rest.strip() == "--explain"
        spec = _read_block("paste the spec")
        if not spec.strip():
            print("no spec given")
            return conv
        res = mind.builder(workers=4).build(spec, scope=mind.scope())
        mind.last_build = res
        print(res.explain() if explain else res.render())
        if res.code:
            print(f"\n```python\n{res.code}```\n(/accept to keep it as an exemplar kgirl learns from)")
        mind.store.append(conv, "tool", f"[build] {res.render()}\n{res.code}")
    elif cmd == "explain":
        if mind.last_build is None:
            print("no build in this session")
        else:
            print(mind.last_build.explain())
    elif cmd == "accept":
        if mind.last_build is None or not mind.last_build.code:
            print("no build in this session")
        else:
            fid = mind.builder().learn(mind.last_build, accepted=True, scope=mind.scope(), note=rest)
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
    from ..atlas.secrets import secret_reason
    from .mind.build_spec import atlas_context

    spec = sys.stdin.read() if args.spec == "-" else Path(args.spec).read_text(encoding="utf-8")
    scope = getattr(args, "repo", None) or mind.scope()
    builder = mind.builder(n_impl=args.n, n_suites=args.k, repair_rounds=getattr(args, "repair_rounds", 2),
                           allow_io=args.allow_io, use_exemplars=not getattr(args, "no_exemplars", False))
    ctx_paths = getattr(args, "context", None) or []
    repo = getattr(args, "repo", None)
    context_text = ""
    extra_files: dict[str, str] = {}
    if ctx_paths:
        if repo:
            context_text, extra_files = atlas_context(mind.a.atlas, repo, ctx_paths)
        else:
            blocks: list[str] = []
            for p in ctx_paths:
                fp = Path(p).expanduser()
                raw = fp.read_text(encoding="utf-8")
                why = secret_reason(str(fp), raw)
                if why:
                    raise ValueError(f"{fp} looks like it holds a secret ({why})")
                blocks.append(f"# ---- {fp.name} ----\n```python\n{raw[:6000].rstrip()}\n```")
                if fp.suffix == ".py" and fp.name != "solution.py":
                    extra_files[fp.name] = raw
            if blocks:
                context_text = "EXISTING REPO CONTEXT:\n" + "\n\n".join(blocks)
    res = builder.build(spec, scope=scope, context=context_text, extra_files=extra_files)
    print(res.explain() if getattr(args, "explain", False) else res.render(), file=sys.stderr)
    if not res.code:
        return 1
    if args.out:
        written = res.write(args.out)
        print("wrote " + ", ".join(written), file=sys.stderr)
    else:
        print(res.code)
    fid = builder.learn(res, accepted=args.accept, scope=scope)
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
    elif args.convo_cmd == "export":
        text = s.transcript(args.id)
        if not text:
            print(f"no conversation #{args.id}", file=sys.stderr)
            return 1
        if args.out:
            Path(args.out).write_text(text, encoding="utf-8")
            print(f"wrote {args.out}", file=sys.stderr)
        else:
            print(text)
    elif args.convo_cmd == "delete":
        print(mind.forget(args.id))
    return 0
