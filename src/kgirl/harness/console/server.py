"""Localhost HTTP surface for the console: JSON API + one SSE stream + the static app.

Stdlib only. Binds 127.0.0.1 and checks the Host header so a browser page from elsewhere cannot drive
this console (DNS rebinding). Long operations become jobs; the browser watches them over /api/events.
"""

from __future__ import annotations

import dataclasses
import difflib
import json
import mimetypes
import os
import queue
import re
import shlex
import sqlite3
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from ..atlas.secrets import secret_reason
from .core import Console

STATIC = Path(__file__).parent / "static"
SLOW_MCP = {"kgirl_build", "jev_swarm_task", "life_experiment", "skills_forge"}


def _plain(obj):
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if isinstance(obj, sqlite3.Row):
        return dict(obj)
    if isinstance(obj, (list, tuple)):
        return [_plain(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    return obj


def _frag(f) -> dict:
    return {**_plain(f), "utility": round(f.utility, 3), "bound": round(f.bound(-1), 3)}


def _swarm(res, root: str) -> dict:
    out = _plain(res)
    out["root"] = root
    out["files"] = []
    if res.winner is not None:
        for path, (orig, new) in sorted(res.winner.edited.items()):
            diff = "".join(difflib.unified_diff(orig.splitlines(keepends=True), new.splitlines(keepends=True),
                                                fromfile=f"a/{path}", tofile=f"b/{path}"))
            out["files"].append({"path": path, "diff": diff})
    return out


def _build(res) -> dict:
    return {**_plain(res), "confidence": round(res.confidence, 3)}


class Handler(BaseHTTPRequestHandler):
    server_version = "kgirl-console"
    protocol_version = "HTTP/1.1"
    console: Console = None                      # set by serve()

    def log_message(self, fmt, *args):           # quiet: the UI is the log
        pass

    # ---------------------------------------------------------------- plumbing

    def _json(self, obj, status: int = 200) -> None:
        body = json.dumps(obj, default=str, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _err(self, status: int, message: str) -> None:
        self._json({"error": message}, status)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        try:
            obj = json.loads(self.rfile.read(n))
            return obj if isinstance(obj, dict) else {}
        except ValueError:
            return {}

    def _host_ok(self) -> bool:
        host = (self.headers.get("Host") or "").split(":")[0]
        return host in ("127.0.0.1", "localhost", "::1", "[::1]", "")

    def _static(self, rel: str) -> None:
        path = (STATIC / (rel or "index.html")).resolve()
        if not path.is_relative_to(STATIC) or not path.is_file():
            return self._err(404, "not found")
        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _sse(self) -> None:
        c = self.console
        q = c.hub.subscribe()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        try:
            hello = {"type": "hello", "approvals": c.approvals.pending(), "jobs": c.jobs.list()}
            self.wfile.write(f"data: {json.dumps(hello, default=str)}\n\n".encode())
            self.wfile.flush()
            while True:
                try:
                    event = q.get(timeout=15)
                except queue.Empty:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                    continue
                self.wfile.write(f"data: {json.dumps(event, default=str, ensure_ascii=False)}\n\n".encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            c.hub.unsubscribe(q)

    # ---------------------------------------------------------------- routing

    def do_GET(self):
        if not self._host_ok():
            return self._err(403, "localhost only")
        url = urlparse(self.path)
        path, query = url.path, {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            if path == "/api/events":
                return self._sse()
            if path.startswith("/api/"):
                return self._json(self._get(path, query))
            if path == "/" or path.startswith("/static/"):
                return self._static(path[len("/static/"):] if path.startswith("/static/") else "")
            return self._err(404, "not found")
        except (KeyError, ValueError, TypeError) as exc:
            return self._err(400, str(exc))
        except Exception as exc:  # noqa: BLE001 - surface harness failures as JSON, not dead air
            return self._err(500, f"{type(exc).__name__}: {exc}")

    def do_POST(self):
        if not self._host_ok():
            return self._err(403, "localhost only")
        path, body = urlparse(self.path).path, self._body()
        try:
            return self._json(self._post(path, body))
        except (KeyError, ValueError, TypeError) as exc:
            return self._err(400, str(exc))
        except PermissionError as exc:
            return self._err(403, str(exc))
        except Exception as exc:  # noqa: BLE001
            return self._err(500, f"{type(exc).__name__}: {exc}")

    def do_DELETE(self):
        if not self._host_ok():
            return self._err(403, "localhost only")
        path = urlparse(self.path).path
        m = re.fullmatch(r"/api/conversations/(\d+)", path)
        if not m:
            return self._err(404, "not found")
        self.console.store.delete(int(m.group(1)))
        return self._json({"ok": True})

    # ---------------------------------------------------------------- GET API

    def _get(self, path: str, q: dict):
        c = self.console
        if path == "/api/state":
            return self._state()
        if path == "/api/jobs":
            return c.jobs.list()
        m = re.fullmatch(r"/api/jobs/([0-9a-f]+)", path)
        if m:
            job = c.jobs.get(m.group(1))
            if job is None:
                raise ValueError("no such job")
            return job.full()
        if path == "/api/conversations":
            return [_plain(x) for x in c.store.list(limit=50)]
        m = re.fullmatch(r"/api/conversations/(\d+)", path)
        if m:
            cid = int(m.group(1))
            conv = c.store.get(cid)
            if conv is None:
                raise ValueError("no such conversation")
            return {"conversation": _plain(conv), "turns": [_plain(t) for t in c.store.turns(cid)]}
        if path == "/api/tools":
            return [{"name": t.name, "usage": t.usage, "help": t.help} for t in c.ops().tools().values()]
        if path == "/api/approvals":
            return c.approvals.pending()
        if path == "/api/workspace":
            ws = c.workspace
            if ws is None:
                return {"attached": False}
            return {"attached": True, "root": str(ws.root), "auto_edit": ws.auto_edit,
                    "allowed": [shlex.join(x) for x in ws.allowed_commands()], "diff": ws.diff()}
        if path == "/api/soup":
            soup = c.a.soup
            limit = min(500, int(q.get("limit", 100)))
            if q.get("q"):
                hits = soup.similar(q["q"], kind=q.get("kind") or None, limit=limit)
                return [{**_frag(f), "score": round(s, 3)} for f, s in hits]
            return [_frag(f) for f in soup.all(status=q.get("status") or None,
                                               kind=q.get("kind") or None)[:limit]]
        if path == "/api/soup/ledger":
            return [dict(r) for r in c.a.soup.ledger(since=float(q.get("since", 0)),
                                                     limit=int(q.get("limit", 200)))]
        if path == "/api/repos":
            return [dict(r) for r in c.a.atlas.repos()]
        if path == "/api/search":
            return self._search(q)
        if path == "/api/source":
            return self._source(q)
        if path == "/api/skills":
            return [_frag(f) for f in c.a.soup.all(kind="skill")]
        if path == "/api/vitals":
            met = c.a.metabolism
            if met is None:
                return {"enabled": False}
            from ..metabolism import render as render_vitals
            v = met.vitals()
            return {"enabled": True, "vitals": _plain(v), "render": render_vitals(v)}
        if path == "/api/plan":
            met = c.a.metabolism
            if met is None:
                return {"enabled": False}
            return _plain(met.plan(int(q["max_n"]) if q.get("max_n") else None))
        if path == "/api/matrix":
            return self._matrix()
        if path == "/api/trace":
            envs = c.a.bus.trace(correlation_id=q.get("correlation_id") or None, topic=q.get("topic", "*"))
            return [_plain(e) for e in envs]
        if path == "/api/mcp/tools":
            reply = c.mcp().handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
            return reply["result"]["tools"]
        raise ValueError(f"no such endpoint: {path}")

    def _state(self) -> dict:
        c, a = self.console, self.console.a
        route = {}
        for role in ("scout", "smith", "jev", "chat", "build"):
            try:
                b = a.router.pick(role)
                route[role] = getattr(b, "model", None) or (type(b).__name__ if b else None)
            except Exception:  # noqa: BLE001 - a role without a backend is a state, not an error
                route[role] = None
        matrix = None
        mp = os.environ.get("KGIRL_MATRIX")
        if mp and Path(mp).is_file():
            try:
                from ..atlas.vectors import Matrix
                m = Matrix(mp)
                matrix = {"path": mp, "count": m.count, "dim": m.dim, "model": m.model, "scale": m.scale}
            except Exception as exc:  # noqa: BLE001
                matrix = {"path": mp, "error": str(exc)}
        return {"home": str(a.home), "repos": [dict(r) for r in a.atlas.repos()], "soup": a.soup.stats(),
                "route": route, "matrix": matrix, "chunks_db": os.environ.get("KGIRL_CHUNKS_DB"),
                "conversations": [_plain(x) for x in c.store.list(limit=50)],
                "tools": [t.name for t in c.ops().tools().values()],
                "workspace": {"attached": c.workspace is not None,
                              "root": str(c.workspace.root) if c.workspace else None},
                "life": a.metabolism is not None, "retriever": c.retriever is not None,
                "jobs": c.jobs.list(), "approvals": c.approvals.pending()}

    def _search(self, q: dict) -> dict:
        query = q.get("q", "").strip()
        if not query:
            raise ValueError("search needs q")
        c = self.console
        limit = min(30, int(q.get("limit", 10)))
        with c.search_lock:
            symbols = [_plain(h) for h in c.a.atlas.search(query, limit=limit)]
            chunks = ([{**_plain(h), "cite": h.cite()} for h in c.retriever.search(query, limit=limit)]
                      if c.retriever is not None else [])
        return {"symbols": symbols, "chunks": chunks}

    def _source(self, q: dict) -> dict:
        repo, path = q["repo"], q["path"]
        start = int(q.get("start", 1))
        end = int(q["end"]) if q.get("end") else None
        text = self.console.a.atlas.read_source(repo, path, start, end)
        why = secret_reason(path, text)
        if why:
            raise PermissionError(f"{repo}:{path} looks like it holds a secret ({why})")
        return {"repo": repo, "path": path, "text": text}

    def _matrix(self) -> dict:
        """A sampled spatial projection of the KGVM store: random 2-D projection (ChaosRNG only),
        points colored by repo. The full search stays coarse sign-bit Hamming -> fine int8 dot."""
        mp = os.environ.get("KGIRL_MATRIX")
        if not mp or not Path(mp).is_file():
            return {"available": False}
        import numpy as np
        from ..atlas.vectors import Matrix
        from ...evolve.chaos import ChaosRNG
        m = Matrix(mp)
        n = min(m.count, 700)
        rng = ChaosRNG(seed=0.4123)
        idx = rng.integers(m.count, size=n)
        proj = rng.normal(size=(m.dim, 2))
        pts = m.vals[idx].astype(np.float32) @ proj
        pts = pts / np.maximum(np.abs(pts).max(axis=0), 1e-9)
        repos: dict = {}
        db = os.environ.get("KGIRL_CHUNKS_DB")
        if db and Path(db).is_file():
            con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            try:
                marks = ",".join("?" * len(idx))
                repos = dict(con.execute(f"SELECT id, repo FROM chunks WHERE id IN ({marks})",
                                         [int(i) for i in m.ids[idx]]).fetchall())
            finally:
                con.close()
        points = [{"x": round(float(pts[i, 0]), 4), "y": round(float(pts[i, 1]), 4),
                   "repo": repos.get(int(m.ids[idx[i]]), "?"), "id": int(m.ids[idx[i]])} for i in range(n)]
        return {"available": True, "count": m.count, "dim": m.dim, "model": m.model, "scale": m.scale,
                "sample": n, "points": points}

    # ---------------------------------------------------------------- POST API

    def _post(self, path: str, b: dict):
        c = self.console
        if path == "/api/conversations":
            return {"id": c.store.new(str(b.get("title", "")))}
        if path == "/api/chat":
            cid = int(b["conv_id"])
            if c.store.get(cid) is None:
                raise ValueError("no such conversation")
            text = str(b.get("text", "")).strip()
            if not text:
                raise ValueError("chat needs text")
            return {"job": self._chat_job(cid, text).id}
        if path == "/api/feedback":
            cid, verdict = int(b["conv_id"]), str(b["verdict"])
            if verdict not in ("positive", "negative"):
                raise ValueError("verdict must be positive or negative")
            return {"text": c.mind(cid).feedback(cid, verdict)}
        if path == "/api/tool":
            return self._tool(b)
        if path == "/api/apply":
            mind = c.mind(int(b["conv_id"])) if b.get("conv_id") else c.ops()
            return {"text": mind.apply_task()}
        if path == "/api/approve":
            ok = c.approvals.resolve(str(b["id"]), bool(b.get("yes")))
            if not ok:
                raise ValueError("no such pending approval")
            return {"ok": True}
        if path == "/api/workspace/undo":
            return {"text": c.workspace.undo() if c.workspace else "no workspace attached"}
        if path == "/api/workspace/auto":
            if c.workspace is None:
                raise ValueError("no workspace attached")
            c.workspace.auto_edit = bool(b.get("on"))
            return {"auto_edit": c.workspace.auto_edit}
        if path == "/api/workspace/allow":
            if c.workspace is None:
                raise ValueError("no workspace attached")
            c.workspace.allow(str(b["command"]))
            return {"allowed": [shlex.join(x) for x in c.workspace.allowed_commands()]}
        if path == "/api/workspace/verify":
            cmd = shlex.split(str(b["command"])) if str(b.get("command", "")).strip() else None
            with c._lock:
                minds = list(c._minds.values()) + ([c._ops] if c._ops else [])
            for m in minds:
                m.verify_cmd = cmd
            c.ops().verify_cmd = cmd
            return {"verify": shlex.join(cmd) if cmd else None}
        return self._post_more(path, b)

    def _chat_job(self, cid: int, text: str):
        c = self.console
        mind, lock = c.mind(cid), c.chat_lock(cid)

        def run(emit):
            with lock:
                reply = mind.chat(cid, text, on_text=lambda d: emit("delta", text=d),
                                  on_event=lambda s: emit("status", text=s))
            return {"text": reply.text, "conv_id": reply.conv_id, "recalled": reply.recalled,
                    "tools": _plain(reply.tools), "stored": reply.stored, "verdict": reply.verdict}

        return c.jobs.start("chat", text[:80], run, heavy=True)

    def _tool(self, b: dict):
        c = self.console
        name, args = str(b["name"]), dict(b.get("args") or {})
        mind = c.mind(int(b["conv_id"])) if b.get("conv_id") else c.ops()
        if name == "task":
            return {"job": self._task_job(mind, args).id}
        if name == "build":
            return {"job": self._build_job(str(args.get("spec", "")), int(args.get("candidates") or 6)).id}
        if name == "write":
            return {"job": self._write_job(mind, args).id}
        if name not in mind.tools():
            raise ValueError(f"unknown tool {name!r}")
        return {"text": mind.call_tool(name, args)}

    def _post_more(self, path: str, b: dict):
        c = self.console
        if path == "/api/soup/remember":
            text = str(b.get("text", "")).strip()
            if not text:
                raise ValueError("remember needs text")
            kind = b.get("kind", "fact")
            if kind not in ("fact", "preference", "abstraction", "note"):
                raise ValueError("kind must be fact, preference, abstraction or note")
            return {"id": c.a.remember(text, kind=kind, tags="console")}
        if path == "/api/soup/credit":
            ids = [int(i) for i in b["ids"]]
            c.a.soup.credit(ids, success=bool(b.get("success", True)))
            return {"ok": True}
        if path == "/api/soup/status":
            status = str(b["status"])
            if status not in ("active", "staged", "retired"):
                raise ValueError("status must be active, staged or retired")
            c.a.soup.set_status(int(b["id"]), status, str(b.get("reason", "console")))
            return {"ok": True}
        if path == "/api/task":
            return {"job": self._task_job(c.ops(), b).id}
        if path == "/api/build":
            spec = str(b.get("spec", "")).strip()
            if not spec:
                raise ValueError("build needs a spec")
            return {"job": self._build_job(spec, int(b.get("candidates") or 6),
                                           bool(b.get("allow_io"))).id}
        if path == "/api/build/accept":
            res = c.ops().last_build
            if res is None:
                raise ValueError("no build waiting")
            fid = c.ops().builder().learn(res, accepted=True, note=str(b.get("note", "")))
            return {"exemplar": fid}
        if path == "/api/write":
            return {"job": self._write_job(c.ops(), b).id}
        if path == "/api/evolve":
            def run(emit):
                from ..mind.evolve import evolve
                emit("status", text="reflect, curate, profile, skills")
                rep = evolve(c.ops()._llm_text, c.a.soup, c.a.curator, c.store)
                return {**_plain(rep), "render": rep.render(c.a.soup)}
            return {"job": c.jobs.start("evolve", "consolidate memory", run, heavy=True).id}
        if path == "/api/index":
            spec = str(b.get("spec", "")).strip()
            if not spec:
                raise ValueError("index needs a repo name or URL")
            def run(emit):
                emit("status", text=f"indexing {spec}")
                stats = c.a.index([spec], refresh=bool(b.get("refresh")))
                return {"stats": _plain(stats), "failures": c.a.index_failures}
            return {"job": c.jobs.start("index", spec, run, heavy=True).id}
        if path == "/api/skills/forge":
            from ..skills import forge
            return {"skills": _plain(forge(c.a.soup))}
        if path == "/api/skills/export":
            from ..skills import export, forge
            directory = str(b.get("dir", "")).strip()
            if not directory:
                raise ValueError("export needs dir")
            skills = forge(c.a.soup, store=False)
            return {"paths": [str(p) for p in export(skills, directory)]}
        if path == "/api/matrix/search":
            return {"job": self._matrix_search_job(str(b.get("query", "")), int(b.get("k") or 12)).id}
        if path == "/api/mcp/call":
            return self._mcp_call(b)
        raise ValueError(f"no such endpoint: {path}")

    def _task_job(self, mind, args: dict):
        c = self.console
        goal = str(args.get("goal", "")).strip()
        if not goal:
            raise ValueError("task needs a goal")

        def run(emit):
            mind.last_task = None
            emit("status", text="sandboxed specialists working the goal")
            text = mind.call_tool("task", args, on_event=lambda s: emit("status", text=s))
            if mind.last_task is not None:
                res, root = mind.last_task
                return {**_swarm(res, str(root)), "text": text}
            return {"text": text, "winner": None, "files": []}

        return c.jobs.start("task", goal[:80], run, heavy=True)

    def _build_job(self, spec: str, candidates: int = 6, allow_io: bool = False):
        c = self.console
        if not spec.strip():
            raise ValueError("build needs a spec")

        def run(emit):
            emit("status", text="tests first, several candidates, agreement selection")
            res = c.ops().builder(n_impl=max(2, min(8, candidates)), allow_io=allow_io).build(spec)
            c.ops().last_build = res
            return _build(res)

        return c.jobs.start("build", spec[:80], run, heavy=True)

    def _write_job(self, mind, args: dict):
        c = self.console
        brief = str(args.get("brief", "")).strip()
        if not brief:
            raise ValueError("write needs a brief")

        def run(emit):
            text = mind.call_tool("write", args, on_event=lambda s: emit("status", text=s))
            return {"markdown": text}

        return c.jobs.start("write", brief[:80], run, heavy=True)

    def _matrix_search_job(self, query: str, k: int):
        c = self.console
        if not query.strip():
            raise ValueError("matrix search needs a query")
        mp = os.environ.get("KGIRL_MATRIX")
        if not mp or not Path(mp).is_file():
            raise ValueError("no KGIRL_MATRIX configured")

        def run(emit):
            from ..atlas.vectors import Matrix, ollama_embedder
            emit("status", text="embedding the query (CPU)")
            m = Matrix(mp)
            vec = ollama_embedder(cpu=True)([query])[0]
            bits, vals = m.encode_query(vec)
            emit("status", text=f"coarse sign-bit pass over {m.count} vectors, fine int8 pass on 400")
            hits = m.search(bits, vals, k=max(1, min(50, k)), shortlist=400)
            rows: dict = {}
            db = os.environ.get("KGIRL_CHUNKS_DB")
            if db and Path(db).is_file() and hits:
                con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
                try:
                    marks = ",".join("?" * len(hits))
                    rows = {r["id"]: dict(r) for r in con.execute(
                        f"SELECT id, repo, path, symbol, start_line, text FROM chunks WHERE id IN ({marks})",
                        [i for i, _ in hits]).fetchall()}
                finally:
                    con.close()
            out = []
            for i, s in hits:
                row = rows.get(i, {})
                row["text"] = (row.get("text") or "")[:600]
                out.append({"id": i, "score": s, **row})
            return {"query": query, "count": m.count, "shortlist": 400, "hits": out}

        return c.jobs.start("matrix", query[:80], run, heavy=True)

    def _mcp_call(self, b: dict):
        c = self.console
        name = str(b["name"])
        msg = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
               "params": {"name": name, "arguments": dict(b.get("arguments") or {})}}
        if name in SLOW_MCP:
            def run(emit):
                emit("status", text=f"MCP tool {name}")
                reply = c.mcp().handle(msg)
                return reply.get("result", reply)
            return {"job": c.jobs.start("mcp", name, run, heavy=True).id}
        reply = c.mcp().handle(msg)
        if reply is None or "result" not in reply:
            raise ValueError(str(reply))
        return reply["result"]


def serve(a, port: int = 8765, workspace_root: Path | None = None, auto_edit: bool = False) -> int:
    console = Console(a, workspace_root=workspace_root, auto_edit=auto_edit)
    Handler.console = console
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    httpd.daemon_threads = True
    print(f"kgirl console: http://127.0.0.1:{httpd.server_address[1]}/  (Ctrl-C to stop)", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        console.close()
    return 0