"""Localhost HTTP surface for the console: JSON API + one SSE stream + the static app.

Stdlib only. Binds 127.0.0.1 and checks the Host header so a browser page from elsewhere cannot drive
this console (DNS rebinding). Long operations become jobs; the browser watches them over /api/events.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import mimetypes
import os
import queue
import re
import shlex
import socket
import sqlite3
import subprocess
import threading
import time
import urllib.request  # noqa: F401  (tests patch server.urllib.request; the GPU probe lives in core)
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from ..atlas.secrets import secret_reason
from .core import Console, gpu_status
from .core import build_dict as _build, frag_dict as _frag, plain as _plain, swarm_dict as _swarm  # noqa: F401

STATIC = Path(__file__).parent / "static"
SLOW_MCP = {"kgirl_build", "jev_swarm_task", "life_experiment", "skills_forge"}


def _critic_host_allowed(host: str) -> bool:
    """A critic host named from the browser may only be loopback (the adb forward to a phone) or a host the
    operator listed in KGIRL_CRITIC_HOSTS. Otherwise a forged request could ship prompts and code off-box."""
    name = (urlparse(host).hostname or "").lower()
    if name in ("127.0.0.1", "localhost", "::1"):
        return True
    allow = os.environ.get("KGIRL_CRITIC_HOSTS", "")
    names = {(urlparse(h.strip()).hostname or h.strip()).lower() for h in allow.split(",") if h.strip()}
    return name in names


def _porter_db() -> Path:
    return Path(os.environ.get("AURIC_PORTER_DB") or Path.home() / ".local" / "share" / "auric" / "porter.sqlite")


def _rescue_script() -> Path:
    return Path(os.environ.get("KGIRL_RESCUE_SCRIPT") or "~/work/kgirl-rescue/scripts/rescue.py").expanduser()


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _session_alive(row, now: datetime, ttl_s: float, host: str) -> bool:
    """Mirrors auric.porter.Porter.alive: not ended, seen within the TTL, and (on this host) a live pid."""
    if row["ended"]:
        return False
    try:
        seen = datetime.fromisoformat(row["last_seen"])
    except (ValueError, TypeError):
        return False
    if (now - seen).total_seconds() > ttl_s:
        return False
    if row["pid"] and row["host"] == host:
        return _pid_alive(row["pid"])
    return True


LANDING_COMMAND = ("python3 scripts/smoke/land_lanes.py --stack ~/work/kgirl-stack "
                   "--lane ~/work/kgirl-eng --lane ~/work/kgirl-helix-ui --lane ~/work/kgirl-builder")


def _landing() -> dict:
    """The landing plan's worktrees and their uncommitted path counts. Display only: the sole git commands
    are `status --porcelain` and `rev-parse --abbrev-ref HEAD`, both reads; the console never runs git writes."""
    stack = Path(os.environ.get("KGIRL_LANDING_STACK") or "~/work/kgirl-stack").expanduser()
    lanes = [Path(p.strip()).expanduser() for p in
             (os.environ.get("KGIRL_LANDING_LANES")
              or "~/work/kgirl-eng,~/work/kgirl-helix-ui,~/work/kgirl-builder").split(",") if p.strip()]
    worktrees = []
    for path in [stack, *lanes]:
        row = {"name": path.name, "path": str(path), "role": "stack" if path == stack else "lane"}
        try:
            st = subprocess.run(["git", "-C", str(path), "status", "--porcelain"],
                                capture_output=True, text=True, timeout=10)
            if st.returncode != 0:
                row["error"] = (st.stderr or "not a git worktree").strip().splitlines()[0]
            else:
                br = subprocess.run(["git", "-C", str(path), "rev-parse", "--abbrev-ref", "HEAD"],
                                    capture_output=True, text=True, timeout=10)
                row["branch"] = br.stdout.strip()
                row["uncommitted"] = len([ln for ln in st.stdout.splitlines() if ln.strip()])
        except (OSError, subprocess.SubprocessError) as exc:
            row["error"] = str(exc)
        worktrees.append(row)
    return {"command": LANDING_COMMAND, "worktrees": worktrees}


def _code_root() -> Path:
    return Path(os.environ.get("KGIRL_CODE") or "~/work/kgirl-stack").expanduser()


def _code_fingerprint(root: Path) -> str | None:
    """HEAD plus a short hash of the uncommitted diff and untracked src files — the same shape the
    helix-eval runner records as code_rev. None when the root is not a readable git worktree."""
    def git(*args):
        r = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=10)
        return r.stdout if r.returncode == 0 else None
    try:
        head = git("rev-parse", "--short=10", "HEAD")
        if head is None:
            return None
        diff, untracked = git("diff", "HEAD"), git("ls-files", "--others", "--exclude-standard", "src")
        if diff is None or untracked is None:
            return None
        dirty = diff + untracked
        return head.strip() + (f"+dirty.{hashlib.sha256(dirty.encode()).hexdigest()[:8]}" if dirty.strip() else "")
    except (OSError, subprocess.SubprocessError):
        return None


class CodeWatch:
    """The code fingerprint taken at startup, re-read at most once a minute. When it drifts, the running
    server no longer matches the code on disk and the header hints a restart. Git reads only."""

    INTERVAL_S = 60.0

    def __init__(self, root: Path | None = None):
        self.root = root if root is not None else _code_root()
        self.startup = _code_fingerprint(self.root)
        self._lock = threading.Lock()
        self._checked = 0.0
        self._current = self.startup
        self._changed = False

    def status(self) -> dict:
        with self._lock:
            now = time.monotonic()
            if now - self._checked >= self.INTERVAL_S:
                self._checked = now
                self._current = _code_fingerprint(self.root)
                self._changed = bool(self.startup and self._current and self._current != self.startup)
            return {"available": self.startup is not None, "root": str(self.root),
                    "startup": self.startup, "current": self._current, "changed": self._changed}


POCKET_STATE = Path.home() / ".local" / "state" / "kgirl-pocket"
POCKET_NODE = ("127.0.0.1", 11435, 11434)      # desktop adb forward -> phone Ollama, both loopback


def _pocket_listening(host: str, port: int) -> bool:
    """A local TCP check of the adb-forwarded port: is the forward up? Connects and closes at once — no data
    is sent and no command reaches the phone."""
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


class Handler(BaseHTTPRequestHandler):
    server_version = "kgirl-console"
    protocol_version = "HTTP/1.1"
    console: Console = None                      # set by serve()
    code_watch: CodeWatch | None = None          # set by serve(); the startup code fingerprint

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

    def _origin_ok(self) -> bool:
        """A cross-site page sends its own Origin; only this console's own origin may drive it. An absent
        Origin (same-origin GET, curl) is allowed here and still has to pass the token check."""
        origin = self.headers.get("Origin")
        if not origin:
            return True
        port = self.server.server_address[1]
        return origin in (f"http://127.0.0.1:{port}", f"http://localhost:{port}", f"http://[::1]:{port}")

    def _token_ok(self, query: dict | None = None) -> bool:
        """The per-launch secret, from the X-Kgirl-Token header or (for EventSource, which cannot set
        headers) the ?token= query parameter. Constant-time compare."""
        supplied = self.headers.get("X-Kgirl-Token") or (query or {}).get("token", "")
        return hmac.compare_digest(supplied.encode(), self.console.token.encode())

    def _guarded(self, query: dict | None = None) -> bool:
        return self._host_ok() and self._origin_ok() and self._token_ok(query)

    def _static(self, rel: str) -> None:
        path = (STATIC / (rel or "index.html")).resolve()
        if not path.is_relative_to(STATIC) or not path.is_file():
            return self._err(404, "not found")
        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        body = path.read_bytes()
        if path.name == "index.html":                  # hand the page its per-launch secret, same-origin only
            body = body.replace(b"__KGIRL_TOKEN__", self.console.token.encode())
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
            if path.startswith("/api/"):
                if not self._guarded(query):
                    return self._err(403, "localhost, same origin and the launch token only")
                if path == "/api/events":
                    return self._sse()
                return self._json(self._get(path, query))
            if path == "/" or path.startswith("/static/"):
                return self._static(path[len("/static/"):] if path.startswith("/static/") else "")
            return self._err(404, "not found")
        except (KeyError, ValueError, TypeError) as exc:
            return self._err(400, str(exc))
        except Exception as exc:  # noqa: BLE001 - surface harness failures as JSON, not dead air
            return self._err(500, f"{type(exc).__name__}: {exc}")

    def do_POST(self):
        if not self._guarded():
            return self._err(403, "localhost, same origin and the launch token only")
        # A cross-site no-cors POST can only be a "simple" Content-Type; requiring JSON forces a CORS
        # preflight this server never answers, so the forged request is never sent.
        ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if ctype != "application/json":
            return self._err(415, "POST needs Content-Type: application/json")
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
        if not self._guarded():
            return self._err(403, "localhost, same origin and the launch token only")
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
        if path == "/api/rules":
            return self._rules()
        if path == "/api/drafts":
            return c.drafts(int(q["conv_id"]) if q.get("conv_id") else None)
        if path == "/api/context":
            return c.context(int(q["conv_id"]))
        if path == "/api/doctor":
            return c.doctor()
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
        if path == "/api/helix/state":
            return self._helix_state(q)
        if path == "/api/helix/history":
            from ..helix.store import StateStore
            with StateStore(c.a.home) as store:
                return store.history(q.get("session", "main"))
        if path == "/api/helix/verify":
            from ..helix.store import StateStore
            with StateStore(c.a.home) as store:
                return store.verify(q.get("session") or None)
        if path == "/api/helix/diff":
            from ..helix.store import HelixError, StateStore
            try:
                with StateStore(c.a.home) as store:
                    return store.diff(q.get("session", "main"), int(q["a"]), int(q["b"]))
            except HelixError as exc:
                raise ValueError(str(exc))
        if path == "/api/holo/status":
            from ..holo import status
            return status(c.a.home)
        if path == "/api/holo/owners":
            from ..holo import HoloStore
            with HoloStore(c.a.home) as store:
                return {"digest": q["digest"], "owners": store.owners(q["digest"])}
        if path == "/api/holo/overlaps":
            from ..holo import HoloStore
            limit = max(1, min(500, int(q.get("limit", 100))))
            with HoloStore(c.a.home) as store:
                ov = store.overlaps()
            items = [{"digest": d, "owners": o} for d, o in sorted(ov.items())][:limit]
            return {"count": len(ov), "shown": len(items), "overlaps": items}
        if path == "/api/holo/verify":
            from ..holo import HoloStore, status
            if status(c.a.home)["built"] is None:            # a read must not create an empty store
                return {}
            with HoloStore(c.a.home) as store:
                return store.verify()
        if path == "/api/lanes":
            return self._lanes(q)
        if path == "/api/pocket":
            return self._pocket()
        if path == "/api/rescue":
            return self._rescue()
        if path == "/api/code":
            watch = self.code_watch
            return watch.status() if watch is not None else {"available": False, "changed": False}
        if path == "/api/gpu":
            return self._gpu()
        raise ValueError(f"no such endpoint: {path}")

    def _gpu(self) -> dict:
        return gpu_status()

    def _pocket(self) -> dict:
        """Read-only phone/node status from kgirl-pocket's LOCAL state files and a local port check. Never
        invokes adb and never sends anything to the phone."""
        host, port, phone_port = POCKET_NODE

        def read(name):
            try:
                return json.loads((POCKET_STATE / name).read_text())
            except (OSError, ValueError):
                return None

        return {"state_dir": str(POCKET_STATE),
                "node": {"host": host, "port": port, "phone_port": phone_port,
                         "listening": _pocket_listening(host, port)},
                "backup": read("backup.json"), "drill": read("drill.json"), "bench": read("bench.json")}

    def _rules(self) -> dict:
        return self.console.rules()

    def _rescue(self) -> dict:
        """Read-only rescue scan: feature-detect the rescue lane's CLI and run ONLY its `scan --json`
        mode, with a timeout. The console never invokes record or commit — it displays, it does not rescue."""
        script = _rescue_script()
        out = {"available": script.is_file(), "script": str(script)}
        if not out["available"]:
            return out
        try:
            r = subprocess.run(["python3", str(script), "scan", "--json"],
                               capture_output=True, text=True, timeout=30)
        except subprocess.TimeoutExpired:
            return {**out, "error": "rescue scan timed out after 30 s"}
        except OSError as exc:
            return {**out, "error": str(exc)}
        if r.returncode != 0:
            lines = (r.stderr or r.stdout).strip().splitlines()
            return {**out, "error": lines[-1] if lines else "rescue scan failed"}
        try:
            rows = json.loads(r.stdout)
        except ValueError:
            return {**out, "error": "rescue scan did not return JSON"}
        if not isinstance(rows, list):
            return {**out, "error": "rescue scan returned an unexpected shape"}
        return {**out, "sessions": rows}

    def _lanes(self, q: dict) -> dict:
        """A read-only window onto the shared porter ledger: every active agent session, its claims, and the
        recent events. Reads porter.sqlite read-only (stdlib sqlite, no auric import); never writes."""
        path = _porter_db()
        if not path.is_file():
            return {"available": False, "db": str(path), "projects": [], "sessions": [], "events": [],
                    "landing": _landing()}
        now = datetime.now(timezone.utc)
        ttl_s = float(os.environ.get("AURIC_PORTER_TTL_HOURS", 12)) * 3600
        host = socket.gethostname()
        project = q.get("project") or None
        limit = max(1, min(200, int(q.get("limit", 60))))
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        try:
            where = " WHERE ended IS NULL" + (" AND project=?" if project else "")
            args = (project,) if project else ()
            sessions = []
            for r in con.execute("SELECT * FROM sessions" + where + " ORDER BY last_seen DESC", args):
                try:
                    age = round((now - datetime.fromisoformat(r["last_seen"])).total_seconds())
                except (ValueError, TypeError):
                    age = None
                claims = [dict(c) for c in con.execute(
                    "SELECT path, intent FROM claims WHERE session=? ORDER BY path", (r["id"],))]
                sessions.append({"id": r["id"], "agent": r["agent"], "project": r["project"], "task": r["task"],
                                 "started": r["started"], "last_seen": r["last_seen"], "age_s": age,
                                 "ended": r["ended"], "summary": r["summary"], "pid": r["pid"],
                                 "alive": _session_alive(r, now, ttl_s, host), "claims": claims})
            projects = [r[0] for r in con.execute("SELECT DISTINCT project FROM sessions ORDER BY project")]
            ewhere = (" WHERE project=?" if project else "")
            eargs = (project, limit) if project else (limit,)
            events = [dict(e) for e in con.execute(
                "SELECT id, created, actor, kind, recipient, message, paths FROM events" + ewhere +
                " ORDER BY id DESC LIMIT ?", eargs)]
            return {"available": True, "db": str(path), "projects": projects, "sessions": sessions,
                    "events": events, "landing": _landing()}
        finally:
            con.close()

    def _helix_state(self, q: dict) -> dict:
        from ..helix.store import StateStore
        session = q.get("session", "main")
        with StateStore(self.console.a.home) as store:
            head = store.head(session)
            return {"sessions": store.sessions(), "session": session,
                    "state": head.to_dict() if head else None, "floor": store.floor()}

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
        return self.console.search(query, int(q.get("limit", 10)))

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
            return {"allowed": c.allow(str(b["command"]))}
        if path == "/api/workspace/verify":
            return {"verify": c.set_verify(str(b.get("command") or ""))}
        if path == "/api/drafts/resume":
            return {"job": c.resume_draft_job(int(b["id"])).id}
        return self._post_more(path, b)

    def _chat_job(self, cid: int, text: str):
        return self.console.chat_job(cid, text)

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
        if path == "/api/builder/run":
            spec = str(b.get("spec", "")).strip()
            if not spec:
                raise ValueError("builder needs a spec")
            return {"job": self._build_job(spec, int(b.get("candidates") or 6),
                                           bool(b.get("allow_io")), kind="builder").id}
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
        if path == "/api/helix/run":
            return {"job": self._helix_run_job(b).id}
        if path == "/api/helix/ingest":
            from ..helix.store import StateStore
            if "state" not in b:
                raise ValueError("ingest needs a state")
            with StateStore(c.a.home) as store:
                v = store.ingest(str(b.get("session") or "main"), b["state"],
                                 source=str(b.get("source") or "console"))
            return {"accepted": v.ok, "violations": list(v.violations), "clamped": list(v.clamped),
                    "state": v.state.to_dict()}
        if path == "/api/helix/audit":
            from ..helix.evaluate import evaluate
            text = str(b.get("text", ""))
            if not text.strip():
                raise ValueError("audit needs a transcript")
            external = int(b["external"]) if b.get("external") else None
            return evaluate(text, external)
        if path == "/api/holo/build":
            return {"job": self._holo_build_job(b).id}
        if path == "/api/holo/query":
            return {"job": self._holo_query_job(b).id}
        if path == "/api/mcp/call":
            return self._mcp_call(b)
        raise ValueError(f"no such endpoint: {path}")

    def _task_job(self, mind, args: dict):
        return self.console.task_job(mind, args)

    def _build_job(self, spec: str, candidates: int = 6, allow_io: bool = False, kind: str = "build"):
        return self.console.build_job(spec, candidates, allow_io, kind)

    def _write_job(self, mind, args: dict):
        return self.console.write_job(mind, args)

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

    def _helix_run_job(self, b: dict):
        # Validate at the HTTP boundary first: a bad request answers 400 before anything is delegated.
        task = str(b.get("task", "")).strip()
        if not task:
            raise ValueError("helix run needs a task")
        critic_host = str(b.get("critic_host") or "").strip() or None
        if critic_host and not _critic_host_allowed(critic_host):
            raise ValueError("critic_host must be loopback or listed in KGIRL_CRITIC_HOSTS")
        return self.console.helix_job(
            task, budget=int(b.get("budget") or 4000), session=str(b.get("session") or "main"),
            critic_model=str(b.get("critic_model") or "").strip() or None, critic_host=critic_host,
            experts=[str(e) for e in b["expert"]] if b.get("expert") else None, lattice=bool(b.get("lattice")),
            expect=int(b["expect"]) if b.get("expect") else None)

    def _holo_build_job(self, b: dict):
        c = self.console
        kw = {"per_cell": int(b["per_cell"])} if b.get("per_cell") is not None else {}

        def run(emit):
            from ..holo import build
            emit("status", text="deriving the store from the chunk index (offline, no model)")
            return build(c.a.home, bus=c.a.bus, **kw)

        return c.jobs.start("holo-build", "build holo store", run, heavy=True)

    def _holo_query_job(self, b: dict):
        c = self.console
        text = str(b.get("query", "")).strip()
        if not text:
            raise ValueError("holo query needs text")
        k = max(1, min(50, int(b.get("k") or 8)))

        def run(emit):
            from ..holo import HoloStore, query
            emit("status", text="one embedding call (CPU), then KGVM coarse-to-fine with provenance")
            with HoloStore(c.a.home) as store:
                return query(store, text, k, bus=c.a.bus)

        return c.jobs.start("holo-query", text[:80], run, heavy=True)

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
    Handler.code_watch = CodeWatch()
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