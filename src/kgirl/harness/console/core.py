"""Console state: live events, jobs, approvals, and one shared harness.

Design notes that matter for correctness:
- The bus dispatches handlers synchronously on the publisher's thread (swarm workers publish their own
  steps), so the hub handler only enqueues. A slow browser drops events; it never stalls a run.
- Model-driven work (chat, task, build, write, evolve, index) shares one heavy lock: the metabolism keeps
  a single episode slot and the GPU holds one model, so these serialize by design, not by accident.
- Workspace.confirm becomes an HTTP round-trip. An unanswered approval times out to "no", so an
  unattended console never edits or runs anything silently.
"""

from __future__ import annotations

import dataclasses
import difflib
import json
import os
import queue
import re
import secrets
import shlex
import sqlite3
import subprocess
import threading
import time
import urllib.request
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..assistant import Assistant
from ..mind import Mind
from ..mind.conversations import ConversationStore
from ..mind.workspace import Workspace


def plain(obj):
    """A JSON-ready copy of dataclasses, sqlite rows and containers of them."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if isinstance(obj, sqlite3.Row):
        return dict(obj)
    if isinstance(obj, (list, tuple)):
        return [plain(x) for x in obj]
    if isinstance(obj, dict):
        return {k: plain(v) for k, v in obj.items()}
    return obj


def frag_dict(f) -> dict:
    return {**plain(f), "utility": round(f.utility, 3), "bound": round(f.bound(-1), 3)}


def swarm_dict(res, root: str) -> dict:
    out = plain(res)
    out["root"] = root
    out["files"] = []
    if res.winner is not None:
        for path, (orig, new) in sorted(res.winner.edited.items()):
            diff = "".join(difflib.unified_diff(orig.splitlines(keepends=True), new.splitlines(keepends=True),
                                                fromfile=f"a/{path}", tofile=f"b/{path}"))
            out["files"].append({"path": path, "diff": diff})
    return out


def build_dict(res) -> dict:
    out = {**plain(res), "confidence": round(res.confidence, 3)}
    # explain()/summary_dict() arrive with the builder lane; on older results the panel stays hidden.
    try:
        explain = getattr(res, "explain", None)
        out["explain"] = explain() if callable(explain) else ""
    except Exception:
        out["explain"] = ""
    try:
        summary = getattr(res, "summary_dict", None)
        out["summary"] = summary() if callable(summary) else None
    except Exception:
        out["summary"] = None
    return out


def rules_meta(turn) -> dict:
    """The rules-check record of an answer turn: names it revised away and names it still breaks."""
    meta = getattr(turn, "meta", None) or {}
    hits, left = meta.get("rules_hits"), meta.get("rules_left")
    return {"rules_hits": list(hits) if isinstance(hits, list) else [],
            "rules_left": list(left) if isinstance(left, list) else [],
            "rules_revised": bool(meta.get("rules_revised"))}


# "(saved as draft #N; `kgirl write --resume N` continues one)" — the voice lane's writer footer.
DRAFT_FOOTER = re.compile(r"\n*\(saved as draft #(\d+)[^\n]*\)\s*$")


def split_draft_footer(text: str) -> tuple[str, int | None]:
    """The body of a write result and the draft id its footer names (None when there is no footer)."""
    m = DRAFT_FOOTER.search(text or "")
    return (text[:m.start()].rstrip() + "\n", int(m.group(1))) if m else (text, None)


def _dicts(value) -> list[dict]:
    return [x for x in value if isinstance(x, dict)] if isinstance(value, list) else []


def answer_meta(turn) -> dict:
    """What an answer turn's meta tells a reader beyond the text, each part feature-detected (absent → empty):
    the rules check, `budget` (block sizes and what the prompt budget cut) and `related` (earlier
    conversations pulled in, and why)."""
    meta = getattr(turn, "meta", None) or {}
    budget = meta.get("budget")
    return {**rules_meta(turn), "budget": budget if isinstance(budget, dict) else None,
            "related": _dicts(meta.get("related"))}


def budget_line(budget: dict | None) -> str:
    """One line for a chip: "context 3120/4096 · cut memory, earlier" (empty without a budget record)."""
    if not budget:
        return ""
    line = f"context {budget.get('used', 0)}/{budget.get('limit', 0)}"
    dropped = budget.get("dropped") or {}
    if isinstance(dropped, dict) and dropped:
        line += " · cut " + ", ".join(sorted(dropped))
    if budget.get("overflow"):
        line += f" · over by {budget['overflow']}"
    return line


def related_line(item: dict) -> str:
    """Why an earlier conversation was pulled into the prompt."""
    why = f"same topic: {', '.join(map(str, item.get('tags') or []))}" if item.get("via") == "tags" else "matching words"
    return f"conv #{item.get('conv')} ({why})"


def audit_line(event: dict) -> str:
    """One workspace audit record (seq, action, target, outcome) as a line."""
    return " ".join(str(event.get(k, "")) for k in ("action", "target")).strip() + \
        (f" → {event['outcome']}" if event.get("outcome") else "")


def gpu_status() -> dict:
    """Read-only GPU and Ollama status: nvidia-smi, /api/ps (with a best-effort /api/show for the native
    context window), and the active KGIRL_MIND_CTX. Warns when a model is not fully GPU-resident or the
    GPU is at/above 95 C. Nothing here mutates the GPU or the models; Ollama is reached on loopback only."""
    out = {"gpu": None, "ollama": None, "mind_ctx": None, "warnings": []}
    ctx = (os.environ.get("KGIRL_MIND_CTX") or "").strip()
    out["mind_ctx"] = int(ctx) if ctx.isdigit() else None
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
        if r.returncode == 0 and r.stdout.strip():
            used, total, util, temp = (float(x) for x in r.stdout.strip().splitlines()[0].split(","))
            out["gpu"] = {"memory_used": used, "memory_total": total, "utilization": util, "temperature": temp}
            if temp >= 95:
                out["warnings"].append(f"GPU at {temp:.0f} C — heat alert (>= 95 C)")
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/ps", timeout=5) as resp:
            ps = json.load(resp)
        models = []
        for m in ps.get("models", []):
            size, vram = m.get("size") or 0, m.get("size_vram") or 0
            pct = round(vram / size * 100, 1) if size else 0.0
            details = m.get("details") or {}
            entry = {"name": m.get("name"), "size": size, "size_vram": vram, "gpu_pct": pct,
                     "parameter_size": details.get("parameter_size"), "quantization": details.get("quantization_level"),
                     "context_length": None}
            if size and vram < size:
                out["warnings"].append(f"{entry['name']} is {pct:.0f}% GPU — not 100% resident")
            try:
                req = urllib.request.Request("http://127.0.0.1:11434/api/show",
                                             data=json.dumps({"name": m.get("name")}).encode(),
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=5) as sr:
                    info = json.load(sr).get("model_info") or {}
                entry["context_length"] = next((v for k, v in info.items() if k.endswith(".context_length")), None)
            except (OSError, ValueError):
                pass
            models.append(entry)
        out["ollama"] = {"models": models}
    except (OSError, ValueError):
        out["ollama"] = {"error": "Ollama /api/ps unreachable on 127.0.0.1:11434"}
    return out


class Hub:
    """Fan-out of live events to SSE clients."""

    def __init__(self, maxsize: int = 2000):
        self._lock = threading.Lock()
        self._clients: list[queue.Queue] = []
        self._maxsize = maxsize

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(self._maxsize)
        with self._lock:
            self._clients.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._clients:
                self._clients.remove(q)

    def emit(self, kind: str, /, **data) -> dict:
        # kind is positional-only: job_start/job_end carry the job's own `kind` field in **data.
        event = {"type": kind, "ts": time.time(), **data}
        with self._lock:
            clients = list(self._clients)
        for q in clients:
            try:
                q.put_nowait(event)
            except queue.Full:
                try:
                    q.get_nowait()
                    q.put_nowait(event)
                except (queue.Empty, queue.Full):
                    pass
        return event


@dataclass
class Job:
    id: str
    kind: str
    label: str
    status: str = "running"          # running | done | error
    created: float = field(default_factory=time.time)
    ended: float | None = None
    result: Any = None               # JSON-ready; the HTTP layer converts before storing
    error: str = ""
    events: deque = field(default_factory=lambda: deque(maxlen=1000))

    def summary(self) -> dict:
        # last_event lets the tray flag a running job that has gone silent (stuck after 120 s).
        last = self.events[-1]["ts"] if self.events else self.created
        return {"id": self.id, "kind": self.kind, "label": self.label, "status": self.status,
                "created": self.created, "ended": self.ended, "error": self.error, "last_event": last}

    def full(self) -> dict:
        return {**self.summary(), "result": self.result, "events": list(self.events)}


class JobManager:
    """Long operations run here, one worker thread each. Results are fetched by id; the SSE stream
    carries deltas while they run."""

    def __init__(self, hub: Hub):
        self.hub = hub
        self.heavy = threading.Lock()
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[dict]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.created, reverse=True)[:50]
        return [j.summary() for j in jobs]

    def start(self, kind: str, label: str, fn: Callable[[Callable], Any], heavy: bool = False) -> Job:
        job = Job(uuid.uuid4().hex[:12], kind, label)
        with self._lock:
            self._jobs[job.id] = job
            if len(self._jobs) > 200:
                for old in sorted(self._jobs.values(), key=lambda j: j.created)[:-200]:
                    if old.status != "running":
                        del self._jobs[old.id]

        def emit(_kind: str, **data) -> None:
            ev = {"type": _kind, "ts": time.time(), **data}
            job.events.append(ev)
            self.hub.emit("job", job=job.id, event=ev)

        def worker() -> None:
            try:
                if heavy:
                    if self.heavy.locked():
                        emit("queued", text="waiting for the current operation to finish")
                    with self.heavy:
                        job.result = fn(emit)
                else:
                    job.result = fn(emit)
                job.status = "done"
            except Exception as exc:  # noqa: BLE001 - a failed job is a result the UI shows
                job.status, job.error = "error", f"{type(exc).__name__}: {exc}"
            finally:
                job.ended = time.time()
                self.hub.emit("job_end", job=job.id, kind=job.kind, status=job.status, error=job.error)

        try:
            self.hub.emit("job_start", job=job.id, kind=kind, label=label)
            threading.Thread(target=worker, daemon=True, name=f"console-{kind}-{job.id}").start()
        except Exception as exc:  # noqa: BLE001 - a job whose thread never started must not stay "running"
            job.status, job.error, job.ended = "error", f"{type(exc).__name__}: {exc}", time.time()
            self.hub.emit("job_end", job=job.id, kind=job.kind, status=job.status, error=job.error)
        return job


class Approvals:
    """The workspace's confirm callback, bridged to the browser."""

    def __init__(self, hub: Hub, timeout: float = 600.0):
        self.hub = hub
        self.timeout = timeout
        self._lock = threading.Lock()
        self._pending: dict[str, dict] = {}

    def ask(self, question: str, detail: str) -> bool:
        aid = uuid.uuid4().hex[:12]
        ap = {"id": aid, "question": question, "detail": detail, "created": time.time(),
              "event": threading.Event(), "answer": False}
        with self._lock:
            self._pending[aid] = ap
        self.hub.emit("approval", id=aid, question=question, detail=detail)
        try:
            ap["event"].wait(self.timeout)
            return bool(ap["answer"])
        finally:
            with self._lock:
                self._pending.pop(aid, None)
            self.hub.emit("approval_done", id=aid, answer=bool(ap["answer"]))

    def resolve(self, aid: str, yes: bool) -> bool:
        with self._lock:
            ap = self._pending.get(aid)
        if ap is None:
            return False
        ap["answer"] = bool(yes)
        ap["event"].set()
        return True

    def pending(self) -> list[dict]:
        with self._lock:
            return [{k: ap[k] for k in ("id", "question", "detail", "created")}
                    for ap in self._pending.values()]


class Console:
    """One Assistant, one shared workspace, a Mind per conversation plus one for direct tool calls."""

    def __init__(self, a: Assistant, workspace_root: Path | None = None, auto_edit: bool = False):
        self.a = a
        # Per-launch secret the page must echo on every /api call. A cross-site page cannot read it
        # (same-origin only), so it cannot forge a request even though the browser would send cookies.
        self.token = secrets.token_urlsafe(32)
        self.hub = Hub()
        self.jobs = JobManager(self.hub)
        self.approvals = Approvals(self.hub)
        self.store = ConversationStore(a.home / "conversations.db")
        self.workspace = (Workspace(workspace_root, confirm=self.approvals.ask, auto_edit=auto_edit,
                                    notify=lambda text: self.hub.emit("workspace", text=text))
                          if workspace_root is not None else None)
        self.search_lock = threading.Lock()   # retriever pipes are shared subprocess streams
        self._lock = threading.Lock()
        self._minds: dict[int, Mind] = {}
        self._chat_locks: dict[int, threading.Lock] = {}
        self._ops: Mind | None = None
        self._mcp = None
        self.retriever = None
        try:
            from ..atlas.retrieval import Retriever
            self.retriever = Retriever.from_env()
        except Exception:  # noqa: BLE001 - an unconfigured index is a state the UI shows, not a crash
            self.retriever = None
        a.bus.subscribe("*", self._on_bus, name="console")

    def _on_bus(self, env) -> None:
        self.hub.emit("bus", topic=env.topic, sender=env.sender, payload=env.payload,
                      id=env.id, correlation_id=env.correlation_id)

    def mind(self, conv_id: int) -> Mind:
        with self._lock:
            m = self._minds.get(conv_id)
            if m is None:
                m = Mind(self.a, store=self.store, workspace=self.workspace)
                self._minds[conv_id] = m
                self._chat_locks[conv_id] = threading.Lock()
            return m

    def chat_lock(self, conv_id: int) -> threading.Lock:
        with self._lock:
            return self._chat_locks.setdefault(conv_id, threading.Lock())

    def ops(self) -> Mind:
        """The Mind behind direct tool calls (Specialists, Forge): owns the console's last_task/last_build."""
        with self._lock:
            if self._ops is None:
                self._ops = Mind(self.a, store=self.store, workspace=self.workspace)
            return self._ops

    # ------------------------------------------------------------------ shared operations
    # The web console and the `kgirl ui` TUI both start work through these, so a job looks the same
    # (kind, label, result shape) whichever surface started it.

    def search(self, query: str, limit: int = 10) -> dict:
        """Symbols from the Atlas plus chunks from the retriever index, when one is configured."""
        limit = max(1, min(30, int(limit)))
        with self.search_lock:
            symbols = [plain(h) for h in self.a.atlas.search(query, limit=limit)]
            chunks = ([{**plain(h), "cite": h.cite()} for h in self.retriever.search(query, limit=limit)]
                      if self.retriever is not None else [])
        return {"symbols": symbols, "chunks": chunks}

    def rules(self) -> dict:
        """The standing rules: active preferences the mind-rules lane's is_rule() marks as rules.
        Read-only — editing stays with kgirl soup. Without that lane, available is False."""
        try:
            from ..mind.rules import is_rule
        except ImportError:
            return {"available": False, "rules": []}
        rules = [{"id": f.id, "text": f.text, "scope": f.scope}
                 for f in self.a.soup.all(status="active", kind="preference") if is_rule(f.text)]
        return {"available": True, "rules": rules}

    def chat_job(self, cid: int, text: str) -> Job:
        mind, lock = self.mind(cid), self.chat_lock(cid)

        def run(emit):
            with lock:
                before = self.store.last(cid)
                reply = mind.chat(cid, text, on_text=lambda d: emit("delta", text=d),
                                  on_event=lambda s: emit("status", text=s))
                tools = self.store.turns(cid, since_id=before.id if before else 0, roles=("tool",))
            audit = [e for t in tools for e in _dicts(t.meta.get("audit"))]
            return {"text": reply.text, "conv_id": reply.conv_id, "recalled": reply.recalled,
                    "tools": plain(reply.tools), "stored": reply.stored, "verdict": reply.verdict,
                    **answer_meta(self.store.last(cid, "assistant")), "audit": audit}

        return self.jobs.start("chat", text[:80], run, heavy=True)

    def task_job(self, mind: Mind, args: dict) -> Job:
        goal = str(args.get("goal", "")).strip()
        if not goal:
            raise ValueError("task needs a goal")

        def run(emit):
            mind.last_task = None
            emit("status", text="sandboxed specialists working the goal")
            text = mind.call_tool("task", args, on_event=lambda s: emit("status", text=s))
            if mind.last_task is not None:
                res, root = mind.last_task
                return {**swarm_dict(res, str(root)), "text": text}
            return {"text": text, "winner": None, "files": []}

        return self.jobs.start("task", goal[:80], run, heavy=True)

    def build_job(self, spec: str, candidates: int = 6, allow_io: bool = False, kind: str = "build") -> Job:
        if not spec.strip():
            raise ValueError("build needs a spec")

        def run(emit):
            emit("status", text="tests first, several candidates, agreement selection")
            res = self.ops().builder(n_impl=max(2, min(8, candidates)), allow_io=allow_io).build(spec)
            self.ops().last_build = res
            return build_dict(res)

        return self.jobs.start(kind, spec[:80], run, heavy=True)

    def write_job(self, mind: Mind, args: dict) -> Job:
        brief = str(args.get("brief", "")).strip()
        if not brief:
            raise ValueError("write needs a brief")

        def run(emit):
            text = mind.call_tool("write", args, on_event=lambda s: emit("status", text=s))
            body, draft = split_draft_footer(text)
            return {"markdown": body, "draft": draft}

        return self.jobs.start("write", brief[:80], run, heavy=True)

    def drafts(self, conv_id: int | None = None) -> dict:
        """Saved writer drafts, newest first, when the mind keeps them (the voice lane); else available=False."""
        lister = getattr(self.ops(), "drafts", None)
        if not callable(lister):
            return {"available": False, "drafts": []}
        return {"available": True, "drafts": [{"id": i, "conv_id": c, "title": t, "progress": p}
                                              for i, c, t, p in lister(conv_id)]}

    def resume_draft_job(self, draft_id: int) -> Job:
        """Continue draft `draft_id` where it stopped, as a write job (the result carries the draft id)."""
        mind = self.ops()
        if not callable(getattr(mind, "drafts", None)):
            raise ValueError("this build keeps no drafts")

        def run(emit):
            draft, art = mind.write(0, "", resume=draft_id, progress=lambda s: emit("status", text=s))
            return {"markdown": draft.markdown(), "draft": art}

        return self.jobs.start("write", f"resume draft #{draft_id}", run, heavy=True)

    def context(self, conv_id: int) -> dict:
        """What the latest prompt of a conversation held (the voice lane's prompt budget), or available=False."""
        mind = self.mind(conv_id)
        report = getattr(mind, "context_report", None)
        last = self.store.last(conv_id, "assistant")
        budget = answer_meta(last)["budget"] if last else None
        if not callable(report):
            return {"available": False, "budget": budget, "report": ""}
        return {"available": True, "budget": budget, "report": report(conv_id)}

    def doctor(self) -> dict:
        """The read-only health checks of `kgirl doctor` (the guard lane), called in-process; available=False
        without that lane. Blocking (a few local probes with short timeouts)."""
        try:
            from ..doctor import collect
        except ImportError:
            return {"available": False, "checks": []}
        return {"available": True, "checks": [plain(c) for c in collect(self.a.home)]}

    def set_verify(self, command: str) -> str | None:
        """The command that proves a swarm's change, for every Mind here (None clears it)."""
        cmd = shlex.split(command) if command.strip() else None
        with self._lock:
            minds = list(self._minds.values()) + ([self._ops] if self._ops else [])
        for m in minds:
            m.verify_cmd = cmd
        self.ops().verify_cmd = cmd
        return shlex.join(cmd) if cmd else None

    def allow(self, command: str) -> list[str]:
        """Let kgirl run `command` in the workspace without asking, for this session. Returns the allowed list."""
        if self.workspace is None:
            raise ValueError("no workspace attached")
        self.workspace.allow(command)
        return [shlex.join(x) for x in self.workspace.allowed_commands()]

    def ask_job(self, question: str, repo: str | None = None) -> Job:
        """A one-shot answer from the indexed repos and memory (`kgirl ask`): no conversation, nothing saved
        to one, no workspace tools. It uses the model, so it waits on the heavy lock like the other jobs."""
        question = question.strip()
        if not question:
            raise ValueError("ask needs a question")

        def run(emit):
            emit("status", text="atlas + memory -> one answer")
            ans = self.a.ask(question, repo)
            return {"text": ans.text, "intent": ans.intent, "backend": ans.backend,
                    "citations": list(ans.citations), "context_tokens": ans.context_tokens}

        return self.jobs.start("ask", question[:80], run, heavy=True)

    def helix_job(self, task: str, budget: int = 4000, session: str = "main", critic_model: str | None = None,
                  critic_host: str | None = None, experts: list[str] | None = None, lattice: bool = False,
                  expect: int | None = None) -> Job:
        """A HELIX-9 run. The caller vets critic_host (the HTTP layer allows loopback or KGIRL_CRITIC_HOSTS)."""
        task = task.strip()
        if not task:
            raise ValueError("helix run needs a task")

        def run(emit):
            from ..helix.__main__ import render
            from ..helix.harness import Helix, critic_chain
            from ..helix.store import StateStore
            from ..hermes.router import Router
            emit("status", text="experts -> CRITIC -> SYNTHESIST; code keeps the books")
            critic = critic_chain(critic_model, critic_host) if critic_model or critic_host else None
            with StateStore(self.a.home) as store:
                r = Helix(Router.from_env(), store=store, session=session, critic=critic,
                          bus=self.a.bus).run(task, budget, experts=experts, lattice=lattice, expect=expect)
            return {"result": r, "render": render(r)}

        return self.jobs.start("helix", task[:80], run, heavy=True)

    def mcp(self):
        with self._lock:
            if self._mcp is None:
                from ..mcp_server import MCPServer
                self._mcp = MCPServer(self.a)
            return self._mcp

    def close(self) -> None:
        # Join in-flight reflections only: end() would force a fresh model reflection per conversation,
        # which is the CLI's /quit behavior, not a server shutdown's.
        with self._lock:
            minds = list(self._minds.values()) + ([self._ops] if self._ops else [])
        for m in minds:
            t = m._reflecting
            if t is not None and t.is_alive():
                t.join(timeout=10)
        self.store.close()