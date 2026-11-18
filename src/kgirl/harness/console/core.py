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

import queue
import secrets
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..assistant import Assistant
from ..mind import Mind
from ..mind.conversations import ConversationStore
from ..mind.workspace import Workspace


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