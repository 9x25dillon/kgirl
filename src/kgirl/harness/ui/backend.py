"""The TUI's view of the harness: a thin adapter over the console's in-process Console.

No textual here, so it imports anywhere. The app talks only to this interface; tests hand the app a fake
with the same methods, so a pilot run generates nothing.

Events arrive on `events()`, a queue fed by the console Hub, in the same shapes the web console's SSE
stream carries: job_start, job (delta/status/queued), job_end, approval, approval_done, workspace, bus.
"""

from __future__ import annotations

import queue
import shlex
from pathlib import Path

from ..console.core import Console, answer_meta, gpu_status


class HarnessBackend:
    def __init__(self, a, workspace_root: Path | None = None, auto_edit: bool = False,
                 code_root: Path | None = None):
        from ..console.server import CodeWatch          # git reads only; the startup fingerprint
        self.console = Console(a, workspace_root=workspace_root, auto_edit=auto_edit)
        self.code_watch = CodeWatch(code_root)
        self._queue = self.console.hub.subscribe()

    # ------------------------------------------------------------------ events
    def events(self) -> queue.Queue:
        return self._queue

    # ------------------------------------------------------------------ conversations
    def conversation(self, new: bool = False, title: str = "") -> int:
        ops = self.console.ops()
        return ops.start(title) if new else ops.resume()

    def exists(self, conv_id: int) -> bool:
        return self.console.store.get(conv_id) is not None

    def title(self, conv_id: int) -> str:
        conv = self.console.store.get(conv_id)
        return (conv.title or "") if conv else ""

    def history(self, conv_id: int, limit: int = 6) -> list[dict]:
        """The last `limit` user messages and answers, oldest first. Partial and tool turns are not shown;
        the workspace audit records of tool turns ride on the answer that followed them."""
        out: list[dict] = []
        audit: list[dict] = []
        for t in self.console.store.turns(conv_id, roles=("user", "assistant", "tool")):
            if t.role == "tool":
                audit += [e for e in t.meta.get("audit") or [] if isinstance(e, dict)]
            elif t.role == "user":
                out.append({"role": "user", "text": t.content})
                audit = []
            elif not t.partial:
                out.append({"role": "assistant", "text": t.content, **answer_meta(t), "audit": audit})
                audit = []
        return out[-limit:]

    def conversations(self, limit: int = 15) -> list[dict]:
        return [{"id": c.id, "title": c.title, "summary": c.summary, "scope": c.scope}
                for c in self.console.store.list(limit)]

    def find(self, query: str, limit: int = 8) -> list[dict]:
        """Past messages matching `query`, across conversations."""
        return [{"conv_id": t.conv_id, "role": t.role, "text": t.content}
                for t in self.console.store.search(query, limit=limit)]

    def recap(self, conv_id: int) -> str:
        return self.console.mind(conv_id).recap(conv_id)

    # ------------------------------------------------------------------ chat
    def chat(self, conv_id: int, text: str) -> str:
        return self.console.chat_job(conv_id, text).id

    def feedback(self, conv_id: int, verdict: str) -> str:
        return self.console.mind(conv_id).feedback(conv_id, verdict)

    def approve(self, approval_id: str, yes: bool) -> bool:
        return self.console.approvals.resolve(approval_id, yes)

    def rules(self) -> dict:
        return self.console.rules()

    def context(self, conv_id: int) -> dict:
        return self.console.context(conv_id)

    def doctor(self) -> dict:
        """Blocking (local probes with short timeouts): call off the UI loop."""
        return self.console.doctor()

    # ------------------------------------------------------------------ workspace
    def workspace(self) -> dict | None:
        ws = self.console.workspace
        if ws is None:
            return None
        return {"root": str(ws.root), "auto_edit": ws.auto_edit,
                "allowed": [shlex.join(x) for x in ws.allowed_commands()]}

    def diff(self) -> str:
        ws = self.console.workspace
        return (ws.diff() or "kgirl has changed nothing in this session") if ws else NO_WORKSPACE

    def undo(self) -> str:
        ws = self.console.workspace
        return ws.undo() if ws else NO_WORKSPACE

    def set_auto(self, on: bool) -> str:
        ws = self.console.workspace
        if ws is None:
            return NO_WORKSPACE
        ws.auto_edit = on
        return ("edits now apply without asking (still undoable with /undo)" if on
                else "edits now apply only after you approve them")

    def apply(self, conv_id: int) -> str:
        return self.console.mind(conv_id).apply_task()

    def set_verify(self, command: str) -> str | None:
        return self.console.set_verify(command)

    def allow(self, command: str) -> str:
        if self.console.workspace is None:
            return NO_WORKSPACE
        allowed = self.console.allow(command)
        return f"kgirl may now run without asking: {', '.join(allowed)}"

    # ------------------------------------------------------------------ jobs
    def task(self, conv_id: int, goal: str) -> str:
        return self.console.task_job(self.console.mind(conv_id), {"goal": goal}).id

    def build(self, spec: str) -> str:
        return self.console.build_job(spec).id

    def write(self, conv_id: int, brief: str) -> str:
        return self.console.write_job(self.console.mind(conv_id), {"brief": brief}).id

    def drafts(self) -> dict:
        return self.console.drafts()

    def resume_draft(self, draft_id: int) -> str:
        return self.console.resume_draft_job(draft_id).id

    def ask(self, question: str) -> str:
        return self.console.ask_job(question).id

    def helix(self, task: str, budget: int = 4000) -> str:
        return self.console.helix_job(task, budget=budget).id

    def jobs(self) -> list[dict]:
        return self.console.jobs.list()

    def job(self, job_id: str) -> dict | None:
        job = self.console.jobs.get(job_id)
        return job.full() if job is not None else None

    # ------------------------------------------------------------------ status and search
    def status(self) -> dict:
        """Router roles, GPU and loaded models. Blocking (nvidia-smi, loopback Ollama): call off the UI loop."""
        route = {}
        for role in ("chat", "build", "scout", "smith"):
            try:
                b = self.console.a.router.pick(role)
                route[role] = getattr(b, "model", None) or (type(b).__name__ if b else None)
            except Exception:  # noqa: BLE001 - a role without a backend is a state, not an error
                route[role] = None
        return {"route": route, **gpu_status(), "retriever": self.console.retriever is not None}

    def code(self) -> dict:
        """The CodeWatch restart hint. Blocking (git, at most once a minute): call off the UI loop."""
        return self.code_watch.status()

    def search(self, query: str, limit: int = 20) -> dict:
        return self.console.search(query, limit)

    def close(self) -> None:
        self.console.hub.unsubscribe(self._queue)
        self.console.close()


NO_WORKSPACE = "no project directory is attached (start kgirl ui inside one, or pass --dir)"
