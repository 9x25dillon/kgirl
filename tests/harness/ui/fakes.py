"""A fake `kgirl ui` backend: the HarnessBackend interface with no harness and no model behind it.

Tests drive the app by pushing console-shaped events onto `inbox` (exactly what the Hub would carry) and
read back what the app asked for in `calls`.
"""

from __future__ import annotations

import queue
import time


class FakeBackend:
    def __init__(self, history=None, workspace=None, code=None, jobs=None, results=None):
        self.inbox: queue.Queue = queue.Queue()
        self.calls: list[tuple] = []
        self._history = list(history or [])
        self._workspace = workspace
        self._code = code or {"available": True, "root": "/tmp/code", "startup": "abc", "current": "abc",
                              "changed": False}
        self._jobs: list[dict] = list(jobs or [])
        self._results: dict[str, dict] = dict(results or {})
        self.next_conv = 7
        self._n = 0

    # events
    def events(self) -> queue.Queue:
        return self.inbox

    def push(self, event: str, /, **data) -> None:
        # positional-only: job_start/job_end carry their own `kind` field, as on the real Hub
        self.inbox.put({"type": event, "ts": time.time(), **data})

    # conversations
    def conversation(self, new=False, title=""):
        self.calls.append(("conversation", new, title))
        conv = self.next_conv
        if new:
            self.next_conv += 1
            self._history = []
        return conv

    def exists(self, conv_id):
        return conv_id in (7, 8, 9)

    def title(self, conv_id):
        return "a test chat"

    def history(self, conv_id, limit=6):
        return self._history[-limit:]

    def conversations(self, limit=15):
        return [{"id": 7, "title": "a test chat", "summary": "testing", "scope": "*"}]

    def find(self, query, limit=8):
        self.calls.append(("find", query))
        return [{"conv_id": 7, "role": "user", "text": f"something about {query}"}]

    def recap(self, conv_id):
        return f"conversation #{conv_id}: recap"

    # chat
    def _job(self, kind: str, label: str) -> str:
        self._n += 1
        jid = f"{kind[:2]}{self._n:010d}"
        self._jobs.insert(0, {"id": jid, "kind": kind, "label": label, "status": "running", "created": time.time(),
                              "ended": None, "error": "", "last_event": time.time()})
        return jid

    def chat(self, conv_id, text):
        self.calls.append(("chat", conv_id, text))
        return self._job("chat", text)

    def feedback(self, conv_id, verdict):
        self.calls.append(("feedback", conv_id, verdict))
        return f"noted: {verdict}"

    def approve(self, approval_id, yes):
        self.calls.append(("approve", approval_id, yes))
        return True

    def rules(self):
        return {"available": True, "rules": [{"id": 3, "text": "never force-push", "scope": "*"}]}

    def context(self, conv_id):
        self.calls.append(("context", conv_id))
        return {"available": True, "budget": None, "report": "prompt budget 3000 tokens, used 1200\n  nothing was cut"}

    def doctor(self):
        self.calls.append(("doctor",))
        return {"available": True, "checks": [
            {"name": "ollama", "status": "OK", "detail": "up on 127.0.0.1", "fix": ""},
            {"name": "chunk index", "status": "FAIL", "detail": "missing", "fix": "run kgirl chunks"}]}

    # workspace
    def workspace(self):
        return self._workspace

    def diff(self):
        self.calls.append(("diff",))
        return "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-old\n+new"

    def undo(self):
        self.calls.append(("undo",))
        return "undid the change to x.py"

    def set_auto(self, on):
        self.calls.append(("auto", on))
        if self._workspace is not None:
            self._workspace["auto_edit"] = on
        return "auto " + ("on" if on else "off")

    def apply(self, conv_id):
        self.calls.append(("apply", conv_id))
        return "applied to /tmp/repo: x.py"

    def set_verify(self, command):
        self.calls.append(("verify", command))
        return command or None

    def allow(self, command):
        self.calls.append(("allow", command))
        return f"kgirl may now run without asking: {command}"

    # jobs
    def write(self, conv_id, brief):
        self.calls.append(("write", conv_id, brief))
        return self._job("write", brief)

    def drafts(self):
        return {"available": True, "drafts": [{"id": 4, "conv_id": 7, "title": "release notes",
                                               "progress": "drafting: 2/5 sections"}]}

    def resume_draft(self, draft_id):
        self.calls.append(("resume", draft_id))
        return self._job("write", f"resume draft #{draft_id}")

    def task(self, conv_id, goal):
        self.calls.append(("task", conv_id, goal))
        return self._job("task", goal)

    def build(self, spec):
        self.calls.append(("build", spec))
        return self._job("build", spec)

    def ask(self, question):
        self.calls.append(("ask", question))
        return self._job("ask", question)

    def helix(self, task, budget=4000):
        self.calls.append(("helix", task))
        return self._job("helix", task)

    def jobs(self):
        return list(self._jobs)

    def job(self, job_id):
        for j in self._jobs:
            if j["id"] == job_id:
                return {**j, "result": self._results.get(job_id), "events": []}
        return None

    def finish(self, job_id, status="done", result=None, error=""):
        """End a job: update the tray's view of it, store its result, and emit job_end."""
        for j in self._jobs:
            if j["id"] == job_id:
                j.update(status=status, ended=time.time(), error=error)
                kind = j["kind"]
        self._results[job_id] = result
        self.push("job_end", job=job_id, kind=kind, status=status, error=error)

    # status and search
    def status(self):
        return {"route": {"chat": "qwen-test", "build": None}, "gpu": None, "ollama": {"models": []},
                "warnings": [], "retriever": False}

    def code(self):
        return dict(self._code)

    def search(self, query, limit=20):
        self.calls.append(("search", query))
        return {"symbols": [{"repo": "kgirl", "path": "src/a.py", "line": 3, "kind": "function",
                             "qualname": "do_thing", "signature": "def do_thing(x)"}],
                "chunks": []}

    def close(self):
        self.calls.append(("close",))
