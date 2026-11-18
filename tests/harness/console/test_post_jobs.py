"""Regression: every heavy job kind driven through the real HTTP _post path with a fake LLM must reach
done or error — never stay "running". A chat once crashed at start and the UI showed nothing; 45 green
tests missed it because none drove _post end to end. These do: real Assistant, real Console, real
ThreadingHTTPServer, real CSRF token; only the model is a ScriptedBackend. Written, not run.
"""

import http.client
import json
import os
import re
import tempfile
import threading
import time
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest import mock

import tests.harness.console  # noqa: F401  (sys.path)
from kgirl.harness.assistant import Assistant
from kgirl.harness.console.core import Console
from kgirl.harness.console.server import Handler
from kgirl.harness.hermes.router import BUILD, CHAT, JEV, SCOUT, SMITH, Router
from kgirl.harness.jev.agent import SYSTEM as JEV_SYSTEM
from kgirl.harness.jev.swarm import DISTILL_SYSTEM, SMITH_SYSTEM
from kgirl.harness.llm.scripted import ScriptedBackend
from kgirl.harness.mind.builder import IMPL_SYSTEM, REPAIR_SYSTEM, TEST_SYSTEM
from kgirl.harness.mind.mind import PERSONA
from kgirl.harness.mind.reflect import REFLECT_SYSTEM
from kgirl.harness.mind.writer import CRITIC_SYSTEM, DRAFT_SYSTEM, PLAN_SYSTEM, REVISE_SYSTEM

SPEC = "Implement add(a, b) returning the sum of two integers."
GOAL = "fix add in calc.py so it returns the sum"
CALC = "def add(a, b):\n    return a - b\n"
TEST_CALC = "from calc import add\n\nassert add(2, 3) == 5\nassert add(-1, 1) == 0\nprint('ok')\n"
VERIFY = ["python3", "test_calc.py"]

MODULE = "```python\ndef add(a: int, b: int) -> int:\n    \"\"\"Return the sum of a and b.\"\"\"\n    return a + b\n```"
SUITE = ("```python\nimport unittest\n\nfrom solution import add\n\n\n"
         "class TestAdd(unittest.TestCase):\n"
         "    def test_positive(self):\n        self.assertEqual(add(2, 3), 5)\n\n"
         "    def test_negative(self):\n        self.assertEqual(add(-1, 1), 0)\n\n"
         "    def test_zero(self):\n        self.assertEqual(add(0, 0), 0)\n```")
SMITH_FIX = "<<<<<<< SEARCH\n    return a - b\n=======\n    return a + b\n>>>>>>> REPLACE"
JEV_PREFIX = JEV_SYSTEM.split("{grammar}")[0]


def jev_decision(prompt: str) -> str:
    """Edit the add symbol when the table shows it; otherwise search for it. Never touches test files."""
    table = prompt.split("TABLE:\n", 1)[-1]
    for m in re.finditer(r"^\[(\d+)\]\s+\S+\s+(.*)$", table, re.M):
        if "def add" in m.group(2) and "test" not in m.group(2):
            return f"EDIT {m.group(1)}"
    return "SEARCH :: add calc"


def fake_llm(system: str, messages) -> str:
    if system.startswith(PERSONA):
        return "add(a, b) in calc.py returns the sum of its two arguments."
    if system.startswith(IMPL_SYSTEM) or system.startswith(REPAIR_SYSTEM):
        return MODULE
    if system.startswith(TEST_SYSTEM):
        return SUITE
    if system.startswith(PLAN_SYSTEM):
        return json.dumps({"title": "Sums", "audience": "engineers", "thesis": "add returns sums",
                           "sections": [{"heading": "Behavior", "points": ["add(2, 3) is 5"], "words": 120}]})
    if system.startswith(DRAFT_SYSTEM) or system.startswith(REVISE_SYSTEM):
        return "add(2, 3) returns 5, and negative inputs work too."
    if system.startswith(CRITIC_SYSTEM):
        return '{"issues": []}'
    if system.startswith(SMITH_SYSTEM):
        return SMITH_FIX
    if system.startswith(DISTILL_SYSTEM):
        return "Verify edits with python3 test_calc.py before claiming DONE."
    if system.startswith(REFLECT_SYSTEM):
        return ('{"summary": "A test conversation.", "title": "test chat", "threads": [],'
                ' "facts": [], "preferences": [], "lessons": [], "mistakes": []}')
    if system.startswith(JEV_PREFIX):
        return jev_decision(messages[-1].content)
    return "ok"


class PostJobs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        home, repo = root / "home", root / "repo"
        repo.mkdir()
        (repo / "calc.py").write_text(CALC)
        (repo / "test_calc.py").write_text(TEST_CALC)
        env = mock.patch.dict(os.environ, {"KGIRL_CHUNKS_DB": "", "KGIRL_MATRIX": ""})
        env.start()
        self.addCleanup(env.stop)
        self.backend = ScriptedBackend(fake_llm)
        router = Router({role: [self.backend] for role in (SCOUT, SMITH, JEV, CHAT, BUILD)})
        self.a = Assistant(home=home, router=router, life=False)
        self.addCleanup(self.a.close)
        self.console = Console(self.a, workspace_root=repo)
        self.addCleanup(self.console.close)
        self.console.ops().verify_cmd = VERIFY
        self.q = self.console.hub.subscribe()
        self.addCleanup(self.console.hub.unsubscribe, self.q)
        Handler.console = self.console
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)

    # ---------------------------------------------------------------- HTTP helpers

    def request(self, method, path, body=None, token=True):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        headers = {"X-Kgirl-Token": self.console.token} if token else {}
        data = None
        if body is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(body)
        conn.request(method, path, body=data, headers=headers)
        resp = conn.getresponse()
        out = json.loads(resp.read())
        conn.close()
        return resp.status, out

    def post(self, path, body, token=True):
        return self.request("POST", path, body, token)

    def get(self, path):
        return self.request("GET", path)

    def drain(self):
        out = []
        while not self.q.empty():
            out.append(self.q.get_nowait())
        return out

    def wait_job(self, job_id, timeout=90.0):
        deadline = time.time() + timeout
        last = None
        while time.time() < deadline:
            status, last = self.get("/api/jobs/" + job_id)
            self.assertEqual(status, 200)
            if last["status"] != "running":
                return last
            time.sleep(0.05)
        self.fail(f"job {job_id} still 'running' after {timeout:g}s: {last}")

    def assert_terminal(self, job_id):
        """The regression itself: a terminal status, an ended timestamp, and exactly one matching job_end."""
        full = self.wait_job(job_id)
        self.assertIn(full["status"], ("done", "error"))
        self.assertIsNotNone(full["ended"])
        deadline, seen = time.time() + 5, []
        while time.time() < deadline:
            seen.extend(self.drain())
            ends = [e for e in seen if e.get("type") == "job_end" and e.get("job") == job_id]
            if ends:
                self.assertEqual(len(ends), 1)
                self.assertEqual(ends[0]["status"], full["status"])
                self.assertTrue(any(e.get("type") == "job_start" and e.get("job") == job_id for e in seen))
                return full
            time.sleep(0.02)
        self.fail(f"no job_end event for {job_id}")

    # ---------------------------------------------------------------- the four heavy kinds

    def test_chat_reaches_a_terminal_state(self):
        status, conv = self.post("/api/conversations", {"title": "regression"})
        self.assertEqual(status, 200)
        status, d = self.post("/api/chat", {"conv_id": conv["id"], "text": "what does add do?"})
        self.assertEqual(status, 200)
        full = self.assert_terminal(d["job"])
        self.assertEqual(full["kind"], "chat")
        if full["status"] == "done":
            self.assertTrue(full["result"]["text"])
        status, jobs = self.get("/api/jobs")
        mine = next(j for j in jobs if j["id"] == d["job"])
        self.assertIn("last_event", mine)          # the tray measures stuck silence from this

    def test_build_reaches_a_terminal_state(self):
        status, d = self.post("/api/build", {"spec": SPEC, "candidates": 2})
        self.assertEqual(status, 200)
        full = self.assert_terminal(d["job"])
        self.assertEqual(full["kind"], "build")
        if full["status"] == "done":
            self.assertIn("code", full["result"])
            self.assertIn("confidence", full["result"])

    def test_write_reaches_a_terminal_state(self):
        status, d = self.post("/api/write", {"brief": "explain what add does", "words": 200})
        self.assertEqual(status, 200)
        full = self.assert_terminal(d["job"])
        self.assertEqual(full["kind"], "write")
        if full["status"] == "done":
            self.assertTrue(full["result"]["markdown"].startswith("#"))

    def test_task_reaches_a_terminal_state(self):
        status, d = self.post("/api/task", {"goal": GOAL})
        self.assertEqual(status, 200)
        full = self.assert_terminal(d["job"])
        self.assertEqual(full["kind"], "task")
        if full["status"] == "done":
            self.assertIn("files", full["result"])
            self.assertIsNotNone(full["result"]["winner"])   # the scripted smith fix passes the verifier

    # ---------------------------------------------------------------- failure must be loud too

    def test_dead_model_ends_the_chat_job_in_error(self):
        self.a.router.chains[CHAT] = [ScriptedBackend([], ok=False)]
        status, conv = self.post("/api/conversations", {"title": "dead model"})
        self.assertEqual(status, 200)
        status, d = self.post("/api/chat", {"conv_id": conv["id"], "text": "hello"})
        self.assertEqual(status, 200)
        full = self.assert_terminal(d["job"])
        self.assertEqual(full["status"], "error")
        self.assertIn("NoBackend", full["error"])

    def test_post_without_the_token_is_refused(self):
        status, _ = self.post("/api/chat", {"conv_id": 1, "text": "x"}, token=False)
        self.assertEqual(status, 403)


if __name__ == "__main__":
    unittest.main()
