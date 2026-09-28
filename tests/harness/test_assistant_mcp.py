import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

from tests.harness._fixtures import SRC, make_repos

from kgirl.harness.assistant import Assistant
from kgirl.harness.hermes import Router
from kgirl.harness.llm import ScriptedBackend
from kgirl.harness.mcp_server import MCPServer
from kgirl.harness.report import atlas_report


def scout(system, messages):
    ctx = messages[-1].content
    return "Engine.run doubles helper(x) [lib:engine.py:5]" if "engine.py" in ctx else "unknown"


class AssistantTests(unittest.TestCase):
    def setUp(self):
        self.tmp, self.hub, self.lib = make_repos()
        home = Path(self.tmp.name) / "home"
        self.router = Router({"scout": [ScriptedBackend(scout)], "smith": [], "jev": []})
        self.a = Assistant(home=home, router=self.router, trace=True)
        self.a.index([str(self.lib), str(self.hub)])

    def tearDown(self):
        self.a.close()
        self.tmp.cleanup()

    def test_ask_uses_scout_with_citations_and_trace(self):
        ans = self.a.ask("how does Engine run work")
        self.assertIn("doubles", ans.text)
        self.assertTrue(any(c.startswith("lib:engine.py") for c in ans.citations))
        self.assertEqual(ans.backend, "scripted:scripted")
        topics = [e.topic for e in self.a.bus.trace()]
        for t in ("assistant.ask", "atlas.search", "soup.recall", "llm.complete", "assistant.answer"):
            self.assertIn(t, topics)
        self.assertTrue((self.a.home / "trace.jsonl").exists())

    def test_blast_and_arch_intents_are_deterministic(self):
        ans = self.a.ask("what is the blast radius of `engine.py`")
        self.assertEqual(ans.intent, "blast")
        self.assertIn("hub:app.py", ans.text)
        ans = self.a.ask("architecture overview of lib")
        self.assertIn("## lib", ans.text)

    def test_no_backend_degrades_to_extractive(self):
        self.a.router.chains["scout"] = [ScriptedBackend([], ok=False)]
        ans = self.a.ask("how does Engine run work")
        self.assertEqual(ans.backend, "extractive")
        self.assertIn("Engine", ans.text)

    def test_index_writes_repo_facts_and_report(self):
        facts = self.a.soup.all(kind="fact")
        self.assertEqual({f.scope for f in facts}, {"lib", "hub"})
        report = atlas_report(self.a.atlas, "hub")
        self.assertIn("| lib | 1 |", report)
        self.assertIn("## Repository cards", report)


class MCPTests(unittest.TestCase):
    def setUp(self):
        self.tmp, self.hub, self.lib = make_repos()
        self.home = Path(self.tmp.name) / "home"
        a = Assistant(home=self.home, router=Router({"scout": [], "smith": [], "jev": []}), trace=False)
        a.index([str(self.lib), str(self.hub)])
        self.a = a
        self.srv = MCPServer(a)

    def tearDown(self):
        self.a.close()
        self.tmp.cleanup()

    def call(self, method, params=None, mid=1):
        return self.srv.handle({"jsonrpc": "2.0", "id": mid, "method": method, "params": params or {}})

    def test_handshake_list_and_call(self):
        init = self.call("initialize", {"protocolVersion": "2025-06-18", "capabilities": {}})
        self.assertEqual(init["result"]["serverInfo"]["name"], "kgirl-harness")
        self.assertIsNone(self.srv.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}))
        names = {t["name"] for t in self.call("tools/list")["result"]["tools"]}
        self.assertTrue({"atlas_search", "atlas_blast_radius", "soup_recall", "jev_swarm_task"} <= names)
        res = self.call("tools/call", {"name": "atlas_blast_radius", "arguments": {"target": "lib:engine.py"}})
        self.assertFalse(res["result"]["isError"])
        self.assertIn("hub:app.py", res["result"]["content"][0]["text"])
        bad = self.call("tools/call", {"name": "atlas_card", "arguments": {"repo": "nope"}})
        self.assertTrue(bad["result"]["isError"])
        self.assertEqual(self.call("nope/method")["error"]["code"], -32601)

    def test_stdio_roundtrip(self):
        msgs = [{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
                {"jsonrpc": "2.0", "method": "notifications/initialized"},
                {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                 "params": {"name": "atlas_search", "arguments": {"query": "Engine"}}}]
        env = {**os.environ, "PYTHONPATH": str(SRC), "KGIRL_SCOUT_MODEL": "none", "OLLAMA_HOST": "http://127.0.0.1:9"}
        p = subprocess.run([sys.executable, "-m", "kgirl.harness", "--home", str(self.home), "mcp"],
                           input="\n".join(json.dumps(m) for m in msgs) + "\n", capture_output=True, text=True,
                           env=env, timeout=60)
        lines = [json.loads(ln) for ln in p.stdout.splitlines() if ln.strip()]
        self.assertEqual([m["id"] for m in lines], [1, 2], p.stderr)
        self.assertIn("lib:engine.py", lines[1]["result"]["content"][0]["text"])


if __name__ == "__main__":
    unittest.main()
