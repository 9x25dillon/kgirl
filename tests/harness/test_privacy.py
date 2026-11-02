import os
import re
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from kgirl.harness.assistant import Assistant
from kgirl.harness.hermes.router import Router
from kgirl.harness.hermes.router import JEV, SCOUT, SMITH

ROOT = Path(__file__).resolve().parents[2]
HARNESS = ROOT / "src/kgirl/harness"
SERVERS = ["src/kgirl/main.py", "src/kgirl/api/integrated_api_server.py", "src/kgirl/cognitive/recursive_ai_core.py",
           "src/julia/Server.jl", "src/julia/server.jl", "src/julia/setup_limps_service.jl"]
TRACKERS = re.compile(r"^\s*(import|from)\s+(sentry_sdk|posthog|mixpanel|segment|amplitude|stripe|web3)\b", re.M)
ALLOWED_NETWORK = {"llm/anthropic_backend.py", "llm/ollama_backend.py", "atlas/vectors.py", "atlas/retrieval.py",
                   "atlas/sources.py", "jev/browser.py", "jev/browser_env.py", "mcp_server.py"}
NETWORK = re.compile(r"\b(urlopen|requests\.(get|post|put)|httpx\.|socket\.socket|aiohttp\.)")


class StateTests(unittest.TestCase):
    def test_state_folder_is_private_even_if_it_already_existed_open(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            home.mkdir(mode=0o755)
            home.chmod(0o755)
            a = Assistant(home=home, trace=False)
            self.addCleanup(a.close)
            self.assertEqual(stat.S_IMODE(home.stat().st_mode), 0o700)


class RoutingTests(unittest.TestCase):
    def names(self, env):
        with mock.patch.dict(os.environ, env, clear=False):
            r = Router.from_env()
        return {role: [b.name for b in r.chains[role]] for role in (SCOUT, SMITH, JEV)}

    def test_local_only_removes_every_cloud_backend(self):
        got = self.names({"KGIRL_LOCAL_ONLY": "1", "KGIRL_ALLOW_CLOUD_CODE": "1"})
        for chain in got.values():
            self.assertNotIn("anthropic", chain)
            self.assertTrue(chain)

    def test_code_edits_stay_local_unless_cloud_is_allowed(self):
        with mock.patch.dict(os.environ):
            os.environ.pop("KGIRL_ALLOW_CLOUD_CODE", None)
            os.environ.pop("KGIRL_LOCAL_ONLY", None)
            self.assertEqual(self.names({})[SMITH], ["ollama"])


class SourceGuardTests(unittest.TestCase):
    def sources(self):
        return [(p, p.read_text(errors="replace")) for p in HARNESS.rglob("*.py")]

    def test_harness_has_no_tracking_or_payment_imports(self):
        for path, text in self.sources():
            self.assertIsNone(TRACKERS.search(text), path)

    def test_only_known_modules_touch_the_network(self):
        for path, text in self.sources():
            if NETWORK.search(text):
                self.assertIn(path.relative_to(HARNESS).as_posix(), ALLOWED_NETWORK)

    def test_servers_do_not_bind_every_interface_by_default(self):
        for rel in SERVERS:
            text = (ROOT / rel).read_text(errors="replace")
            for line in text.splitlines():
                if "0.0.0.0" in line and re.search(r"host\s*=|serve\(|uvicorn\.run", line):
                    self.fail(f"{rel}: {line.strip()}")


if __name__ == "__main__":
    unittest.main()
