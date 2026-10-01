"""Tests for blast_radius_check.py: python -m unittest scripts/test_blast_radius_check.py"""

import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from blast_radius_check import check, main  # noqa: E402

# Shaped like a real dict-dispatch server before review (kgirl harness, commit 26748c4).
UNSAFE_PY = '''
import os, subprocess

def _tools(a):
    def remember(p):
        fid = a.remember(p["text"], p.get("kind", "note"))
        return f"stored #{fid}"

    def note(p):
        return a.memory.add(p["text"], source="user")

    def task(p):
        verify = p.get("verify")
        return a.task(p["goal"], verify.split() if isinstance(verify, str) else verify)

    def save(p):
        with open(p["path"], "w") as fh:
            fh.write(p["text"])
    return {"remember": remember, "note": note, "task": task, "save": save}

def handle(params):
    return {"protocolVersion": params.get("protocolVersion", "2025-06-18"), "capabilities": {}}

def verify(cmd):
    return subprocess.run(cmd, env={**os.environ, "X": "1"})
'''

SAFE_PY = '''
import os, shlex, subprocess
from pathlib import Path

ALLOWED = [["python", "-m", "pytest", "-q"]]
SUPPORTED = ("2025-06-18", "2025-03-26")
ROOT = Path("/srv/data").resolve()

def _tools(a):
    def remember(p):
        return a.memory.add(p["text"], source="mcp", status="staged")

    def task(p):
        cmd = shlex.split(p.get("verify") or "")
        if cmd and cmd not in ALLOWED:
            raise PermissionError("not allowed")
        return a.task(p["goal"], cmd)

    def save(p):
        target = (ROOT / p["path"]).resolve()
        if not target.is_relative_to(ROOT):
            raise PermissionError("outside root")
        target.write_text(p["text"])
    return {"remember": remember, "task": task, "save": save}

def handle(params):
    requested = params.get("protocolVersion")
    return {"protocolVersion": requested if requested in SUPPORTED else SUPPORTED[0]}

def verify(cmd):
    env = {k: v for k, v in os.environ.items() if "KEY" not in k}
    return subprocess.run(["git", "status"], env=env)
'''

FASTMCP_UNSAFE = '''
import subprocess
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("ops")

@mcp.tool()
async def run(command: str) -> str:
    return subprocess.run(command, shell=True, capture_output=True, text=True).stdout
'''

FASTMCP_SAFE = '''
import subprocess
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("ops")
CHECKS = {"lint": ["ruff", "check", "."], "test": ["pytest", "-q"]}

@mcp.tool()
async def run(check: str) -> str:
    if check not in CHECKS:
        raise ValueError("unknown check")
    return subprocess.run(CHECKS[check], capture_output=True, text=True).stdout
'''

UNSAFE_JS = '''
import { spawn } from "node:child_process";
server.tool("run", async (args) => {
  const child = spawn(args.command, { env: process.env });
});
const reply = { protocolVersion: msg.params?.protocolVersion, capabilities: {} };
'''

SAFE_JS = '''
const SUPPORTED = ["2025-06-18"];
const requested = msg.params?.protocolVersion;
const reply = { protocolVersion: SUPPORTED.includes(requested) ? requested : SUPPORTED[0] };
'''


class CheckerTests(unittest.TestCase):
    def scan(self, name, source, **kw):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / name
            f.write_text(textwrap.dedent(source))
            return [(x.line, x.rule) for x in check([d], **kw)]

    def rules(self, name, source):
        return sorted({r for _, r in self.scan(name, source)})

    def test_unsafe_python_server_trips_every_rule(self):
        found = self.scan("server.py", UNSAFE_PY)
        self.assertEqual(sorted({r for _, r in found}), ["BR001", "BR002", "BR003", "BR004", "BR005"])
        self.assertEqual(sum(1 for _, r in found if r == "BR003"), 3)  # live .remember, live .add, source="user"

    def test_reviewed_python_server_is_clean(self):
        self.assertEqual(self.scan("server.py", SAFE_PY), [])

    def test_fastmcp_decorated_tools(self):
        self.assertEqual(self.rules("ops.py", FASTMCP_UNSAFE), ["BR001"])
        self.assertEqual(self.scan("ops.py", FASTMCP_SAFE), [])

    def test_javascript_rules(self):
        self.assertEqual(self.rules("server.mjs", UNSAFE_JS), ["BR001", "BR002", "BR004"])
        self.assertEqual(self.scan("server.mjs", SAFE_JS), [])

    def test_cli_functions_taking_argparse_args_are_not_tool_handlers(self):
        cli = 'def _soup(a, args):\n    a.soup.add(args.text, source="user")\n    open(args.out, "w").write("x")\n'
        self.assertEqual(self.scan("cli.py", cli), [])

    def test_fixed_program_with_internal_arguments_is_not_flagged(self):
        src = 'import subprocess\ndef handler(params):\n    return subprocess.run(["git", "log", params["ref"]])\n'
        self.assertEqual(self.scan("s.py", src), [])
        shell = 'import subprocess\ndef handler(params):\n    return subprocess.run("git log " + params["ref"], shell=True)\n'
        self.assertEqual(self.rules("s.py", shell), ["BR001"])

    def test_tests_are_skipped_unless_asked(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "tests").mkdir()
            (Path(d) / "tests" / "t.py").write_text("import os, subprocess\nsubprocess.run(['x'], env=os.environ)\n")
            self.assertEqual(check([d]), [])
            self.assertEqual([x.rule for x in check([d], include_tests=True)], ["BR002"])

    def test_unparseable_file_is_reported_not_crashed(self):
        self.assertEqual(self.rules("broken.py", "def (:\n"), ["BR000"])

    def test_exit_status(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "ok.py").write_text("x = 1\n")
            self.assertEqual(main([d]), 0)
            (Path(d) / "bad.py").write_text(textwrap.dedent(FASTMCP_UNSAFE))
            self.assertEqual(main([d, "--json"]), 1)


if __name__ == "__main__":
    unittest.main()
