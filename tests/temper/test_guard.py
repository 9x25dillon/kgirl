"""The guard runs as Claude Code runs it: a subprocess fed hook JSON on stdin.

    python -m unittest discover -s tests/temper -t .
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

GUARD = Path(__file__).resolve().parents[2] / ".claude" / "hooks" / "temper_guard.py"


class GuardCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.project = Path(os.path.realpath(self._tmp.name))
        for d in ("runs/temper/a", "src/kgirl/harness", "docs", "tests/harness"):
            (self.project / d).mkdir(parents=True, exist_ok=True)
        self.env = {k: v for k, v in os.environ.items() if k != "KGIRL_HOME"}
        self.env["CLAUDE_PROJECT_DIR"] = str(self.project)

    def tearDown(self):
        self._tmp.cleanup()

    def run_guard(self, payload, raw=None):
        data = raw if raw is not None else json.dumps({"cwd": str(self.project), "hook_event_name": "PreToolUse",
                                                       **payload})
        return subprocess.run([sys.executable, str(GUARD)], input=data, capture_output=True, text=True,
                              env=self.env, timeout=30)

    def bash(self, command):
        return self.run_guard({"tool_name": "Bash", "tool_input": {"command": command}})

    def assert_allowed(self, r):
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout, "", "an allowed call makes no decision (it never approves)")

    def assert_denied(self, r, needle=""):
        self.assertEqual(r.returncode, 2, f"expected a refusal, got {r.returncode}: {r.stdout}")
        out = json.loads(r.stdout)["hookSpecificOutput"]
        self.assertEqual(out["permissionDecision"], "deny")
        self.assertIn(needle, out["permissionDecisionReason"])


class AllowedWork(GuardCase):
    def test_read_only_tools(self):
        for tool in ("Read", "Grep", "Glob", "LS"):
            self.assert_allowed(self.run_guard({"tool_name": tool, "tool_input": {"file_path": "/etc/passwd"}}))

    def test_write_under_runs_temper(self):
        for p in ("runs/temper/a/report.json", str(self.project / "runs/temper/ledger.jsonl"),
                  "runs/temper/new/dir/plan.json"):
            self.assert_allowed(self.run_guard({"tool_name": "Write", "tool_input": {"file_path": p, "content": "{}"}}))

    def test_bench_and_analysis_commands(self):
        (self.project / "home").mkdir()
        for cmd in (
            "git rev-parse HEAD",
            "git log --oneline -5 && git status --short",
            "git -C . diff --stat",
            "git branch --show-current",
            "PYTHONPATH=src python3 -m kgirl.evolve fuel -g 300 --seed 0.4123 --out runs/temper/a/elites.json",
            "PYTHONPATH=src python3 -m kgirl.iwme verify --json > runs/temper/a/iwme.json",
            "PYTHONPATH=src python3 -m kgirl.neutronics fit --out runs/temper/a/neutronics",
            "PYTHONPATH=src python3 -m kgirl.tauls hpsearch --out runs/temper/a/hp --steps 400",
            "python3 -m unittest discover -s tests/harness -t .",
            "python3 runs/temper/scripts/paired.py runs/temper/a 2>&1 | tee runs/temper/a/log.txt",
            "mkdir -p runs/temper/a/scripts",
            f"cp -r {self.project / 'home'} runs/temper/a/home",
            "cd runs/temper && ls > listing.txt",
            "timeout 600 python3 -m kgirl.evolve fuel --out runs/temper/b.json",
            "PYTHONPATH=src python3 -m kgirl.harness --home runs/temper/a/home soup curate",
            "KGIRL_HOME=runs/temper/a/home PYTHONPATH=src python3 -m kgirl.harness task 'fix mean' --repo . "
            "--verify 'python3 -m pytest -q tests/test_x.py'",
            "PYTHONPATH=src python3 -m kgirl.harness blast kgirl:src/kgirl/x.py",
            "PYTHONPATH=src python3 -m kgirl.harness soup export-sft runs/temper/a/sft.jsonl",
            "PYTHONPATH=src python3 -m kgirl.harness --home runs/temper/a/home report -o runs/temper/a/report.md",
            "ollama list",
            "echo done >&2",
            "ls -la; wc -l runs/temper/ledger.jsonl",
        ):
            with self.subTest(cmd=cmd):
                self.assert_allowed(self.bash(cmd))


class Refusals(GuardCase):
    def test_other_tools_fail_closed(self):
        for tool in ("Edit", "NotebookEdit", "MultiEdit", "WebFetch", "Agent", ""):
            with self.subTest(tool=tool):
                self.assert_denied(self.run_guard({"tool_name": tool, "tool_input": {"file_path": "runs/temper/x"}}))

    def test_writes_outside(self):
        for p in ("src/kgirl/x.py", "../outside.txt", "runs/temper/../../x", ".claude/agents/temper.md",
                  "runs/temperature/x"):
            with self.subTest(path=p):
                self.assert_denied(self.run_guard({"tool_name": "Write", "tool_input": {"file_path": p}}), "outside")

    def test_symlink_escapes(self):
        (self.project / "runs/temper/link").symlink_to(self.project / "src")
        self.assert_denied(self.run_guard({"tool_name": "Write", "tool_input": {"file_path": "runs/temper/link/x.py"}}),
                           "outside")
        os.rename(self.project / "runs/temper", self.project / "runs/old")
        (self.project / "runs/temper").symlink_to(self.project / "src")
        self.assert_denied(self.run_guard({"tool_name": "Write", "tool_input": {"file_path": "runs/temper/x.py"}}),
                           "resolves somewhere else")

    def test_git_state_and_config(self):
        for cmd, needle in (("git push origin main", "git push"), ("git commit -m x", "git commit"),
                            ("git -C . checkout -b x", "git checkout"), ("git -c core.pager=sh log", "-c"),
                            ("GIT_PAGER=sh git log", "GIT_PAGER"), ("git log --output=src/x", "outside"),
                            ("git branch -D main", "git branch")):
            with self.subTest(cmd=cmd):
                self.assert_denied(self.bash(cmd), needle)

    def test_shell_writes_outside(self):
        for cmd in ("ls\nrm -rf src", "echo x > src/a.py", "echo x >> ../a", "cat a | tee src/b",
                    "cp runs/temper/a src/", "mv src/a runs/temper/a", "rm -rf src", "touch docs/x",
                    "(cd runs/temper) ; echo x > a", "ln -s /etc runs/../escape", "echo x > ~/a"):
            with self.subTest(cmd=cmd):
                self.assert_denied(self.bash(cmd), "outside")

    def test_network_install_escalation(self):
        for cmd in ("curl https://example.com", "/usr/bin/wget x", "gh pr create", "pip install numpy",
                    "python3 -m pip install numpy", "python3 -W ignore -m pip install x", "sudo ls",
                    "ollama pull qwen2.5-coder:7b", "npm i left-pad"):
            with self.subTest(cmd=cmd):
                self.assert_denied(self.bash(cmd))

    def test_inline_code(self):
        for cmd in ("python3 -c 'print(1)'", "python3 -Ic 'x'", "python3", "cat s.py | python3", "bash -c 'ls'",
                    "sh", "cat x | sh", "echo $(git rev-parse HEAD)", "echo `id`", "python3 - <<EOF\nprint(1)\nEOF",
                    "awk '{print}' f", "sed -i s/a/b/ src/x", "find . -delete", "xargs rm < list", "eval ls",
                    "node -e 1", "julia -e 'using Pkg'", "echo x > $HOME/a"):
            with self.subTest(cmd=cmd):
                self.assert_denied(self.bash(cmd))

    def test_kgirl_benches_need_explicit_outputs(self):
        for cmd, needle in (
            ("PYTHONPATH=src python3 -m kgirl.neutronics fit", "--out"),
            ("PYTHONPATH=src python3 -m kgirl.tauls train --preset small", "--out"),
            ("PYTHONPATH=src python3 -m kgirl.iwme plot", "-o"),
            ("PYTHONPATH=src python3 -m kgirl.evolve assay --plot docs/evolve/figures/x.png", "outside"),
            ("PYTHONPATH=src python3 -m kgirl.iwme verify --latex research/iwme/verification_table.tex", "outside"),
        ):
            with self.subTest(cmd=cmd):
                self.assert_denied(self.bash(cmd), needle)

    def test_harness_state_and_authority(self):
        for cmd, needle in (
            ("python3 -m kgirl.harness soup promote 12", "user's decision"),
            ("python3 -m kgirl.harness --home runs/temper/a/h soup promote 12", "user's decision"),
            ("python3 -m kgirl.harness soup curate", "real KGIRL_HOME"),
            ("python3 -m kgirl.harness index .", "real KGIRL_HOME"),
            ("python3 -m kgirl.harness --home runs/temper/a/h task g --repo . --apply", "--apply"),
            ("KGIRL_HOME=runs/temper/a/h python3 -m kgirl.harness task g --repo . --verify 'curl x'", "curl"),
            ("python3 -m kgirl.harness --home runs/temper/a/h index 9x25dillon/numbskull", "network"),
            ("python3 -m kgirl.harness mcp", "MCP"),
            ("python3 -m kgirl.harness ask 'what is x'", "scout"),
            ("python3 -m kgirl.harness --home runs/temper/a/h skills forge --export .claude/skills", "outside"),
            ("python3 -m kgirl.harness --home runs/temper/a/h report -o docs/x.md", "outside"),
            ("python3 -m kgirl.harness soup export-sft docs/sft.jsonl", "outside"),
            ("python3 src/kgirl/harness/cli.py soup promote 1", "-m kgirl.harness"),
        ):
            with self.subTest(cmd=cmd):
                self.assert_denied(self.bash(cmd), needle)

    def test_bad_input_fails_closed(self):
        self.assert_denied(self.run_guard({}, raw="not json"), "guard error")
        self.assert_denied(self.run_guard({}, raw="[1, 2]"), "not a JSON object")
        self.assert_denied(self.run_guard({"tool_name": "Bash", "tool_input": "ls"}), "guard error")
        self.assert_denied(self.run_guard({"tool_name": "Bash", "tool_input": {"command": "echo 'unterminated"}}),
                           "guard error")


class AgentFile(unittest.TestCase):
    """The agent file wires the guard as documented: every tool, fail closed, project-relative path."""

    def test_frontmatter(self):
        text = (GUARD.parents[1] / "agents" / "temper.md").read_text(encoding="utf-8")
        head = text.split("---")[1]
        for line in ('name: temper', 'tools: Read, Grep, Glob, Bash, Write', 'matcher: "*"',
                     'onFailure: block', 'args: ["${CLAUDE_PROJECT_DIR}/.claude/hooks/temper_guard.py"]'):
            self.assertIn(line, head)


if __name__ == "__main__":
    unittest.main()
