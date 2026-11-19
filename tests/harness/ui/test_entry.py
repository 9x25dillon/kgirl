"""The `kgirl ui` entry point without textual: the install hint and exit 2, the workspace rule, and that
importing the package never imports textual. Written, not run.
"""

import argparse
import io
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import tests.harness.ui  # noqa: F401  (sys.path)
from kgirl.harness import ui
from tests.harness.ui import SRC


class Entry(unittest.TestCase):
    def test_missing_textual_prints_the_hint_and_exits_2(self):
        err = io.StringIO()
        with mock.patch.object(ui, "available", return_value=False), redirect_stderr(err):
            rc = ui.run(object(), SimpleNamespace(dir=None, no_workspace=True, yes=False, new=False, conv=None))
        self.assertEqual(rc, 2)
        self.assertIn("pip install textual", err.getvalue())

    def test_package_import_does_not_import_textual(self):
        code = ("import sys; import kgirl.harness.ui, kgirl.harness.ui.backend; "
                "print(any(m == 'textual' or m.startswith('textual.') for m in sys.modules))")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60,
                             env={"PYTHONPATH": str(SRC), "PATH": "/usr/bin:/bin"})
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.strip(), "False")

    def test_workspace_root_rule(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(dir=tmp, no_workspace=False)
            self.assertEqual(ui.workspace_root(args), Path(tmp))
            self.assertIsNone(ui.workspace_root(SimpleNamespace(dir=tmp, no_workspace=True)))
        self.assertIsNone(ui.workspace_root(SimpleNamespace(dir=str(Path.home()), no_workspace=False)))
        self.assertIsNone(ui.workspace_root(SimpleNamespace(dir="/", no_workspace=False)))


class CliWiring(unittest.TestCase):
    def parse(self, *argv):
        from kgirl.harness import cli_mind
        ap = argparse.ArgumentParser()
        cli_mind.add_parsers(ap.add_subparsers(dest="cmd", required=True))
        return cli_mind, ap.parse_args(list(argv))

    def test_ui_verb_parses_its_flags(self):
        _, args = self.parse("ui", "--new", "--conv", "3", "--dir", "/tmp", "--yes")
        self.assertEqual((args.cmd, args.new, args.conv, args.dir, args.yes, args.no_workspace),
                         ("ui", True, 3, "/tmp", True, False))

    def test_ui_verb_dispatches_to_the_ui_package(self):
        cli_mind, args = self.parse("ui", "--no-workspace")
        with mock.patch.object(ui, "run", return_value=0) as run:
            self.assertEqual(cli_mind.run("assistant", args), 0)
        run.assert_called_once_with("assistant", args)


if __name__ == "__main__":
    unittest.main()
