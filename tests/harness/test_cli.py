import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from kgirl.harness.cli import main


class SoupCliTests(unittest.TestCase):
    def soup(self, home, *args):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(main(["--home", str(home), "soup", *args]), 0)
        return buf.getvalue()

    def test_recall_prints_what_add_stored(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            added = self.soup(home, "add", "prefer pathlib over os.path for file handling", "--kind", "preference")
            self.assertRegex(added, r"^#\d+")
            got = self.soup(home, "recall", "pathlib file handling")
            self.assertIn("prefer pathlib over os.path", got)
            self.assertRegex(got, r"\[\d+/600 tokens\]")


if __name__ == "__main__":
    unittest.main()
