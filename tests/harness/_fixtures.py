"""Tiny on-disk repos shared by the harness tests."""

from __future__ import annotations

import sys
import tempfile
import textwrap
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

CLONED = '''
def normalize(values):
    total = sum(values)
    if total == 0:
        return [0.0 for _ in values]
    out = []
    for v in values:
        out.append(v / total)
    return out
'''


def write(root: Path, files: dict[str, str]) -> Path:
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(text).lstrip("\n"), encoding="utf-8")
    return root


def make_repos() -> tuple[tempfile.TemporaryDirectory, Path, Path]:
    tmp = tempfile.TemporaryDirectory()
    base = Path(tmp.name)
    lib = write(base / "lib", {
        "engine.py": textwrap.dedent('''
            """Engine: the numeric core."""

            class Engine:
                """Runs the thing."""
                def run(self, x: int) -> int:
                    return helper(x) * 2

            def helper(x):
                return x + 1
            ''') + CLONED,
        "README.md": "# lib\n\nA numeric library.\n\n## Usage\n\nCall Engine.run.\n",
    })
    hub = write(base / "hub", {
        "app.py": '''
            from engine import Engine

            def main():
                return Engine().run(3)

            if __name__ == "__main__":
                main()
            ''',
        "pkg/__init__.py": "",
        "pkg/other.py": "from app import main\n\ndef twice():\n    return main() * 2\n",
        "pkg/rel.py": "from . import other\n\nX = other.twice\n",
        "util.py": CLONED,
        "broken.py": "def ok():\n    return 1\n\ndef bad(:\n    pass\n",
        "web/main.ts": "import { add } from './math';\nexport function start(): number { return add(1, 2); }\n",
        "web/math.ts": "export function add(a: number, b: number): number { return a + b; }\n",
        "tqdm": "#!/home/x/venv/bin/python3\nimport sys\nfrom tqdm.cli import main\nif __name__ == '__main__':\n"
                "    sys.exit(main())\n",
    })
    return tmp, hub, lib


BUGGY_REPO = {
    "calc.py": '''
        def mean(xs):
            """Arithmetic mean of a non-empty list."""
            return sum(xs) / (len(xs) + 1)
        ''',
    "test_calc.py": '''
        import unittest
        from calc import mean

        class T(unittest.TestCase):
            def test_mean(self):
                self.assertEqual(mean([1, 2, 3]), 2)

        if __name__ == "__main__":
            unittest.main()
        ''',
}
