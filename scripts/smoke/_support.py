"""Shared isolation and reporting for explicit, offline check-day commands."""

from __future__ import annotations

import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True


@contextmanager
def isolated_home():
    """Ignore inherited model/index settings; put all harness state in scratch space."""
    with tempfile.TemporaryDirectory(prefix="kgirl-check-") as tmp:
        root = Path(tmp)
        with patch.dict(os.environ):
            for key in list(os.environ):
                if key.startswith(("KGIRL_", "OLLAMA_")):
                    os.environ.pop(key)
            os.environ.update({
                "KGIRL_HOME": str(root / "home"),
                "KGIRL_LOCAL_ONLY": "1",
                "KGIRL_LIFE": "0",
                "KGIRL_VERIFY_ALLOWLIST": "",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHON_COLORS": "0",
                "AURIC_PORTER_DB": str(root / "porter-not-configured.sqlite"),
            })
            yield root


class Stages:
    """Report independent failures, and mark dependent work as skipped, never passed."""

    def __init__(self):
        self.results: dict[str, bool] = {}

    def check(self, name, action, *, needs=()):
        missing = [n for n in needs if not self.results.get(n)]
        if missing:
            self.results[name] = False
            print(f"SKIP {name}: blocked by {', '.join(missing)}", flush=True)
            return
        try:
            detail = action()
        except Exception as exc:
            self.results[name] = False
            print(f"FAIL {name}: {type(exc).__name__}: {exc}", flush=True)
        else:
            self.results[name] = True
            print(f"PASS {name}" + (f": {detail}" if detail else ""), flush=True)

    def status(self):
        passed = sum(self.results.values())
        print(f"{passed}/{len(self.results)} stages passed", flush=True)
        return 0 if self.results and all(self.results.values()) else 1


def require(condition, message):
    """Keep checks active even when Python is invoked with -O."""
    if not condition:
        raise AssertionError(message)
