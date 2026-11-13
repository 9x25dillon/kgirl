"""The only module of this package that touches the KGIRL host.

Everything the experiment takes from KGIRL comes through here: the cascade generators, the two-point
attention memory, the repaired IWME core and the deterministic chaos generator. The host is read, never
modified, and nothing in the host imports this package.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[1]

if importlib.util.find_spec("kgirl") is None:      # an installed kgirl wins; otherwise use the checkout
    sys.path.insert(0, str(REPO_ROOT / "src"))

from kgirl.evolve.chaos import ChaosRNG                      # noqa: E402
from kgirl.iwme.attention import two_point_memory            # noqa: E402
from kgirl.iwme.cascade import EMACascade, MarkovCascade     # noqa: E402
from kgirl.iwme.core import IWMECore                         # noqa: E402

__all__ = ["ChaosRNG", "EMACascade", "IWMECore", "MarkovCascade", "two_point_memory",
           "host_commit", "host_source_untouched", "PACKAGE_ROOT", "REPO_ROOT"]


def _git(*args: str) -> str:
    try:
        done = subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True, timeout=20)
        return done.stdout.strip() if done.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def host_commit() -> str:
    """Commit of the checkout the experiment ran in, for provenance."""
    return _git("rev-parse", "HEAD") or "unknown"


def host_source_untouched() -> bool:
    """True when the working tree has no modification under src/ relative to its commit."""
    return _git("status", "--porcelain", "--", "src") == "" and host_commit() != "unknown"
