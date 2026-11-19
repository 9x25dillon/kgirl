"""kgirl ui: the split-screen terminal front end (chat | status, jobs and code search).

It drives the same in-process Console the web console does (`console/core.py`): one Assistant, one
workspace, a Mind per conversation, long work as jobs, approvals as events. Nothing goes over HTTP.

`textual` is optional. It is imported only inside this package's `app` module, so the rest of kgirl never
needs it; without it `kgirl ui` prints how to install it and exits 2.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

INSTALL_HINT = ("kgirl ui needs the optional `textual` package (a terminal UI library).\n"
                "  install it into the environment that runs kgirl:  python3 -m pip install textual\n"
                "  the web console needs nothing extra:              kgirl console")


def available() -> bool:
    return importlib.util.find_spec("textual") is not None


def workspace_root(args) -> Path | None:
    """--dir, else the current directory, unless --no-workspace or that is the home folder or the
    filesystem root (too broad to hand to an agent). The same rule as `kgirl console`."""
    if getattr(args, "no_workspace", False):
        return None
    cand = Path(args.dir).expanduser() if getattr(args, "dir", None) else Path.cwd()
    if cand.resolve() in (Path.home().resolve(), Path(cand.anchor)):
        return None
    return cand


def run(a, args) -> int:
    if not available():
        print(INSTALL_HINT, file=sys.stderr)
        return 2
    from .app import KgirlUI
    from .backend import HarnessBackend
    try:
        backend = HarnessBackend(a, workspace_root=workspace_root(args), auto_edit=bool(getattr(args, "yes", False)))
    except ValueError as exc:            # WorkspaceError: --dir is not a directory
        print(f"kgirl ui: {exc}", file=sys.stderr)
        return 2
    try:
        KgirlUI(backend, new=bool(getattr(args, "new", False)), conv=getattr(args, "conv", None)).run()
    finally:
        backend.close()
    return 0


__all__ = ["INSTALL_HINT", "available", "run", "workspace_root"]
