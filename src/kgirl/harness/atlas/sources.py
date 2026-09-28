"""Where repositories come from: local directories or shallow git clones.

A source spec is one of
    /abs/or/relative/path           -> indexed in place
    owner/repo                      -> https://github.com/owner/repo (shallow clone into cache)
    https://… or git@…              -> shallow clone into cache
Clones are refreshed with `git fetch --depth 1` + hard reset of the cache copy
only; user working trees are never modified.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ..util import harness_home

_OWNER_REPO = re.compile(r"^[\w.-]+/[\w.-]+$")


@dataclass(frozen=True)
class Source:
    name: str
    root: Path
    origin: str


def resolve_source(spec: str, cache_dir: Path | None = None, refresh: bool = False) -> Source:
    p = Path(spec).expanduser()
    if p.exists():
        return Source(p.resolve().name, p.resolve(), _origin(p))
    if _OWNER_REPO.match(spec):
        url = f"https://github.com/{spec}.git"
    elif spec.startswith(("https://", "http://", "git@", "ssh://")):
        url = spec
    else:
        raise FileNotFoundError(f"not a directory or git remote: {spec}")
    name = url.rstrip("/").rsplit("/", 1)[-1].removesuffix(".git")
    dest = (cache_dir or harness_home() / "repos") / name
    if not (dest / ".git").exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        _git(["clone", "--quiet", "--depth", "1", url, str(dest)], timeout=900)
    elif refresh:
        _git(["-C", str(dest), "fetch", "--quiet", "--depth", "1", "origin"], timeout=600)
        _git(["-C", str(dest), "reset", "--quiet", "--hard", "FETCH_HEAD"], timeout=60)
    return Source(name, dest, url)


def _git(args: list[str], timeout: int) -> str:
    out = subprocess.run(["git", *args], capture_output=True, text=True, timeout=timeout)
    if out.returncode != 0:
        raise RuntimeError(f"git {' '.join(args[:3])} failed: {out.stderr.strip()[:300]}")
    return out.stdout


def _origin(p: Path) -> str:
    try:
        return _git(["-C", str(p), "config", "--get", "remote.origin.url"], timeout=5).strip()
    except (RuntimeError, OSError, subprocess.TimeoutExpired):
        return ""
