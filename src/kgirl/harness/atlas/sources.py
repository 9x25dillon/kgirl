"""Where repositories come from: local directories or shallow git clones.

A source spec is one of
    /abs/or/relative/path           -> indexed in place
    owner/repo                      -> https://github.com/owner/repo (shallow clone into cache)
    https://… or git@…              -> shallow clone into cache
Clones are refreshed with `git fetch --depth 1` + hard reset of the cache copy
only; user working trees are never modified.
"""

from __future__ import annotations

import os
import re
import shutil
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
    elif spec.startswith(("https://", "http://", "git@", "ssh://", "file://")):
        url = spec
    else:
        raise FileNotFoundError(f"not a directory or git remote: {spec}")
    name = url.rstrip("/").rsplit("/", 1)[-1].removesuffix(".git")
    dest = (cache_dir or harness_home() / "repos") / name
    if not (dest / ".git").exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        fresh = not dest.exists()
        try:
            _git(["clone", "--quiet", "--depth", "1", url, str(dest)], timeout=900)
        except (RuntimeError, OSError, subprocess.TimeoutExpired):
            # git can leave .git behind when checkout fails; a retry would then index an empty tree.
            if fresh:
                shutil.rmtree(dest, ignore_errors=True)
            raise
    elif refresh:
        _git(["-C", str(dest), "fetch", "--quiet", "--depth", "1", "origin"], timeout=600)
        # "--" pins FETCH_HEAD as a revision even when the clone's tree has a file of that name.
        _git(["-C", str(dest), "reset", "--quiet", "--hard", "FETCH_HEAD", "--"], timeout=60)
    return Source(name, dest, url)


def _git(args: list[str], timeout: int) -> str:
    # The index only reads source text, so LFS pointers are enough and a broken LFS server can't fail a checkout.
    env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}
    out = subprocess.run(["git", *args], capture_output=True, text=True, timeout=timeout, env=env)
    if out.returncode != 0:
        raise RuntimeError(f"git {' '.join(args[:3])} failed: {out.stderr.strip()[:300]}")
    return out.stdout


def _origin(p: Path) -> str:
    try:
        return _git(["-C", str(p), "config", "--get", "remote.origin.url"], timeout=5).strip()
    except (RuntimeError, OSError, subprocess.TimeoutExpired):
        return ""
