"""Register QINCRS instances in the Atlas without touching anything else.

QINCRS exists as several copies and forks scattered across repos. This
registers each *distinct* file (grouped by content hash) as its own tiny
Atlas repo, `qincrs-<sha8>`, so search / blast / clone edges can see it.

Rules that keep it out of numbskull's way
  * any path containing `numbskull` is refused, and a QINCRS repo can never
    be named after one;
  * only the named files are copied (into $KGIRL_HOME/qincrs_instances/<name>/,
    so `read_source` keeps working) — no directory is walked, so no sibling
    file from a host repo is indexed;
  * nothing is executed or imported; the originals are only read.
Every original path of an identical copy is kept in the repo's `origin`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..util import harness_home, sha1
from .store import Atlas

BLOCKED = ("numbskull",)
MAX_BYTES = 2_000_000


@dataclass(frozen=True)
class Instance:
    repo: str
    filename: str
    paths: tuple[str, ...]


def is_qincrs_name(name: str) -> bool:
    return "qincrs" in name.lower()


def register_qincrs(atlas: Atlas, paths: list[str | Path], stage_dir: Path | None = None) -> list[Instance]:
    """Index each distinct QINCRS file as repo `qincrs-<sha8>`. Returns what was registered."""
    stage = Path(stage_dir) if stage_dir else harness_home() / "qincrs_instances"
    groups: dict[str, tuple[str, bytes, list[str]]] = {}
    for raw in paths:
        p = Path(raw).expanduser()
        full = str(p.resolve()) if p.exists() else str(p)
        if any(b in full.lower() for b in BLOCKED):
            raise ValueError(f"refusing path inside a protected repo: {full}")
        if not p.is_file() or p.is_symlink():
            raise FileNotFoundError(f"not a regular file: {full}")
        if not is_qincrs_name(p.name):
            raise ValueError(f"not a QINCRS file name: {p.name}")
        if p.stat().st_size > MAX_BYTES:
            raise ValueError(f"file too large to index: {full}")
        data = p.read_bytes()
        digest = sha1(data)
        entry = groups.setdefault(digest, (p.name, data, []))
        entry[2].append(full)
    out: list[Instance] = []
    for digest, (filename, data, where) in sorted(groups.items()):
        name = f"qincrs-{digest[:8]}"
        root = stage / name
        root.mkdir(parents=True, exist_ok=True)
        for old in root.iterdir():
            if old.is_file() and old.name != filename:
                old.unlink()
        (root / filename).write_bytes(data)
        atlas.ingest(root, name=name, origin="; ".join(sorted(where))[:1000])
        out.append(Instance(name, filename, tuple(sorted(where))))
    return out

