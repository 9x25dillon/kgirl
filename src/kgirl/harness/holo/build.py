"""The offline builder: derive the holo store of a harness home from the chunk index it already has.

    build(home)   $KGIRL_HOME/chunks/all/all-repos.db + all-repos.kgvm  ──▶  $KGIRL_HOME/holo.db
    status(home)  what the store holds, whether every ledger verifies, what it was last built from

No model is called and nothing is embedded: `kgirl chunks` made the vectors, this reads them. The same
index and the same chaos seed give the same digests and the same ledger heads.
"""

from __future__ import annotations

import os
from pathlib import Path

from ..util import harness_home
from .infer import MEMORY_NAME
from .store import DB_NAME, PER_CELL, SEED, HoloStore

TOPIC_BUILT = "holo.built"
_REPORTED = ("entities", "chunks", "orphans", "sampled", "records", "unchanged", "overlaps", "bytes", "per_cell",
             "seed")


def locate(home: Path, chunks_db: str | os.PathLike | None = None, matrix: str | os.PathLike | None = None
           ) -> tuple[Path, Path]:
    """The chunk index and its matrix: the arguments, else KGIRL_CHUNKS_DB / KGIRL_MATRIX, else the home's own."""
    out = home / "chunks" / "all"
    db = Path(chunks_db or os.environ.get("KGIRL_CHUNKS_DB") or out / "all-repos.db")
    kgvm = Path(matrix or os.environ.get("KGIRL_MATRIX") or out / "all-repos.kgvm")
    for path, what in ((db, "chunk index"), (kgvm, "vector matrix")):
        if not path.is_file():
            raise FileNotFoundError(f"no {what} at {path}; `python -m kgirl.harness chunks` builds both")
    return db, kgvm


def build(home: str | os.PathLike | None = None, *, chunks_db: str | os.PathLike | None = None,
          matrix: str | os.PathLike | None = None, per_cell: int | None = PER_CELL, seed: float = SEED,
          use_soup: bool = True, bus=None) -> dict:
    """Build (or refresh) the holo store of `home`. Returns the report: counts, overlaps, digests, bytes."""
    home = Path(home) if home else harness_home()
    db, kgvm = locate(home, chunks_db, matrix)
    atlas = soup = None
    try:
        if (home / "atlas.db").is_file():                    # names and heads of the indexed repositories
            from ..atlas import Atlas
            atlas = Atlas(home / "atlas.db")
        if use_soup and (home / "soup.db").is_file():
            from ..soup import Soup
            soup = Soup(home / "soup.db")
        with HoloStore(home) as store:
            report = store.build_from_index(atlas, db, kgvm, soup=soup, per_cell=per_cell, seed=seed)
    finally:
        if atlas is not None:
            atlas.close()
        if soup is not None:
            soup.close()
    if bus is not None:
        bus.publish(TOPIC_BUILT, {key: report[key] for key in _REPORTED}, "holo")
    return report


def status(home: str | os.PathLike | None = None) -> dict:
    """The store's state without creating it: entities, storage, chain verification, the last build."""
    home = Path(home) if home else harness_home()
    path = home / DB_NAME
    memory = (home / MEMORY_NAME).is_file()
    if not path.is_file():
        return {"db": str(path), "built": None, "entities": 0, "by_type": {},
                "storage": {"matrices": 0, "raw_bytes": 0, "stored_bytes": 0}, "overlaps": 0, "chains_ok": True,
                "broken": [], "memory": memory}
    with HoloStore(home) as store:
        return {**store.status(), "memory": memory}
