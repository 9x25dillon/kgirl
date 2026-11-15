"""HoloStore: small, content-addressed, overlapping matrix databases over the vectors kgirl already has.

The store is a `vessel.matrixdb.MatrixDB` at `$KGIRL_HOME/holo.db`. Nothing here is a new formula and no
vector is embedded here: the build reads the KGVM matrix `kgirl chunks` wrote and derives, per entity,

    centroid   (dim,) float32   mean of the entity's KGVM int8 vectors, cast to float32
    spread     (dim,) float32   their standard deviation per dimension
    chunk      (dim,) int8      one chunk's KGVM vector, byte for byte (label = chunk id)
    evidence   (n, 4) float32   Soup entities only: support, wins, losses, uses of each active fragment

Entities are regions of the vector space: `repo:<name>`, `kind:<symbol kind>`, `soup:<fragment kind>`,
and `holo`, the store itself, whose ledger is the build log.

Overlap: a chunk belongs to one repository and one symbol kind, so its vector is recorded under both
entities. MatrixDB addresses matrices by content, so both records point at one digest: one stored matrix,
two owners. `owners(digest)` and `overlaps()` read that back.

Provenance: every record written here is followed by a `holo.record` event on the entity's ledger, which
MatrixDB chains with `chain_hash(prev, entity, seq, event, payload)`. Payloads carry no timestamps and no
paths (the root entity's build event excepted), so the same index and the same chaos seed give the same
digests and the same ledger heads on any machine.

Randomness: a (repository, kind) cell larger than `per_cell` keeps a sample, drawn with `ChaosRNG`.
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ...evolve.chaos import ChaosRNG
from ...vessel.matrixdb import MatrixDB
from ..atlas.vectors import Matrix
from ..util import harness_home

DB_NAME = "holo.db"
ROOT = "holo"                           # the store's own entity
SEED = 0.4123
PER_CELL = 16                           # chunk vectors kept per (repository, kind) cell; None keeps all
CENTROID, SPREAD, CHUNK, EVIDENCE = "centroid", "spread", "chunk", "evidence"
E_RECORD, E_BUILT = "holo.record", "holo.built"
SPACE_TYPES = ("repo", "kind")          # entities that own chunk vectors
_ALL = 10 ** 9


class HoloError(RuntimeError):
    """The store cannot do what was asked; the message says what to do about it."""


@dataclass(frozen=True)
class Record:
    entity: str
    kind: str
    label: str
    digest: str
    geometry: int            # row id of the MatrixDB geometry
    seq: int | None          # ledger sequence of its provenance event; None when nothing new was written
    hash: str                # the entity's ledger hash after this record
    new: bool


def repo_entity(name: str) -> str:
    return f"repo:{name}"


def kind_entity(kind: str | None) -> str:
    return f"kind:{kind or 'other'}"


def soup_entity(kind: str) -> str:
    return f"soup:{kind}"


def entity_type(name: str) -> str:
    return name.split(":", 1)[0]


class HoloStore:
    def __init__(self, home: str | os.PathLike | None = None):
        self.home = Path(home) if home else harness_home()
        self.path = self.home / DB_NAME
        self.db = MatrixDB(self.path)
        # MatrixDB has no "who references this digest" query. A second, read-only connection asks its
        # documented tables directly; every write still goes through MatrixDB.
        self._ro = sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True, check_same_thread=False)
        self.cache: dict[str, Any] = {}          # opened matrix, chunk index and centroids, kept by infer.py

    def drop_cache(self) -> None:
        """Forget what inference opened (a rebuild may have replaced the files behind it)."""
        chunks = self.cache.pop("chunks", None)
        if chunks is not None:
            chunks.close()
        self.cache.clear()

    def close(self) -> None:
        self.drop_cache()
        self._ro.close()
        self.db.close()

    def __enter__(self) -> "HoloStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------ entities and records

    def ensure(self, name: str, genome: dict | None = None) -> bool:
        """Create the entity unless it exists. True when it was created."""
        if self.db.has_entity(name):
            return False
        self.db.create_entity(name, {"holo": 1, "type": entity_type(name), **(genome or {})})
        return True

    def entities(self, *types: str) -> list[str]:
        """Entity names in creation order, the root excluded; optionally only those of the given types."""
        return [n for n in self.db.entities() if n != ROOT and (not types or entity_type(n) in types)]

    def vocabulary(self) -> list[str]:
        """The entities that own chunk vectors, in a fixed order: what a HoloMemory's tokens stand for."""
        return self.entities(*SPACE_TYPES)

    def record(self, entity: str, kind: str, M: np.ndarray, *, label: str = "", meta: dict | None = None) -> Record:
        """Record a matrix under an entity, with its ledger entry. Unchanged content writes nothing."""
        M = np.ascontiguousarray(M)
        digest = self.db.put_matrix(M)
        last = self.db.latest(entity, kind, "", label)
        if last is not None and last.matrix == digest:
            return Record(entity, kind, label, digest, last.id, None, self.db.head(entity), False)
        gid = self.db.record(entity, kind, M, label=label, meta=meta)
        event = self.db.append(entity, E_RECORD, {"kind": kind, "label": label, "matrix": digest,
                                                  "dtype": str(M.dtype), "shape": list(M.shape)})
        return Record(entity, kind, label, digest, gid, event.seq, event.hash, True)

    def digest_of(self, entity: str, kind: str, label: str = "") -> str | None:
        """The digest of the entity's current matrix of this kind and label, or None."""
        if not self.db.has_entity(entity):
            return None
        g = self.db.latest(entity, kind, "", label)
        return g.matrix if g else None

    def matrix(self, entity: str, kind: str, label: str = "") -> np.ndarray | None:
        digest = self.digest_of(entity, kind, label)
        return self.db.get_matrix(digest) if digest else None

    # ------------------------------------------------------------------ overlap and provenance

    def owners(self, digest: str) -> list[str]:
        """Every entity that has recorded this matrix."""
        rows = self._ro.execute("SELECT DISTINCT e.name FROM geometries g JOIN entities e ON e.id = g.entity"
                                " WHERE g.matrix = ? ORDER BY e.name", (digest,))
        return [name for (name,) in rows]

    def overlaps(self, kind: str = CHUNK) -> dict[str, list[str]]:
        """Digests of this kind recorded under two or more entities, with their owners."""
        shared: dict[str, list[str]] = {}
        for digest, name in self._ro.execute(
                "SELECT DISTINCT g.matrix, e.name FROM geometries g JOIN entities e ON e.id = g.entity"
                " WHERE g.kind = ? ORDER BY g.matrix, e.name", (kind,)):
            shared.setdefault(digest, []).append(name)
        return {digest: names for digest, names in shared.items() if len(names) >= 2}

    def verify(self) -> dict[str, dict]:
        """Recompute every entity's hash chain, and list matrices recorded without a ledger entry."""
        out: dict[str, dict] = {}
        for name in self.db.entities():
            chain = self.db.verify(name)
            proven = {e.payload.get("matrix") for e in self.db.events(name, E_RECORD, limit=_ALL)}
            rows = self._ro.execute("SELECT DISTINCT g.matrix FROM geometries g JOIN entities e ON e.id = g.entity"
                                    " WHERE e.name = ?", (name,))
            unproven = sorted(digest for (digest,) in rows if digest not in proven)
            out[name] = {**chain, "unproven": unproven, "ok": bool(chain["ok"]) and not unproven}
        return out

    def source(self) -> dict | None:
        """The newest build's record on the root ledger (paths of the index it was built from), or None."""
        if not self.db.has_entity(ROOT):
            return None
        events = self.db.events(ROOT, E_BUILT, limit=_ALL)
        return events[-1].payload if events else None

    def status(self) -> dict:
        names = self.entities()
        by_type: dict[str, int] = {}
        for name in names:
            by_type[entity_type(name)] = by_type.get(entity_type(name), 0) + 1
        checks = self.verify()
        return {"db": str(self.path), "built": self.source(), "entities": len(names), "by_type": by_type,
                "storage": self.db.storage(), "overlaps": len(self.overlaps()),
                "chains_ok": all(c["ok"] for c in checks.values()),
                "broken": sorted(n for n, c in checks.items() if not c["ok"])}

    # ------------------------------------------------------------------ build

    def build_from_index(self, atlas, chunks_db: str | os.PathLike, matrix_path: str | os.PathLike, *, soup=None,
                         per_cell: int | None = PER_CELL, seed: float = SEED) -> dict:
        """Derive the store from an existing chunk index and its KGVM matrix. No embedding, no model call.

        `atlas` (or None) names the indexed repositories and their heads; `soup` (or None) adds one evidence
        table per fragment kind. Returns the build report: counts, overlaps, digests, ledger heads, bytes.
        """
        matrix = Matrix(matrix_path)
        chunk_of = _chunk_owners(chunks_db)
        ids = np.asarray(matrix.ids)
        heads = {r["name"]: r["head"] for r in atlas.repos()} if atlas is not None else {}

        members: dict[str, list[int]] = {}                  # entity -> matrix rows, ascending
        cells: dict[tuple[str, str], list[int]] = {}        # (repo, kind) -> matrix rows, ascending
        orphans = 0
        for row, chunk_id in enumerate(ids.tolist()):
            where = chunk_of.get(chunk_id)
            if where is None:                               # a vector whose chunk is not in this index
                orphans += 1
                continue
            repo, kind = where
            members.setdefault(repo_entity(repo), []).append(row)
            members.setdefault(kind_entity(kind), []).append(row)
            cells.setdefault((repo, kind), []).append(row)

        self.ensure(ROOT)
        names = sorted(set(members) | {repo_entity(name) for name in heads})
        for name in names:
            self.ensure(name, {"name": name.split(":", 1)[1]})

        changed: dict[str, int] = {}                        # entity -> records written in this build
        unchanged = 0
        digests: dict[str, dict[str, str]] = {name: {} for name in names}

        def write(entity: str, kind: str, M: np.ndarray, label: str = "", meta: dict | None = None) -> Record:
            nonlocal unchanged
            rec = self.record(entity, kind, M, label=label, meta=meta)
            if rec.new:
                changed[entity] = changed.get(entity, 0) + 1
            else:
                unchanged += 1
            return rec

        for name in names:
            rows = members.get(name)
            if not rows:
                continue                                    # an indexed repository without vectors yet
            vecs = np.asarray(matrix.vals[np.asarray(rows)], dtype=np.float32)
            meta = {"chunks": len(rows), "model": matrix.model, "scale": float(matrix.scale)}
            digests[name][CENTROID] = write(name, CENTROID, vecs.mean(axis=0), meta=meta).digest
            digests[name][SPREAD] = write(name, SPREAD, vecs.std(axis=0), meta=meta).digest

        rng = ChaosRNG(seed)
        sampled: dict[str, set[int]] = {}                   # entity -> chunk ids kept
        shared: dict[str, set[str]] = {}                    # chunk digest -> owners, in this build
        for repo, kind in sorted(cells):
            rows = cells[(repo, kind)]
            if per_cell and len(rows) > per_cell:
                keep = np.argsort(rng.uniform(len(rows)), kind="stable")[:per_cell]
                rows = sorted(rows[i] for i in keep.tolist())
            for row in rows:
                chunk_id = int(ids[row])
                vec = np.array(matrix.vals[row])            # the stored int8 row
                for entity in (repo_entity(repo), kind_entity(kind)):
                    rec = write(entity, CHUNK, vec, label=str(chunk_id),
                                meta={"chunk": chunk_id, "repo": repo, "kind": kind or "other"})
                    shared.setdefault(rec.digest, set()).add(entity)
                    sampled.setdefault(entity, set()).add(chunk_id)

        if soup is not None:
            by_kind: dict[str, list] = {}
            for frag in soup.all(status="active"):
                by_kind.setdefault(frag.kind, []).append(frag)
            for kind in sorted(by_kind):
                name = soup_entity(kind)
                self.ensure(name, {"name": kind})
                table = np.array([[f.support, f.wins, f.losses, f.uses] for f in by_kind[kind]], dtype=np.float32)
                digests[name] = {EVIDENCE: write(name, EVIDENCE, table, meta={"fragments": len(table)}).digest}

        for name in sorted(changed):                        # what this build asserted about each changed entity
            payload: dict[str, Any] = {"records": changed[name], "seed": seed}
            if entity_type(name) in SPACE_TYPES:
                payload.update(chunks=len(members.get(name, ())), sampled=len(sampled.get(name, ())))
            if entity_type(name) == "repo":
                payload["head"] = heads.get(name.split(":", 1)[1])
            self.db.append(name, E_BUILT, payload)

        everyone = self.entities()
        by_type: dict[str, int] = {}
        for name in everyone:
            by_type[entity_type(name)] = by_type.get(entity_type(name), 0) + 1
        overlaps = sum(len(owners) >= 2 for owners in shared.values())
        report = {
            "db": str(self.path),
            "matrix": {"path": str(Path(matrix_path).resolve()), "count": int(matrix.count), "dim": int(matrix.dim),
                       "model": matrix.model, "scale": float(matrix.scale)},
            "chunks_db": str(Path(chunks_db).resolve()),
            "entities": by_type, "chunks": int(matrix.count) - orphans, "orphans": orphans,
            "sampled": len(set().union(*sampled.values())) if sampled else 0,
            "records": sum(changed.values()), "unchanged": unchanged, "overlaps": overlaps,
            "per_cell": per_cell, "seed": seed}
        self.db.append(ROOT, E_BUILT, {**report, "bytes": self.db.storage()})
        report["digests"] = digests
        report["heads"] = {name: self.db.head(name) for name in everyone}
        report["bytes"] = self.db.storage()
        self.drop_cache()                                   # inference reopens what this build may have replaced
        return report


def _chunk_owners(chunks_db: str | os.PathLike) -> dict[int, tuple[str, str]]:
    """chunk id -> (repo, symbol kind) from a chunk index, opened read-only."""
    con = sqlite3.connect(Path(chunks_db).resolve().as_uri() + "?mode=ro", uri=True)
    try:
        return {int(i): (repo, kind or "other") for i, repo, kind in con.execute("SELECT id, repo, kind FROM chunks")}
    finally:
        con.close()
