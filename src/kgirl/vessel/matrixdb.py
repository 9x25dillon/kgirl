"""MatrixDB: the vessel's own store of matrices, geometries and history. SQLite, standard library + numpy.

    matrices     content-addressed blobs: sha256 of the .npy bytes (allow_pickle=False), zlib-compressed.
                 Writing the same matrix twice stores it once, so a weight snapshot that did not change
                 costs a row, not a copy.
    entities     one row per vessel (name unique): genome, birth time
    geometries   typed records that point at matrices: (entity, kind, layer, label) → matrix hash + meta.
                 Kinds in use: anchor_gram, gram, subspace, steering_vector, weight, fisher
    ledger       append-only, hash-chained events per entity:
                     h_i = sha256(h_{i−1} ‖ entity ‖ seq ‖ event ‖ canonical-JSON payload)
                 `verify(entity)` recomputes the chain; any edited, deleted or reordered row breaks it.

The chain is tamper-evident, not tamper-proof: someone who can write the file can rewrite the whole chain.
Exporting the head hash elsewhere (the soulpack's `signing.checksum`) is what pins it.

Concurrency: WAL journal, one connection per MatrixDB, writes inside explicit transactions.
"""

from __future__ import annotations

import hashlib
import io
import json
import sqlite3
import threading
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

GENESIS = "0" * 64

SCHEMA = """
CREATE TABLE IF NOT EXISTS matrices (
  hash TEXT PRIMARY KEY, dtype TEXT NOT NULL, shape TEXT NOT NULL, nbytes INTEGER NOT NULL, blob BLOB NOT NULL);
CREATE TABLE IF NOT EXISTS entities (
  id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL, created REAL NOT NULL, genome TEXT NOT NULL DEFAULT '{}');
CREATE TABLE IF NOT EXISTS geometries (
  id INTEGER PRIMARY KEY, entity INTEGER NOT NULL REFERENCES entities(id), kind TEXT NOT NULL,
  layer TEXT NOT NULL DEFAULT '', label TEXT NOT NULL DEFAULT '', matrix TEXT NOT NULL REFERENCES matrices(hash),
  meta TEXT NOT NULL DEFAULT '{}', ts REAL NOT NULL);
CREATE INDEX IF NOT EXISTS ix_geom ON geometries(entity, kind, layer, label, id);
CREATE TABLE IF NOT EXISTS ledger (
  entity INTEGER NOT NULL REFERENCES entities(id), seq INTEGER NOT NULL, ts REAL NOT NULL, event TEXT NOT NULL,
  payload TEXT NOT NULL, prev TEXT NOT NULL, hash TEXT NOT NULL, PRIMARY KEY (entity, seq));
"""


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=_jsonable)


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON-serializable: {type(o).__name__}")


def encode_matrix(M: np.ndarray) -> tuple[str, bytes]:
    buf = io.BytesIO()
    np.save(buf, np.ascontiguousarray(M), allow_pickle=False)
    raw = buf.getvalue()
    return hashlib.sha256(raw).hexdigest(), raw


def decode_matrix(raw: bytes) -> np.ndarray:
    return np.load(io.BytesIO(raw), allow_pickle=False)


def chain_hash(prev: str, entity: str, seq: int, event: str, payload: str) -> str:
    h = hashlib.sha256()
    for part in (prev, entity, str(seq), event, payload):
        h.update(part.encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()


@dataclass(frozen=True)
class Geometry:
    id: int
    entity: str
    kind: str
    layer: str
    label: str
    matrix: str
    meta: dict
    ts: float


@dataclass(frozen=True)
class Event:
    entity: str
    seq: int
    ts: float
    event: str
    payload: dict
    prev: str
    hash: str


class MatrixDB:
    def __init__(self, path: str | Path = ":memory:"):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._lock = threading.RLock()
        self._db.execute("PRAGMA foreign_keys=ON")
        if self.path != ":memory:":
            self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(SCHEMA)

    def close(self) -> None:
        self._db.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------------------------------------- transactions

    def _tx(self):
        return _Tx(self._db, self._lock)

    # ------------------------------------------------------------------------------------------- matrices

    def put_matrix(self, M: np.ndarray) -> str:
        digest, raw = encode_matrix(np.asarray(M))
        with self._tx():
            self._db.execute("INSERT OR IGNORE INTO matrices(hash, dtype, shape, nbytes, blob) VALUES (?,?,?,?,?)",
                             (digest, str(M.dtype), canonical(list(M.shape)), len(raw), zlib.compress(raw, 6)))
        return digest

    def get_matrix(self, digest: str) -> np.ndarray:
        row = self._db.execute("SELECT blob FROM matrices WHERE hash=?", (digest,)).fetchone()
        if row is None:
            raise KeyError(f"no matrix {digest[:12]}")
        raw = zlib.decompress(row[0])
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError(f"matrix {digest[:12]} fails its content hash (corrupted store)")
        return decode_matrix(raw)

    def storage(self) -> dict:
        n, raw, packed = self._db.execute(
            "SELECT COUNT(*), COALESCE(SUM(nbytes),0), COALESCE(SUM(LENGTH(blob)),0) FROM matrices").fetchone()
        return {"matrices": n, "raw_bytes": raw, "stored_bytes": packed}

    # ------------------------------------------------------------------------------------------- entities

    def create_entity(self, name: str, genome: Optional[dict] = None) -> int:
        with self._tx():
            cur = self._db.execute("INSERT INTO entities(name, created, genome) VALUES (?,?,?)",
                                   (name, time.time(), canonical(genome or {})))
        return int(cur.lastrowid)

    def entity_id(self, name: str) -> int:
        row = self._db.execute("SELECT id FROM entities WHERE name=?", (name,)).fetchone()
        if row is None:
            raise KeyError(f"no vessel named {name!r}")
        return int(row[0])

    def has_entity(self, name: str) -> bool:
        return self._db.execute("SELECT 1 FROM entities WHERE name=?", (name,)).fetchone() is not None

    def genome(self, name: str) -> dict:
        return json.loads(self._db.execute("SELECT genome FROM entities WHERE name=?", (name,)).fetchone()[0])

    def entities(self) -> list[str]:
        return [r[0] for r in self._db.execute("SELECT name FROM entities ORDER BY id")]

    # ------------------------------------------------------------------------------------------- geometries

    def record(self, entity: str, kind: str, M: np.ndarray, *, layer: str = "", label: str = "",
               meta: Optional[dict] = None) -> int:
        eid = self.entity_id(entity)
        digest = self.put_matrix(M)
        with self._tx():
            cur = self._db.execute(
                "INSERT INTO geometries(entity, kind, layer, label, matrix, meta, ts) VALUES (?,?,?,?,?,?,?)",
                (eid, kind, layer, label, digest, canonical(meta or {}), time.time()))
        return int(cur.lastrowid)

    def _geometry(self, row, entity: str) -> Geometry:
        return Geometry(row[0], entity, row[1], row[2], row[3], row[4], json.loads(row[5]), row[6])

    def latest(self, entity: str, kind: str, layer: str = "", label: Optional[str] = None) -> Optional[Geometry]:
        eid = self.entity_id(entity)
        q, args = "SELECT id, kind, layer, label, matrix, meta, ts FROM geometries WHERE entity=? AND kind=? AND layer=?", \
            [eid, kind, layer]
        if label is not None:
            q += " AND label=?"
            args.append(label)
        row = self._db.execute(q + " ORDER BY id DESC LIMIT 1", args).fetchone()
        return self._geometry(row, entity) if row else None

    def history(self, entity: str, kind: str, layer: Optional[str] = None, limit: int = 1000) -> list[Geometry]:
        eid = self.entity_id(entity)
        q, args = "SELECT id, kind, layer, label, matrix, meta, ts FROM geometries WHERE entity=? AND kind=?", [eid, kind]
        if layer is not None:
            q += " AND layer=?"
            args.append(layer)
        rows = self._db.execute(q + " ORDER BY id DESC LIMIT ?", args + [limit]).fetchall()
        return [self._geometry(r, entity) for r in reversed(rows)]

    def load(self, g: Geometry) -> np.ndarray:
        return self.get_matrix(g.matrix)

    # ------------------------------------------------------------------------------------------- ledger

    def append(self, entity: str, event: str, payload: Optional[dict] = None) -> Event:
        eid = self.entity_id(entity)
        body = canonical(payload or {})
        with self._tx():
            last = self._db.execute("SELECT seq, hash FROM ledger WHERE entity=? ORDER BY seq DESC LIMIT 1",
                                    (eid,)).fetchone()
            seq, prev = (last[0] + 1, last[1]) if last else (0, GENESIS)
            h = chain_hash(prev, entity, seq, event, body)
            ts = time.time()
            self._db.execute("INSERT INTO ledger(entity, seq, ts, event, payload, prev, hash) VALUES (?,?,?,?,?,?,?)",
                             (eid, seq, ts, event, body, prev, h))
        return Event(entity, seq, ts, event, json.loads(body), prev, h)

    def events(self, entity: str, event: Optional[str] = None, limit: int = 10_000) -> list[Event]:
        eid = self.entity_id(entity)
        q, args = "SELECT seq, ts, event, payload, prev, hash FROM ledger WHERE entity=?", [eid]
        if event is not None:
            q += " AND event=?"
            args.append(event)
        rows = self._db.execute(q + " ORDER BY seq LIMIT ?", args + [limit]).fetchall()
        return [Event(entity, r[0], r[1], r[2], json.loads(r[3]), r[4], r[5]) for r in rows]

    def head(self, entity: str) -> str:
        row = self._db.execute("SELECT hash FROM ledger WHERE entity=? ORDER BY seq DESC LIMIT 1",
                               (self.entity_id(entity),)).fetchone()
        return row[0] if row else GENESIS

    def verify(self, entity: str) -> dict:
        """Recompute the hash chain. Returns ok, length, head and the first broken sequence number (or None)."""
        eid = self.entity_id(entity)
        rows = self._db.execute("SELECT seq, event, payload, prev, hash FROM ledger WHERE entity=? ORDER BY seq",
                                (eid,)).fetchall()
        prev = GENESIS
        for expected, (seq, event, payload, stored_prev, stored) in enumerate(rows):
            if seq != expected or stored_prev != prev or chain_hash(prev, entity, seq, event, payload) != stored:
                return {"ok": False, "length": len(rows), "broken_at": expected, "head": rows[-1][4]}
            prev = stored
        return {"ok": True, "length": len(rows), "broken_at": None, "head": prev}

    def snapshot(self, entity: str, params: dict[str, np.ndarray], label: str, meta: Optional[dict] = None) -> dict:
        """Record every parameter tensor as a `weight` geometry under one label; returns {name: hash}."""
        out = {}
        for name, W in params.items():
            gid = self.record(entity, "weight", W, layer=name, label=label, meta=meta)
            out[name] = self._db.execute("SELECT matrix FROM geometries WHERE id=?", (gid,)).fetchone()[0]
        return out

    def restore(self, entity: str, label: str) -> dict[str, np.ndarray]:
        eid = self.entity_id(entity)
        rows = self._db.execute(
            "SELECT layer, matrix FROM geometries WHERE entity=? AND kind='weight' AND label=? ORDER BY id",
            (eid, label)).fetchall()
        if not rows:
            raise KeyError(f"no weight snapshot {label!r} for {entity!r}")
        return {layer: self.get_matrix(h) for layer, h in rows}

    def labels(self, entity: str, kind: str = "weight") -> list[str]:
        eid = self.entity_id(entity)
        return [r[0] for r in self._db.execute(
            "SELECT label FROM geometries WHERE entity=? AND kind=? GROUP BY label ORDER BY MIN(id)", (eid, kind))]


class _Tx:
    def __init__(self, db: sqlite3.Connection, lock: threading.RLock):
        self.db, self.lock = db, lock

    def __enter__(self):
        self.lock.acquire()
        self.db.execute("BEGIN IMMEDIATE")
        return self.db

    def __exit__(self, exc_type, *_):
        try:
            self.db.execute("ROLLBACK" if exc_type else "COMMIT")
        finally:
            self.lock.release()
