"""Real-time inference over the holo store: KGVM coarse-to-fine search, with provenance on every hit.

    query vector ──Matrix.encode_query──▶ sign bits + int8 ──Matrix.search──▶ (chunk id, score), best first
                                                                                  │
        owning entities (repo:…, kind:…), their centroid digests and ledger heads ◀┘   from the HoloStore
        the content address of the hit's own vector, and whether the store holds that vector

The guidance block says where in the space the query landed: each owning entity's share of the positive
hit score, and the cosine between the query and the entity's centroid.

Optional prior. When a trained `tauls.holo.HoloMemory` exists at `$KGIRL_HOME/holo-memory.npz`, with the
matrix's dimension and one token per entity of `HoloStore.vocabulary()`, its distribution over entities is
mixed with the retrieval's by the free-energy gate defined in tauls/holo.py:

    w = gate(entropy(p_ret), entropy(p_mem), prior, temp)        p = (1 − w)·p_ret + w·p_mem

Nothing is trained here, and the prior changes the guidance block only: hits keep the order KGVM gave them.

Text queries are embedded with the matrix's own model (one model call). A vector query makes none.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ...vessel.matrixdb import encode_matrix
from ..atlas.vectors import QUERY_PREFIX, Embedder, Matrix, normalize, ollama_embedder
from .store import CENTROID, CHUNK, ROOT, HoloError, HoloStore, kind_entity, repo_entity

MEMORY_NAME = "holo-memory.npz"         # a trained HoloMemory (never written by this layer)
GATE_NAME = "holo-memory.json"          # its fitted settings: the dict tauls.holo.fit_gate returns
TOPIC_QUERY, TOPIC_REGULATE = "holo.query", "holo.regulate"
GUIDANCE = 6                            # entities listed in the guidance block
_GATE_KEYS = ("prior", "temp", "beta", "eta", "knn_eta", "tau")


@dataclass(frozen=True)
class GateConfig:
    """Settings of the HoloMemory prior. `prior` and `temp` feed tauls.holo.gate; the rest, the memory."""
    prior: float = 0.1
    temp: float = 1.0
    beta: float | None = None           # None: the memory's own query_beta
    eta: float = 0.0
    knn_eta: float = 0.0
    tau: float = 0.05

    @classmethod
    def load(cls, path: str | os.PathLike) -> "GateConfig":
        """The fitted settings beside the memory file; the defaults when there are none."""
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cls()
        if not isinstance(raw, dict):
            return cls()
        return cls(**{key: raw[key] for key in _GATE_KEYS if raw.get(key) is not None})


def query(store: HoloStore, q: str | Sequence[float] | np.ndarray, k: int = 8, *, shortlist: int = 400,
          embed: Embedder | None = None, bus=None, metabolism=None, memory=None, gate_cfg: GateConfig | None = None,
          energy: float = 0.0) -> dict:
    """Search the space the store was built from. Returns hits with provenance, and the guidance block.

    `bus` gets `holo.query` (and `holo.regulate`). `metabolism`, when given and life is enabled, observes
    the query as one life step with `energy` flowing in and no work charged.
    """
    t0 = time.perf_counter()
    s = _session(store)
    matrix: Matrix = s["matrix"]
    text = q if isinstance(q, str) else None
    if text is not None:
        if not text.strip():
            raise HoloError("the query is empty")
        embed = embed or ollama_embedder(matrix.model)
        vec = np.asarray(embed([QUERY_PREFIX + text])[0], dtype=np.float32)
    else:
        vec = np.asarray(q, dtype=np.float32)
    vec = vec.reshape(-1)
    bits, vals = matrix.encode_query(vec)                    # ValueError when the dimensions differ
    found = matrix.search(bits, vals, k=int(k), shortlist=shortlist)

    names, index = s["names"], s["index"]
    rows = _chunk_rows(s["chunks"], [chunk_id for chunk_id, _ in found])
    mass = np.zeros(len(names), np.float64)                  # positive hit score per owning entity
    owned = np.zeros(len(names), np.int64)
    hits: list[dict] = []
    for chunk_id, score in found:
        hit: dict[str, Any] = {"id": chunk_id, "score": score, "repo": None, "path": None, "symbol": None,
                               "kind": None, "line": None, "cite": None, "entities": [],
                               "digest": _vector_digest(s, chunk_id), "stored": False, "provenance": []}
        row = rows.get(chunk_id)
        if row is not None:
            _, repo, path, kind, symbol, line = row
            owners = [e for e in (repo_entity(repo), kind_entity(kind)) if e in index]
            hit.update(repo=repo, path=path, symbol=symbol or "", kind=kind or "other", line=line,
                       cite=f"{repo}:{path}:{line}", entities=owners)
            for entity in owners:
                hit["provenance"].append({"entity": entity, "centroid": s["centroid_digest"][entity],
                                          "head": s["heads"][entity]})
                hit["stored"] = hit["stored"] or store.digest_of(entity, CHUNK, str(chunk_id)) == hit["digest"]
                mass[index[entity]] += max(score, 0)
                owned[index[entity]] += 1
        hits.append(hit)

    unit = normalize(vec)
    total = float(mass.sum())
    p_ret = mass / total if total > 0 else np.full(len(names), 1.0 / max(len(names), 1))
    cosine = s["centroids"] @ unit
    prior, p, p_mem = _prior(store, s, unit, p_ret, memory, gate_cfg)
    order = sorted(range(len(names)), key=lambda i: (-p[i], names[i]))
    entities = []
    for i in order[:GUIDANCE]:
        row = {"entity": names[i], "p": float(p[i]), "share": float(p_ret[i]), "cosine": float(cosine[i]),
               "hits": int(owned[i])}
        if p_mem is not None:
            row["p_memory"] = float(p_mem[i])
        entities.append(row)

    result = {"query": text, "k": int(k), "ms": round((time.perf_counter() - t0) * 1e3, 3), "hits": hits,
              "entities": entities, "prior": prior,
              "matrix": {"count": int(matrix.count), "dim": int(matrix.dim), "model": matrix.model},
              "regulation": None}
    if bus is not None:
        bus.publish(TOPIC_QUERY, {
            "q": text[:200] if text else None, "k": int(k), "ms": result["ms"],
            "hits": [{"id": h["id"], "score": h["score"], "cite": h["cite"], "entities": h["entities"]} for h in hits],
            "top": entities[0]["entity"] if entities else None, "prior": prior["used"]}, "holo")
    if metabolism is not None:
        from ..metabolism import enabled                     # kgirl.life loads only when regulation is asked for
        if enabled():
            seconds = time.perf_counter() - t0
            obs = (1.0 if hits else 0.0, math.log1p(seconds) / math.log1p(600.0), 0.0, 0.0)
            row = metabolism.observe(obs, energy=energy, work=0.0, kind=TOPIC_QUERY)
            result["regulation"] = row
            if bus is not None:
                bus.publish(TOPIC_REGULATE, {"obs": [round(x, 4) for x in obs],
                                             **{key: row[key] for key in ("F", "O", "C", "R", "err")}}, "holo")
    return result


# ---------------------------------------------------------------------- session

def _session(store: HoloStore) -> dict:
    """The opened matrix, chunk index and centroids, kept on the store until the next build."""
    src = store.source()
    if src is None:
        raise HoloError("the holo store has not been built yet: run build() first")
    head = store.db.head(ROOT)
    s = store.cache
    if s.get("head") == head:
        return s
    store.drop_cache()
    try:
        matrix = Matrix(src["matrix"]["path"])
    except (OSError, ValueError) as exc:
        raise HoloError(f"the matrix the store was built from cannot be opened ({exc}); rebuild the store") from exc
    names = store.vocabulary()
    centroids = np.zeros((len(names), matrix.dim), np.float32)
    digests: dict[str, str | None] = {}
    for i, name in enumerate(names):
        digests[name] = store.digest_of(name, CENTROID)
        if digests[name]:
            centroid = store.db.get_matrix(digests[name])
            if centroid.shape == (matrix.dim,):              # an entity left over from an older matrix has none
                centroids[i] = centroid
    s.update(head=head, matrix=matrix, chunks=_open_chunks(src["chunks_db"]), names=names,
             row_of={chunk_id: row for row, chunk_id in enumerate(np.asarray(matrix.ids).tolist())},
             index={name: i for i, name in enumerate(names)}, centroids=normalize(centroids),
             centroid_digest=digests, heads={name: store.db.head(name) for name in names})
    return s


def _vector_digest(s: dict, chunk_id: int) -> str | None:
    """The content address of a chunk's KGVM vector: the digest MatrixDB gives that int8 row."""
    row = s["row_of"].get(chunk_id)
    return encode_matrix(np.array(s["matrix"].vals[row]))[0] if row is not None else None


def _open_chunks(path: str) -> sqlite3.Connection | None:
    """The chunk index, read-only; None when it has moved (hits then carry ids and scores only)."""
    try:
        return sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True, check_same_thread=False)
    except sqlite3.Error:
        return None


def _chunk_rows(con: sqlite3.Connection | None, ids: list[int]) -> dict[int, tuple]:
    if con is None or not ids:
        return {}
    marks = ",".join("?" * len(ids))
    try:
        rows = con.execute(f"SELECT id, repo, path, kind, symbol, start_line FROM chunks WHERE id IN ({marks})",
                           ids).fetchall()
    except sqlite3.Error:
        return {}
    return {row[0]: row for row in rows}


# ---------------------------------------------------------------------- the HoloMemory prior

def _prior(store: HoloStore, s: dict, unit: np.ndarray, p_ret: np.ndarray, memory, cfg: GateConfig | None
           ) -> tuple[dict, np.ndarray, np.ndarray | None]:
    """(what happened, the distribution over entities to guide with, the memory's own distribution or None)."""
    path = store.home / MEMORY_NAME
    if memory is None:
        if not path.exists():
            return {"used": False, "reason": f"no trained memory at {path}"}, p_ret, None
        try:
            memory = _memory(s, path)
        except (OSError, ValueError, KeyError) as exc:
            return {"used": False, "reason": f"{path} could not be read: {exc}"}, p_ret, None
    names, matrix = s["names"], s["matrix"]
    if memory.dim != matrix.dim or memory.vocab != len(names):
        return {"used": False, "reason": f"the memory has {memory.dim} dimensions and {memory.vocab} tokens; the "
                                         f"store needs {matrix.dim} dimensions and {len(names)} entities"}, p_ret, None
    cfg = cfg or GateConfig.load(store.home / GATE_NAME)
    if not 0.0 < cfg.prior < 1.0 or not cfg.temp > 0.0:
        return {"used": False, "reason": f"gate settings out of range: prior={cfg.prior}, temp={cfg.temp}"}, p_ret, None
    from ...tauls.holo import entropy, gate                  # the formulas, where they are defined
    p_mem, _ = memory.distribution(memory.project(unit[None, :]), eta=cfg.eta, beta=cfg.beta, knn_eta=cfg.knn_eta,
                                   tau=cfg.tau)
    p_mem = np.asarray(p_mem, np.float64)
    p_lm = p_ret[None, :]
    h_ret, h_mem = entropy(p_lm), entropy(p_mem)
    w = gate(h_ret, h_mem, cfg.prior, cfg.temp)
    p = (1.0 - w)[:, None] * p_lm + w[:, None] * p_mem
    info = {"used": True, "weight": float(w[0]), "h_retrieval": float(h_ret[0]), "h_memory": float(h_mem[0]),
            "prior": cfg.prior, "temp": cfg.temp}
    return info, p[0], p_mem[0]


def _memory(s: dict, path: Path):
    """The trained memory, reloaded only when its file changes."""
    stamp = path.stat().st_mtime_ns
    if s.get("memory_stamp") != stamp:
        from ...tauls.holo import HoloMemory
        s["memory"], s["memory_stamp"] = HoloMemory.load(path), stamp
    return s["memory"]
