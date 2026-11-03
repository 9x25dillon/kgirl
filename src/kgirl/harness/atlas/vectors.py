"""Vector matrix for chunk search: a bit matrix for the coarse pass, int8 for the fine pass.

Built on the desktop from a chunk index and copied to the router, where
`router/kgvecd.c` serves the same file. `Matrix.search` ranks exactly as the
daemon does, so either side can answer a query.

    python -m kgirl.harness.atlas.vectors chunks.db chunks.kgvm --cache embed-cache.db
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import struct
import urllib.request
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

MAGIC = b"KGVM"
VERSION = 1
HEADER = 64  # magic, version, count, dim, scale, model[32], padding
HEADER_FMT = "<4sIIIf32s"
DEFAULT_MODEL = "nomic-embed-text"
DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "
MAX_EMBED_CHARS = 6000
OLLAMA_URL = "http://127.0.0.1:11434"

Embedder = Callable[[Sequence[str]], np.ndarray]


def ollama_embedder(model: str = DEFAULT_MODEL, url: str | None = None, cpu: bool = True,
                    timeout: float = 300.0) -> Embedder:
    """Embed with a local Ollama. `cpu` keeps the model out of GPU memory, leaving it for the LLM."""
    base = (url or os.environ.get("OLLAMA_HOST") or OLLAMA_URL).rstrip("/")
    if "://" not in base:
        base = "http://" + base

    def embed(texts: Sequence[str]) -> np.ndarray:
        body: dict = {"model": model, "input": list(texts)}
        if cpu:
            body["options"] = {"num_gpu": 0}
        req = urllib.request.Request(base + "/api/embed", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return np.asarray(json.load(resp)["embeddings"], dtype=np.float32)

    return embed


def normalize(vecs: np.ndarray) -> np.ndarray:
    vecs = np.asarray(vecs, dtype=np.float32)
    return vecs / np.maximum(np.linalg.norm(vecs, axis=-1, keepdims=True), 1e-9)


def to_bits(vecs: np.ndarray) -> np.ndarray:
    return np.packbits(vecs > 0, axis=-1, bitorder="little")


def to_int8(vecs: np.ndarray, scale: float) -> np.ndarray:
    return np.clip(np.rint(vecs * scale), -127, 127).astype(np.int8)


def calibrate_scale(vecs: np.ndarray) -> float:
    """Map the 99.9th percentile magnitude to 127; unit vectors alone would use only a few levels."""
    return float(127.0 / max(float(np.percentile(np.abs(vecs), 99.9)), 1e-6))


def write_matrix(path: str | os.PathLike, ids: Sequence[int], vecs: np.ndarray, model: str) -> dict:
    """Write atomically; the old file stays in place until the new one is complete."""
    vecs = normalize(vecs)
    if vecs.ndim != 2 or vecs.shape[0] != len(ids):
        raise ValueError("expected one vector per id")
    count, dim = vecs.shape
    if dim % 8:
        raise ValueError("dimension must be a multiple of 8")
    if max(ids, default=0) > 0xFFFFFFFF or min(ids, default=0) < 0:
        raise ValueError("ids must fit in u32")
    scale = calibrate_scale(vecs)
    target = Path(path)
    tmp = target.with_name(target.name + ".tmp")
    head = struct.pack(HEADER_FMT, MAGIC, VERSION, count, dim, scale, model.encode()[:32])
    with open(tmp, "wb") as f:
        f.write(head.ljust(HEADER, b"\0"))
        f.write(np.asarray(ids, dtype="<u4").tobytes())
        f.write(to_bits(vecs).tobytes())
        f.write(to_int8(vecs, scale).tobytes())
    tmp.replace(target)
    return {"count": count, "dim": dim, "scale": scale, "bytes": target.stat().st_size}


class Matrix:
    """Read-only view of a matrix file, searched in this process."""

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        raw = np.memmap(self.path, dtype=np.uint8, mode="r")
        if len(raw) < HEADER:
            raise ValueError(f"{path}: too small")
        magic, version, count, dim, scale, model = struct.unpack_from(HEADER_FMT, raw[:HEADER].tobytes())
        if magic != MAGIC or version != VERSION:
            raise ValueError(f"{path}: not a version {VERSION} matrix")
        nbytes = dim // 8
        if len(raw) != HEADER + 4 * count + count * nbytes + count * dim:
            raise ValueError(f"{path}: size does not match its header")
        self.count, self.dim, self.scale = count, dim, scale
        self.model = model.rstrip(b"\0").decode()
        bits_at = HEADER + 4 * count
        self.ids = np.frombuffer(raw, dtype="<u4", count=count, offset=HEADER)
        self.bits = raw[bits_at:bits_at + count * nbytes].reshape(count, nbytes)
        self.vals = np.frombuffer(raw, dtype=np.int8, offset=bits_at + count * nbytes).reshape(count, dim)

    def encode_query(self, vec: np.ndarray) -> tuple[bytes, bytes]:
        v = normalize(vec).reshape(-1)
        if v.shape[0] != self.dim:
            raise ValueError(f"query has {v.shape[0]} dimensions, matrix has {self.dim}")
        return to_bits(v).tobytes(), to_int8(v, self.scale).tobytes()

    def search(self, query_bits: bytes, query_vals: bytes, k: int = 20, shortlist: int = 400) -> list[tuple[int, int]]:
        """Return (id, score), best first; `shortlist` is how many coarse hits get the fine pass."""
        k = min(k, self.count)
        if k <= 0:
            return []
        want = max(min(shortlist, self.count), k)
        q = np.frombuffer(query_bits, dtype=np.uint8)
        dist = np.bitwise_count(self.bits ^ q).sum(axis=1)
        picks = np.argsort(dist, kind="stable")[:want]
        picks.sort()
        scores = self.vals[picks].astype(np.int32) @ np.frombuffer(query_vals, dtype=np.int8).astype(np.int32)
        order = np.argsort(-scores, kind="stable")[:k]
        return [(int(self.ids[picks[i]]), int(scores[i])) for i in order]


def _batches(items: list, size: int) -> Iterable[list]:
    for i in range(0, len(items), size):
        yield items[i:i + size]


class EmbedCache:
    """Document embeddings keyed by model and exact input, so a rebuild embeds only chunks whose text changed."""

    def __init__(self, path: str | os.PathLike):
        self.db = sqlite3.connect(str(path))
        self.db.execute("CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, dim INTEGER, vec BLOB)")

    @staticmethod
    def key(model: str, doc: str) -> str:
        return hashlib.sha256(f"{model}\0{doc}".encode()).hexdigest()

    def get(self, keys: Sequence[str]) -> dict[str, np.ndarray]:
        found: dict[str, np.ndarray] = {}
        for part in _batches(list(keys), 500):
            marks = ",".join("?" * len(part))
            for k, dim, blob in self.db.execute(f"SELECT key, dim, vec FROM embeddings WHERE key IN ({marks})", part):
                found[k] = np.frombuffer(blob, dtype=np.float32, count=dim)
        return found

    def put(self, keys: Sequence[str], vecs: np.ndarray) -> None:
        vecs = np.asarray(vecs, dtype=np.float32)
        self.db.executemany("INSERT OR REPLACE INTO embeddings VALUES (?, ?, ?)",
                            [(k, int(v.shape[0]), v.tobytes()) for k, v in zip(keys, vecs)])
        self.db.commit()

    def close(self) -> None:
        self.db.close()


def build(chunks_db: str | os.PathLike, out: str | os.PathLike, embed: Embedder | None = None,
          model: str = DEFAULT_MODEL, batch: int = 32, progress: Callable[[int, int], None] | None = None,
          cache: str | os.PathLike | None = None) -> dict:
    """Embed every chunk in a chunk index and write the matrix. With `cache`, unchanged chunks reuse stored vectors."""
    embed = embed or ollama_embedder(model)
    db = sqlite3.connect(f"file:{Path(chunks_db)}?mode=ro", uri=True)
    try:
        rows = db.execute("SELECT id, path, symbol, text FROM chunks ORDER BY id").fetchall()
    finally:
        db.close()
    if not rows:
        raise ValueError(f"{chunks_db}: no chunks")
    docs = [f"{DOC_PREFIX}{p} {s or ''}\n{t[:MAX_EMBED_CHARS]}" for _, p, s, t in rows]
    store = EmbedCache(cache) if cache else None
    try:
        keys = [EmbedCache.key(model, d) for d in docs]
        known = store.get(keys) if store else {}
        todo = [i for i, k in enumerate(keys) if k not in known]
        done = len(rows) - len(todo)
        for part in _batches(todo, batch):
            got = embed([docs[i] for i in part])
            for i, v in zip(part, got):
                known[keys[i]] = np.asarray(v, dtype=np.float32)
            if store:
                store.put([keys[i] for i in part], got)
            done += len(part)
            if progress:
                progress(done, len(rows))
    finally:
        if store:
            store.close()
    result = write_matrix(out, [r[0] for r in rows], np.stack([known[k] for k in keys]), model)
    result["embedded"] = len(todo)
    return result

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("chunks_db")
    ap.add_argument("out")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--gpu", action="store_true", help="embed on the GPU (faster, but takes GPU memory)")
    ap.add_argument("--cache", help="embedding cache file; a rebuild then embeds only chunks whose text changed")
    args = ap.parse_args()
    embedder = ollama_embedder(args.model, cpu=not args.gpu)
    print(build(args.chunks_db, args.out, embedder, args.model,
                progress=lambda d, n: print(f"\r{d}/{n}", end="", flush=True), cache=args.cache))


if __name__ == "__main__":
    main()
