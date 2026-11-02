"""Hybrid chunk retrieval: keyword and vector search, on the router when it answers.

The router holds the chunk index and the vector matrix. Each is reached through
one long-lived SSH pipe, so a query costs no new connection and no port is opened.
Every call has a deadline. A slow or dead router costs at most that deadline once,
then the local copies answer until the retry interval has passed.

Set these to turn it on (`KGIRL_ROUTER_SSH` is optional):

    KGIRL_CHUNKS_DB    local chunk index
    KGIRL_MATRIX       local vector matrix
    KGIRL_ROUTER_SSH   "ssh -i KEY ... root@192.168.1.1"
    KGIRL_ROUTER_DIR   directory on the router, default /mnt/kgirl
"""

from __future__ import annotations

import json
import os
import select
import shlex
import sqlite3
import struct
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from ..util import fts_query
from . import chunks as chunk_index
from .vectors import QUERY_PREFIX, Embedder, Matrix, ollama_embedder

MAGIC_READY = 0x5244474B
MAGIC_REQ = 0x5152474B
MAGIC_RESP = 0x5352474B
RRF_K = 60
SENTINEL = "__KGIRL_END__"


class RemoteError(Exception):
    pass


class Pipe:
    """A child process spoken to over stdin/stdout, with a deadline on every read."""

    def __init__(self, argv: list[str]):
        self.proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, bufsize=0)
        self.buf = b""

    def send(self, data: bytes) -> None:
        try:
            self.proc.stdin.write(data)
            self.proc.stdin.flush()
        except OSError as e:
            raise RemoteError(f"write failed: {e}") from e

    def _fill(self, deadline: float) -> None:
        left = deadline - time.monotonic()
        ready, _, _ = select.select([self.proc.stdout], [], [], max(left, 0))
        if not ready:
            raise RemoteError("deadline passed")
        data = os.read(self.proc.stdout.fileno(), 65536)
        if not data:
            raise RemoteError("pipe closed")
        self.buf += data

    def read_exact(self, n: int, deadline: float) -> bytes:
        while len(self.buf) < n:
            self._fill(deadline)
        out, self.buf = self.buf[:n], self.buf[n:]
        return out

    def read_line(self, deadline: float) -> bytes:
        while b"\n" not in self.buf:
            self._fill(deadline)
        line, self.buf = self.buf.split(b"\n", 1)
        return line

    def close(self) -> None:
        for stream in (self.proc.stdin, self.proc.stdout):
            try:
                stream.close()
            except OSError:
                pass
        self.proc.kill()
        self.proc.wait()


class VectorPipe:
    """Client for kgvecd."""

    def __init__(self, argv: list[str], connect_timeout: float = 8.0):
        self.pipe = Pipe(argv)
        try:
            magic, self.count, self.dim = struct.unpack("<III", self.pipe.read_exact(
                12, time.monotonic() + connect_timeout))
        except RemoteError:
            self.close()
            raise
        if magic != MAGIC_READY:
            self.close()
            raise RemoteError("daemon sent a bad greeting")
        self.micros = 0

    def search(self, query_bits: bytes, query_vals: bytes, k: int, shortlist: int,
               timeout: float) -> list[tuple[int, int]]:
        deadline = time.monotonic() + timeout
        self.pipe.send(struct.pack("<III", MAGIC_REQ, k, shortlist) + query_bits + query_vals)
        magic, n, self.micros = struct.unpack("<III", self.pipe.read_exact(12, deadline))
        if magic != MAGIC_RESP or n > k:
            raise RemoteError("daemon sent a bad response")
        body = self.pipe.read_exact(8 * n, deadline)
        return [struct.unpack_from("<Ii", body, 8 * i) for i in range(n)]

    def close(self) -> None:
        self.pipe.close()


def sql_literal(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


class SqlPipe:
    """A long-lived read-only sqlite3 shell on the far side of a pipe."""

    def __init__(self, argv: list[str], connect_timeout: float = 8.0):
        self.pipe = Pipe(argv)
        try:
            self.query("SELECT 1 AS ok", connect_timeout)
        except RemoteError:
            self.close()
            raise

    def query(self, sql: str, timeout: float) -> list[dict]:
        deadline = time.monotonic() + timeout
        self.pipe.send(f"{sql};\nSELECT '{SENTINEL}' AS s;\n".encode())
        lines = []
        while True:
            line = self.pipe.read_line(deadline)
            if SENTINEL.encode() in line:
                break
            lines.append(line)
        if not lines:
            return []
        try:
            return json.loads(b"\n".join(lines))
        except ValueError as e:
            raise RemoteError(f"bad JSON from sqlite3: {e}") from e

    def close(self) -> None:
        self.pipe.close()


@dataclass(frozen=True)
class ChunkHit:
    id: int
    repo: str
    path: str
    start_line: int
    end_line: int
    kind: str
    symbol: str
    text: str
    score: float

    def cite(self) -> str:
        return f"{self.repo}:{self.path}:{self.start_line}"


@dataclass
class RetrievalInfo:
    keyword_backend: str = ""
    vector_backend: str = ""
    millis: dict[str, float] = field(default_factory=dict)


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


FETCH_COLUMNS = "id, repo, path, start_line, end_line, kind, symbol, text"
KEYWORD_SQL = ("SELECT rowid AS id FROM chunks_fts WHERE chunks_fts MATCH {query} ORDER BY "
               + chunk_index.BM25 + " LIMIT {limit}")


class Retriever:
    def __init__(self, chunks_db: str | os.PathLike, matrix_path: str | os.PathLike, embed: Embedder | None = None,
                 vector_cmd: list[str] | None = None, sql_cmd: list[str] | None = None,
                 timeout: float = 0.5, retry_after: float = 30.0, shortlist: int = 400, candidates: int = 40):
        self.db = sqlite3.connect(f"file:{Path(chunks_db)}?mode=ro", uri=True, check_same_thread=False)
        self.matrix = Matrix(matrix_path)
        self.embed = embed or ollama_embedder(self.matrix.model)
        self.vector_cmd, self.sql_cmd = vector_cmd, sql_cmd
        self.timeout, self.retry_after = timeout, retry_after
        self.shortlist, self.candidates = shortlist, candidates
        self._vec: VectorPipe | None = None
        self._sql: SqlPipe | None = None
        self._next_try = {"vec": 0.0, "sql": 0.0}
        self.info = RetrievalInfo()

    @classmethod
    def from_env(cls) -> "Retriever | None":
        chunks_db, matrix = os.environ.get("KGIRL_CHUNKS_DB"), os.environ.get("KGIRL_MATRIX")
        if not chunks_db or not matrix:
            return None
        ssh = os.environ.get("KGIRL_ROUTER_SSH")
        vector_cmd = sql_cmd = None
        if ssh:
            base = shlex.split(ssh)
            where = os.environ.get("KGIRL_ROUTER_DIR", "/mnt/kgirl")
            vector_cmd = base + [f"{where}/kgvecd {where}/{Path(matrix).name}"]
            sql_cmd = base + [f"sqlite3 -batch -readonly -json {where}/{Path(chunks_db).name}"]
        return cls(chunks_db, matrix, vector_cmd=vector_cmd, sql_cmd=sql_cmd)

    def _remote(self, name: str, make: Callable[[], object], attr: str):
        """The open pipe, a new one if it's time to try, or None while the router is out of favor."""
        pipe = getattr(self, attr)
        if pipe is not None:
            return pipe
        if time.monotonic() < self._next_try[name]:
            return None
        try:
            pipe = make()
        except (OSError, RemoteError):
            self._next_try[name] = time.monotonic() + self.retry_after
            return None
        setattr(self, attr, pipe)
        return pipe

    def _drop(self, name: str, attr: str) -> None:
        pipe = getattr(self, attr)
        if pipe is not None:
            pipe.close()
        setattr(self, attr, None)
        self._next_try[name] = time.monotonic() + self.retry_after

    def _vector_pipe(self) -> VectorPipe | None:
        if not self.vector_cmd:
            return None
        pipe = self._remote("vec", lambda: VectorPipe(self.vector_cmd), "_vec")
        if pipe is not None and (pipe.count != self.matrix.count or pipe.dim != self.matrix.dim):
            self._drop("vec", "_vec")  # the router has a different build of the matrix
            return None
        return pipe

    def vector_ids(self, query: str, limit: int) -> list[int]:
        t = time.perf_counter()
        vec = self.embed([QUERY_PREFIX + query])[0]
        self.info.millis["embed"] = (time.perf_counter() - t) * 1e3
        q_bits, q_vals = self.matrix.encode_query(vec)
        t = time.perf_counter()
        pipe = self._vector_pipe()
        hits = None
        if pipe is not None:
            try:
                hits = pipe.search(q_bits, q_vals, limit, self.shortlist, self.timeout)
                self.info.vector_backend = "router"
            except (OSError, RemoteError, struct.error):
                self._drop("vec", "_vec")
        if hits is None:
            hits = self.matrix.search(q_bits, q_vals, limit, self.shortlist)
            self.info.vector_backend = "local"
        self.info.millis["vector"] = (time.perf_counter() - t) * 1e3
        return [h[0] for h in hits]

    def _sql_pipe(self) -> SqlPipe | None:
        if not self.sql_cmd:
            return None
        return self._remote("sql", lambda: SqlPipe(self.sql_cmd), "_sql")

    def keyword_ids(self, query: str, limit: int) -> list[int]:
        fts = fts_query(query)
        if not fts:
            return []
        t = time.perf_counter()
        pipe = self._sql_pipe()
        ids = None
        if pipe is not None:
            try:
                sql = KEYWORD_SQL.format(query=sql_literal(fts), limit=int(limit))
                ids = [r["id"] for r in pipe.query(sql, self.timeout)]
                self.info.keyword_backend = "router"
            except (OSError, RemoteError, KeyError, TypeError):
                self._drop("sql", "_sql")
        if ids is None:
            sql = KEYWORD_SQL.format(query="?", limit=int(limit))
            ids = [r[0] for r in self.db.execute(sql, (fts,))]
            self.info.keyword_backend = "local"
        self.info.millis["keyword"] = (time.perf_counter() - t) * 1e3
        return ids

    def fetch(self, ids: list[int]) -> dict[int, tuple]:
        if not ids:
            return {}
        marks = ",".join(str(int(i)) for i in ids)
        t = time.perf_counter()
        rows = None
        pipe = self._sql_pipe()
        if pipe is not None:
            try:
                got = pipe.query(f"SELECT {FETCH_COLUMNS} FROM chunks WHERE id IN ({marks})", self.timeout * 4)
                rows = [tuple(r[c] for c in FETCH_COLUMNS.split(", ")) for r in got]
            except (OSError, RemoteError, KeyError, TypeError):
                self._drop("sql", "_sql")
        if rows is None:
            rows = self.db.execute(f"SELECT {FETCH_COLUMNS} FROM chunks WHERE id IN ({marks})").fetchall()
        self.info.millis["fetch"] = (time.perf_counter() - t) * 1e3
        return {r[0]: r for r in rows}

    def search(self, query: str, limit: int = 8) -> list[ChunkHit]:
        """Best chunks for `query`. A failed embedder leaves keyword search; it never raises for that."""
        self.info = RetrievalInfo()
        keyword = self.keyword_ids(query, self.candidates)
        try:
            vector = self.vector_ids(query, self.candidates)
        except (OSError, ValueError, KeyError) as e:  # Ollama down or the wrong model
            self.info.vector_backend = f"unavailable: {type(e).__name__}"
            vector = []
        fused = reciprocal_rank_fusion([keyword, vector])[:limit]
        rows = self.fetch([i for i, _ in fused])
        return [ChunkHit(*rows[i], score=s) for i, s in fused if i in rows]

    def close(self) -> None:
        for name, attr in (("vec", "_vec"), ("sql", "_sql")):
            self._drop(name, attr)
        self.db.close()
