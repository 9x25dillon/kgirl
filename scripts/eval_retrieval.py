#!/usr/bin/env python3
"""Score chunk retrieval on known-item questions.

`make-queries` has a local LLM write one question for each of a random sample of
chunks, without naming the chunk's symbol, and freezes them to a file. `run` then
scores each method on whether it finds the source chunk. Do them in that order so
no retrieval result can shape the questions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sqlite3
import statistics
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kgirl.harness.atlas.chunks import search as keyword_search  # noqa: E402
from kgirl.harness.atlas.retrieval import Retriever, reciprocal_rank_fusion  # noqa: E402
from kgirl.harness.atlas.store import Atlas  # noqa: E402
from kgirl.harness.atlas.vectors import QUERY_PREFIX, Matrix, ollama_embedder  # noqa: E402
from kgirl.harness.util import estimate_tokens  # noqa: E402

PROMPT = """Here is a piece of code from a repository.

{code}

Write one question a developer might ask whose answer is this code. Describe what it does in plain words.
Do not use any function, class or variable name from the code. Reply with only the question."""


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def generate(model: str, prompt: str) -> str:
    body = {"model": model, "prompt": prompt, "stream": False, "think": False,
            "options": {"temperature": 0.2, "num_predict": 60, "seed": 1729}}
    req = urllib.request.Request("http://127.0.0.1:11434/api/generate", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return " ".join(json.load(resp)["response"].split())


def make_queries(args) -> None:
    db = sqlite3.connect(args.chunks)
    rows = db.execute("SELECT id, symbol, text FROM chunks WHERE kind IN ('function','method','class') AND lang='python'"
                      " AND path LIKE 'src/kgirl/%' AND length(text) BETWEEN 200 AND 4000 ORDER BY id").fetchall()
    seen, unique = set(), []
    for row in rows:
        h = text_hash(row[2])
        if h not in seen:
            seen.add(h)
            unique.append(row)
    random.Random(args.seed).shuffle(unique)
    kept = rejected = 0
    with open(args.queries, "w") as f:
        for cid, symbol, text in unique:
            if kept >= args.n:
                break
            q = generate(args.model, PROMPT.format(code=text))
            leaf = symbol.split(".")[-1].lower()
            if len(q) < 15 or (len(leaf) > 2 and leaf in q.lower()):
                rejected += 1
                continue
            f.write(json.dumps({"id": cid, "query": q}) + "\n")
            f.flush()
            kept += 1
            print(f"\r{kept}/{args.n}", end="", flush=True)
    print(f"\nkept {kept}, rejected {rejected} of {len(unique)} candidates")


class Score:
    def __init__(self):
        self.ranks: list[int | None] = []
        self.tokens: list[int] = []

    def add(self, rank: int | None, tokens: int | None = None) -> None:
        self.ranks.append(rank)
        if tokens is not None:
            self.tokens.append(tokens)

    def row(self) -> dict:
        n = len(self.ranks)
        at = lambda k: round(sum(1 for r in self.ranks if r is not None and r < k) / n, 3)
        out = {"recall@1": at(1), "recall@5": at(5), "recall@10": at(10),
               "mrr@10": round(sum(1 / (r + 1) for r in self.ranks if r is not None and r < 10) / n, 3)}
        if self.tokens:
            out["median_tokens_to_answer"] = int(statistics.median(self.tokens))
        return out


def percentile(values: list[float], p: float) -> float:
    return round(sorted(values)[min(len(values) - 1, int(len(values) * p))], 1)


def run(args) -> None:
    queries = [json.loads(line) for line in open(args.queries)]
    db = sqlite3.connect(args.chunks)
    all_rows = {r[0]: r for r in db.execute("SELECT id, path, start_line, end_line, text FROM chunks")}
    by_hash: dict[str, set[int]] = {}
    for cid, row in all_rows.items():
        by_hash.setdefault(text_hash(row[4]), set()).add(cid)
    matrix = Matrix(args.matrix)
    embed = ollama_embedder(matrix.model)

    def scored(ranking: list[int], gold: set[int], score: Score) -> None:
        rank = next((i for i, c in enumerate(ranking[:10]) if c in gold), None)
        tokens = None if rank is None else sum(estimate_tokens(all_rows[c][4]) for c in ranking[:rank + 1])
        score.add(rank, tokens)

    methods = {name: Score() for name in ("keyword", "vector_full", "vector_s100", "vector_s400", "vector_s1000",
                                          "hybrid_s400", "atlas_current")}
    tmp = tempfile.TemporaryDirectory()
    atlas = Atlas(Path(tmp.name) / "atlas.db")
    atlas.ingest(args.repo_root, "kgirl", None)
    for q in queries:
        gold = by_hash[text_hash(all_rows[q["id"]][4])]
        text = q["query"]
        kw = [r for r in db.execute(
            "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts, 4.0, 0.7, 1.0, 0.3)"
            " LIMIT 40", (fts_or_empty(text),))] if fts_or_empty(text) else []
        kw = [r[0] for r in kw]
        scored(kw, gold, methods["keyword"])
        bits, vals = matrix.encode_query(embed([QUERY_PREFIX + text])[0])
        for name, shortlist in (("vector_full", matrix.count), ("vector_s100", 100), ("vector_s400", 400),
                                ("vector_s1000", 1000)):
            scored([h[0] for h in matrix.search(bits, vals, 40, shortlist)], gold, methods[name])
        vec = [h[0] for h in matrix.search(bits, vals, 40, 400)]
        scored([i for i, _ in reciprocal_rank_fusion([kw, vec])], gold, methods["hybrid_s400"])
        spans = [(all_rows[c][1], all_rows[c][2], all_rows[c][3]) for c in gold]
        hits = atlas.search(text, 10, None)
        methods["atlas_current"].add(next((i for i, h in enumerate(hits)
                                           if any(h.path == p and a <= h.line <= b for p, a, b in spans)), None))
    atlas.close()
    tmp.cleanup()
    print(json.dumps({"queries": len(queries), "chunks": matrix.count,
                      "methods": {k: v.row() for k, v in methods.items()}}, indent=1))
    if args.router_ssh:
        router_checks(args, queries, matrix, embed)


def fts_or_empty(text: str) -> str:
    from kgirl.harness.util import fts_query
    return fts_query(text)


def router_checks(args, queries, matrix, embed) -> None:
    import shlex
    base = shlex.split(args.router_ssh)
    where = args.router_dir
    remote = Retriever(args.chunks, args.matrix, embed=embed,
                       vector_cmd=base + [f"{where}/kgvecd {where}/{Path(args.matrix).name}"],
                       sql_cmd=base + [f"sqlite3 -batch -readonly -json {where}/{Path(args.chunks).name}"])
    local = Retriever(args.chunks, args.matrix, embed=embed)
    mismatches = 0
    pipe = remote._vector_pipe()
    if pipe is None:
        raise SystemExit("router vector pipe did not open")
    daemon_ms = []
    for q in queries:
        bits, vals = matrix.encode_query(embed([QUERY_PREFIX + q["query"]])[0])
        got = pipe.search(bits, vals, 40, 400, 2.0)
        daemon_ms.append(pipe.micros / 1000)
        mismatches += got != matrix.search(bits, vals, 40, 400)
    timings = {"router": {}, "local": {}}
    for name, r in (("router", remote), ("local", local)):
        for _ in range(2):  # the second pass is the one that counts: pipes open and caches warm
            all_ms: dict[str, list[float]] = {}
            for q in queries:
                t = time.perf_counter()
                r.search(q["query"], 8)
                all_ms.setdefault("total", []).append((time.perf_counter() - t) * 1e3)
                for key, value in r.info.millis.items():
                    all_ms.setdefault(key, []).append(value)
        timings[name] = {key: {"p50": percentile(v, 0.5), "p90": percentile(v, 0.9)} for key, v in all_ms.items()}
        timings[name]["backends"] = [r.info.keyword_backend, r.info.vector_backend]
    print(json.dumps({"router_vs_local_mismatches": mismatches, "of": len(queries),
                      "daemon_scan_ms": {"p50": percentile(daemon_ms, 0.5), "p90": percentile(daemon_ms, 0.9)},
                      "timings_ms": timings}, indent=1))
    remote.close()
    local.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make-queries")
    m.add_argument("chunks")
    m.add_argument("queries")
    m.add_argument("--model", default="qwen2.5:3b")
    m.add_argument("--n", type=int, default=100)
    m.add_argument("--seed", type=int, default=1729)
    m.set_defaults(fn=make_queries)
    r = sub.add_parser("run")
    r.add_argument("chunks")
    r.add_argument("matrix")
    r.add_argument("queries")
    r.add_argument("--repo-root", default=str(ROOT))
    r.add_argument("--router-ssh")
    r.add_argument("--router-dir", default="/mnt/kgirl")
    r.set_defaults(fn=run)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
