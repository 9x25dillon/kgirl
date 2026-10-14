"""Corpus → token shards for TA-ULS, and the deterministic batch sampler that reads them.

    roots ─walk (sorted)─▶ text files ─filter─▶ exact dedupe (sha1) ─▶ documents
          ─split by sha─▶ train / val ─tokenize─▶ train.bin, val.bin (uint16/uint32 memmap)
                                               └─▶ meta.json, manifest.jsonl

Nothing in the repositories is changed or deleted: deduplication only decides which copy of a
vendored file is *read*. Each document is framed as

    # file: <repo>/<relative path>\\n<text><|endoftext|>

so the model learns what kind of file it is writing. Verified Soup trajectories
(`python -m kgirl.harness soup export-sft out.jsonl`, active-only) can be added as chat-formatted
documents with --sft.

The sampler is stateless: the windows of step n are a pure function of (seed, split, n), so a
resumed run sees exactly the batches an uninterrupted run would have seen.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from . import chaos
from .tokenizer import EOT, Tokenizer

TEXT_EXTS = frozenset((
    ".py", ".pyi", ".md", ".rst", ".txt", ".jl", ".rs", ".go", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx",
    ".c", ".h", ".cc", ".cpp", ".hpp", ".java", ".kt", ".swift", ".zig", ".sh", ".toml", ".yaml", ".yml",
    ".ini", ".cfg", ".sql", ".html", ".css", ".tex", ".json"))
SKIP_DIRS = frozenset((".git", "node_modules", "__pycache__", ".venv", "venv", "env", ".mypy_cache",
                       ".pytest_cache", ".tox", "dist", "build", "site-packages", ".ipynb_checkpoints",
                       "outputs", "runs", ".kgirl"))
MAX_BYTES = 1 << 20
MAX_JSON_BYTES = 64 << 10


@dataclass(frozen=True)
class Document:
    repo: str
    path: str
    sha: str
    text: str

    def framed(self) -> str:
        return f"# file: {self.repo}/{self.path}\n{self.text}{EOT}"


def _is_text(raw: bytes) -> str | None:
    if b"\x00" in raw[:8192]:
        return None
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    lines = text.count("\n") + 1
    if len(text) / lines > 400:                     # minified or one-line data dumps
        return None
    return text


def iter_documents(roots: Iterable[str | Path], exts: frozenset[str] = TEXT_EXTS,
                   max_bytes: int = MAX_BYTES, skip_dirs: frozenset[str] = SKIP_DIRS) -> Iterator[Document]:
    """Every distinct text document under the roots, in a deterministic order."""
    seen: set[str] = set()
    for root in sorted({Path(r).resolve() for r in roots}):
        repo = root.name
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d not in skip_dirs and not d.startswith(".") or d == ".claude")
            for fn in sorted(filenames):
                p = Path(dirpath) / fn
                ext = p.suffix.lower()
                if ext not in exts or p.is_symlink():
                    continue
                try:
                    size = p.stat().st_size
                    if size == 0 or size > (MAX_JSON_BYTES if ext == ".json" else max_bytes):
                        continue
                    text = _is_text(p.read_bytes())
                except OSError:
                    continue
                if text is None or not text.strip():
                    continue
                sha = hashlib.sha1(text.encode("utf-8")).hexdigest()
                if sha in seen:
                    continue
                seen.add(sha)
                yield Document(repo, p.relative_to(root).as_posix(), sha, text)


def iter_sft(path: str | Path) -> Iterator[Document]:
    """Chat-formatted documents from `soup export-sft` JSONL (instruction, context, output)."""
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        r = json.loads(line)
        ctx = f"\nFiles: {r['context']}" if r.get("context") else ""
        text = f"<|user|>{r['instruction']}{ctx}\n<|assistant|>{r['output']}"
        yield Document("soup", f"sft/{i}", hashlib.sha1(text.encode("utf-8")).hexdigest(), text)


def is_val(sha: str, val_permille: int) -> bool:
    return int(sha[:8], 16) % 1000 < val_permille


def build(roots: Iterable[str | Path], tokenizer: Tokenizer, out_dir: str | Path, val_permille: int = 50,
          sft: str | Path | None = None, log=print) -> dict:
    """Write train.bin / val.bin / meta.json / manifest.jsonl under out_dir; return the meta."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dtype = np.uint16 if tokenizer.vocab_size <= 65536 else np.uint32
    docs = list(iter_documents(roots))
    if sft:
        docs += list(iter_sft(sft))
    if not docs:
        raise ValueError("no documents found under the given roots")
    splits = ["val" if is_val(d.sha, val_permille) else "train" for d in docs]
    if len(docs) > 1 and "val" not in splits:      # tiny corpora: guarantee a validation document
        splits[min(range(len(docs)), key=lambda i: docs[i].sha)] = "val"
    if "train" not in splits:
        splits[0] = "train"
    files = {s: open(out / f"{s}.bin.tmp", "wb") for s in ("train", "val")}
    tokens = {"train": 0, "val": 0}
    nbytes = {"train": 0, "val": 0}
    ndocs = {"train": 0, "val": 0}
    with open(out / "manifest.jsonl.tmp", "w", encoding="utf-8") as man:
        for d, s in zip(docs, splits):
            text = d.framed()
            ids = np.asarray(tokenizer.encode(text), dtype=dtype)
            files[s].write(ids.tobytes())
            tokens[s] += len(ids)
            nbytes[s] += len(text.encode("utf-8"))
            ndocs[s] += 1
            man.write(json.dumps({"repo": d.repo, "path": d.path, "sha": d.sha, "split": s, "tokens": int(len(ids))}) + "\n")
    for s, f in files.items():
        f.close()
        os.replace(out / f"{s}.bin.tmp", out / f"{s}.bin")
    os.replace(out / "manifest.jsonl.tmp", out / "manifest.jsonl")
    meta = {"version": 1, "tokenizer_sha": tokenizer.sha, "vocab_size": tokenizer.vocab_size,
            "dtype": np.dtype(dtype).name, "tokens": tokens, "bytes": nbytes, "docs": ndocs,
            "roots": [str(Path(r).resolve()) for r in roots], "sft": str(sft) if sft else None,
            "val_permille": val_permille}
    tmp = out / "meta.json.tmp"
    tmp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    os.replace(tmp, out / "meta.json")
    log(f"docs train={ndocs['train']} val={ndocs['val']}  tokens train={tokens['train']:,} val={tokens['val']:,}  "
        f"bytes/token={sum(nbytes.values()) / max(1, sum(tokens.values())):.2f}")
    return meta


class TokenData:
    """Memory-mapped token shards plus the stateless chaos window sampler."""

    def __init__(self, data_dir: str | Path):
        self.dir = Path(data_dir)
        self.meta = json.loads((self.dir / "meta.json").read_text(encoding="utf-8"))
        dt = np.dtype(self.meta["dtype"])
        self.split = {s: np.memmap(self.dir / f"{s}.bin", dtype=dt, mode="r") for s in ("train", "val")
                      if (self.dir / f"{s}.bin").stat().st_size > 0}

    def tokens(self, split: str) -> int:
        return len(self.split[split]) if split in self.split else 0

    def bytes_per_token(self, split: str) -> float:
        return self.meta["bytes"][split] / max(1, self.meta["tokens"][split])

    def windows(self, split: str, step: int, batch: int, ctx: int, seed: float) -> tuple[np.ndarray, np.ndarray]:
        """(x, y) int64 arrays of shape (batch, ctx): chaos-chosen windows for this step."""
        data = self.split[split]
        hi = len(data) - ctx - 1
        if hi <= 0:
            raise ValueError(f"{split} has {len(data)} tokens, fewer than ctx+1 = {ctx + 1}")
        u = chaos.uniform(batch, seed, chaos.stream_id("batch", split), start=step * batch)
        offs = np.minimum((u * hi).astype(np.int64), hi - 1)
        x = np.stack([data[o:o + ctx] for o in offs]).astype(np.int64)
        y = np.stack([data[o + 1:o + 1 + ctx] for o in offs]).astype(np.int64)
        return x, y

    def sequential(self, split: str, batch: int, ctx: int, max_tokens: int | None = None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Non-overlapping windows covering the split in order (for evaluation)."""
        data = self.split[split]
        n = (len(data) - 1) // ctx
        if max_tokens:
            n = min(n, max(1, max_tokens // ctx))
        for i in range(0, n, batch):
            rows = range(i, min(n, i + batch))
            x = np.stack([data[r * ctx:(r + 1) * ctx] for r in rows]).astype(np.int64)
            y = np.stack([data[r * ctx + 1:(r + 1) * ctx + 1] for r in rows]).astype(np.int64)
            yield x, y
