"""Byte-level BPE tokenizer, trained on your own code: no external tokenizer, no network.

    text ──split on specials──▶ segments ──PATTERN──▶ pieces ──UTF-8──▶ bytes ──merges by rank──▶ ids

* Every byte 0–255 is a base token, so any string (code, emoji, binary-ish text) round-trips.
* Pieces never cross the PATTERN boundaries: words, digit runs (≤ 3), punctuation runs and
  whitespace runs (indentation stays its own token), as in GPT-2.
* Training is classic BPE over the *unique* pieces with their counts. Pair counts live in a
  dict, and a max-heap with lazy invalidation picks the next merge, so each merge costs time in
  the number of pieces that contain the pair, not in the corpus size. Ties break on the smaller
  (left, right) id pair, so training is deterministic.
* Special tokens (`<|endoftext|>`, chat roles) take the last ids and are matched before
  splitting, so they can never be produced by merges.

The saved JSON carries a `sha`, which data shards and checkpoints record, so a model can never be
fed ids from a different tokenizer by accident.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?[A-Za-z_][A-Za-z0-9_]*| ?[^\W\d_]+| ?\d{1,3}| ?[^\s\w]+|\s+(?!\S)|\s+"""
EOT = "<|endoftext|>"
DEFAULT_SPECIALS = (EOT, "<|pad|>", "<|system|>", "<|user|>", "<|assistant|>")
_CACHE_LIMIT = 1 << 18


class Tokenizer:
    def __init__(self, merges: list[tuple[int, int]], specials: Iterable[str] = DEFAULT_SPECIALS,
                 pattern: str = PATTERN):
        self.merges = [tuple(m) for m in merges]
        self.pattern = pattern
        self._split = re.compile(pattern)
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}
        self.vocab: list[bytes] = [bytes([i]) for i in range(256)]
        for a, b in self.merges:
            self.vocab.append(self.vocab[a] + self.vocab[b])
        self.specials = {s: len(self.vocab) + i for i, s in enumerate(specials)}
        self.special_text = {i: s for s, i in self.specials.items()}
        self._special_re = (re.compile("(" + "|".join(re.escape(s) for s in sorted(self.specials, key=len, reverse=True)) + ")")
                            if self.specials else None)
        self._cache: dict[bytes, list[int]] = {}
        payload = json.dumps({"merges": self.merges, "specials": list(self.specials), "pattern": pattern})
        self.sha = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]

    # ------------------------------------------------------------------ properties

    @property
    def vocab_size(self) -> int:
        return len(self.vocab) + len(self.specials)

    @property
    def eot(self) -> int:
        return self.specials[EOT]

    # ------------------------------------------------------------------ encode / decode

    def _bpe(self, piece: bytes) -> list[int]:
        hit = self._cache.get(piece)
        if hit is not None:
            return hit
        ids = list(piece)
        ranks = self.ranks
        while len(ids) > 1:
            best, rank = None, None
            for pair in zip(ids, ids[1:]):
                r = ranks.get(pair)
                if r is not None and (rank is None or r < rank):
                    best, rank = pair, r
            if best is None:
                break
            new_id = 256 + rank
            out, i, n = [], 0, len(ids)
            while i < n:
                if i < n - 1 and ids[i] == best[0] and ids[i + 1] == best[1]:
                    out.append(new_id)
                    i += 2
                else:
                    out.append(ids[i])
                    i += 1
            ids = out
        if len(self._cache) < _CACHE_LIMIT:
            self._cache[piece] = ids
        return ids

    def encode_ordinary(self, text: str) -> list[int]:
        out: list[int] = []
        for piece in self._split.findall(text):
            out.extend(self._bpe(piece.encode("utf-8")))
        return out

    def encode(self, text: str, allow_special: bool = True) -> list[int]:
        if not allow_special or not self._special_re:
            return self.encode_ordinary(text)
        out: list[int] = []
        for part in self._special_re.split(text):
            if not part:
                continue
            sid = self.specials.get(part)
            out.extend([sid] if sid is not None else self.encode_ordinary(part))
        return out

    def decode_bytes(self, ids: Iterable[int]) -> bytes:
        n = len(self.vocab)
        return b"".join(self.vocab[i] if i < n else self.special_text[i].encode("utf-8") for i in ids)

    def decode(self, ids: Iterable[int]) -> str:
        return self.decode_bytes(ids).decode("utf-8", errors="replace")

    # ------------------------------------------------------------------ persistence

    def to_dict(self) -> dict:
        return {"version": 1, "kind": "kgirl-tauls-bpe", "sha": self.sha, "pattern": self.pattern,
                "specials": list(self.specials), "merges": [list(m) for m in self.merges]}

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict()), encoding="utf-8")
        tmp.replace(path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Tokenizer":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        if d.get("kind") != "kgirl-tauls-bpe":
            raise ValueError(f"{path} is not a kgirl TA-ULS tokenizer")
        tok = cls([tuple(m) for m in d["merges"]], d["specials"], d["pattern"])
        if d.get("sha") and d["sha"] != tok.sha:
            raise ValueError(f"{path}: sha mismatch (file says {d['sha']}, content gives {tok.sha})")
        return tok


def train(texts: Iterable[str], vocab_size: int, specials: Iterable[str] = DEFAULT_SPECIALS,
          min_frequency: int = 2, pattern: str = PATTERN, progress=None) -> Tokenizer:
    """Learn merges until the vocabulary (bytes + merges + specials) reaches `vocab_size`,
    or no pair occurs `min_frequency` times."""
    specials = tuple(specials)
    target = vocab_size - 256 - len(specials)
    if target < 0:
        raise ValueError(f"vocab_size must be at least {256 + len(specials)}")
    split = re.compile(pattern)
    special_re = re.compile("|".join(re.escape(s) for s in specials)) if specials else None
    counts: Counter[bytes] = Counter()
    for text in texts:
        for seg in (special_re.split(text) if special_re else [text]):
            counts.update(p.encode("utf-8") for p in split.findall(seg))

    words: list[list[int]] = []
    freq: list[int] = []
    for piece, c in sorted(counts.items()):               # sorted: deterministic word ids
        if len(piece) > 1:
            words.append(list(piece))
            freq.append(c)
    stats: dict[tuple[int, int], int] = defaultdict(int)
    where: dict[tuple[int, int], set[int]] = defaultdict(set)
    for wi, w in enumerate(words):
        f = freq[wi]
        for pair in zip(w, w[1:]):
            stats[pair] += f
            where[pair].add(wi)
    heap = [(-c, p) for p, c in stats.items()]
    heapq.heapify(heap)

    merges: list[tuple[int, int]] = []
    while len(merges) < target and heap:
        negc, pair = heapq.heappop(heap)
        c = stats.get(pair, 0)
        if c != -negc:                                     # stale entry: re-queue with the live count
            if c > 0:
                heapq.heappush(heap, (-c, pair))
            continue
        if c < min_frequency:
            break
        new_id = 256 + len(merges)
        merges.append(pair)
        a, b = pair
        touched: dict[tuple[int, int], int] = {}
        for wi in sorted(where.pop(pair, ())):
            w = words[wi]
            if len(w) < 2:
                continue
            out, i, n, hit = [], 0, len(w), False
            while i < n:
                if i < n - 1 and w[i] == a and w[i + 1] == b:
                    out.append(new_id)
                    i += 2
                    hit = True
                else:
                    out.append(w[i])
                    i += 1
            if not hit:
                continue
            f = freq[wi]
            for p in zip(w, w[1:]):
                stats[p] -= f
                touched[p] = stats[p]
            for p in zip(out, out[1:]):
                stats[p] += f
                touched[p] = stats[p]
                where[p].add(wi)
            words[wi] = out
        stats.pop(pair, None)
        for p, cnt in touched.items():
            if p == pair:
                continue
            if cnt <= 0:
                stats.pop(p, None)
            else:
                heapq.heappush(heap, (-cnt, p))
        if progress and len(merges) % 500 == 0:
            progress(len(merges), target)
    return Tokenizer(merges, specials, pattern)
