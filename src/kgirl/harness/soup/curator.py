"""Staged, regularized recursive self-improvement (RSI) of the memory pool.

The harness improves itself only through *memory*, never by rewriting its own
code or weights. Each accepted agent run proposes abstractions; they move
through a gate:

    propose ──> staged ──(support >= k, LCB(utility) >= p, validator ok)──> active
                  │                                                         │
                  └─(TTL expired, support < k)──> retired <──(UCB < r, or capacity)

Regularizers (why it cannot run away):
  * staging        — nothing a single run proposes is ever recalled until k
                     independent runs reinforce it
  * decay (γ)      — evidence shrinks toward the Beta prior each step, so stale
                     wins must be re-earned (L2 shrinkage on utility)
  * capacity       — active set is bounded; the weakest by LCB x recency leave
  * lower/upper confidence bounds — promotion needs evidence, retirement too
  * append-only ledger + rollback(ts) — every transition is auditable/reversible

Confidence levels: gate/ledger/decay mechanics are Level 0 (standard bandit and
cache-admission engineering). That curated memories make a small local model
solve *your* tasks better is Level 2 — measure it with `soup stats` over time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from ..util import now
from .pool import Fragment, Soup


@dataclass(frozen=True)
class CurationPolicy:
    min_support: int = 2
    promote_lcb: float = 0.45       # z=-1 lower bound on utility needed to go live
    retire_ucb: float = 0.30        # z=+1 upper bound below which an active fragment is retired
    min_evidence_retire: float = 4.0
    capacity: int = 256
    gamma: float = 0.97
    dedupe_jaccard: float = 0.55
    staged_ttl_days: float = 45.0
    manual_sources: tuple[str, ...] = ("mcp",)  # written by a model: only a person promotes these


@dataclass
class CurationReport:
    promoted: list[int] = field(default_factory=list)
    retired: list[int] = field(default_factory=list)
    expired: list[int] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.promoted or self.retired or self.expired)


Validator = Callable[[Fragment], bool]


class Curator:
    def __init__(self, soup: Soup, policy: CurationPolicy | None = None, validator: Validator | None = None):
        self.soup = soup
        self.policy = policy or CurationPolicy()
        self.validator = validator

    def propose(self, text: str, kind: str = "abstraction", tags: str = "", source: str = "", scope: str = "*",
                data: str | None = None) -> tuple[int, str]:
        """Stage a candidate, or reinforce a near-duplicate that already exists."""
        for frag, sim in self.soup.similar(text, kind=kind, limit=8):
            if sim >= self.policy.dedupe_jaccard and frag.scope in ("*", scope):
                self.soup.reinforce(frag.id, detail=f"near-dup {sim:.2f} from {source}")
                self.soup.credit([frag.id], success=True)
                return frag.id, "reinforced"
        fid = self.soup.add(text, kind=kind, tags=tags, source=source, scope=scope, status="staged", data=data)
        self.soup.credit([fid], success=True)  # born from a verified success: that is its first evidence
        return fid, "staged"

    def step(self) -> CurationReport:
        p = self.policy
        rep = CurationReport()
        self.soup.decay(p.gamma)
        t = now()
        for f in self.soup.all(status="staged"):
            auto = f.source not in p.manual_sources
            if auto and f.support >= p.min_support and f.bound(-1.0) >= p.promote_lcb \
                    and (self.validator is None or self.validator(f)):
                self.soup.set_status(f.id, "active", f"promote support={f.support} lcb={f.bound(-1):.2f}")
                rep.promoted.append(f.id)
            elif t - (f.created or t) > p.staged_ttl_days * 86400 and (not auto or f.support < p.min_support):
                self.soup.set_status(f.id, "retired", "staged ttl expired")
                rep.expired.append(f.id)
        active = self.soup.all(status="active")
        for f in active:
            if f.evidence >= p.min_evidence_retire and f.bound(1.0) < p.retire_ucb:
                self.soup.set_status(f.id, "retired", f"ucb={f.bound(1):.2f} < {p.retire_ucb}")
                rep.retired.append(f.id)
        active = [f for f in active if f.id not in rep.retired]
        if len(active) > p.capacity:
            half = self.soup.half_life or 1.0
            def keep_score(f: Fragment) -> float:
                age = t - (f.last_used or f.created or t)
                return f.bound(-1.0) * (0.5 ** (age / half)) + (1.0 if f.kind == "preference" else 0.0)
            for f in sorted(active, key=keep_score)[: len(active) - p.capacity]:
                self.soup.set_status(f.id, "retired", "capacity")
                rep.retired.append(f.id)
        return rep

    def rollback(self, since: float) -> int:
        """Undo every status transition recorded at or after `since` (newest first)."""
        n = 0
        for row in self.soup.ledger(since=since, limit=100000):
            if row["op"] == "status" and row["fragment_id"] is not None and row["before"]:
                self.soup.set_status(row["fragment_id"], row["before"], f"rollback of seq {row['seq']}")
                n += 1
        return n
